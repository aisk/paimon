"""The agents and background commands an Agent starts, without a UI."""

import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from paimon import agent as agent_module
from paimon import headless, lockfile
from paimon.agent import Agent, JobNotice, TextDelta, ToolEnd, UserInput
from paimon.config import Config
from paimon.session import Session, is_job_message, job_text
from tests.support.agent import make_session, spawning_model, stub_model
from tests.support.turns import FakeCommand, settle


class JobsTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cwd = Path(directory.name)

    def agent(self, **kwargs) -> Agent:
        session = make_session(self.cwd)
        session.append_system_prompt("snapshot")
        agent = Agent.open(cwd=self.cwd, session=session, config=Config(model="test:stub"),
                           **kwargs)
        self.addCleanup(agent.close)
        return agent

    async def turn(self, agent: Agent, text: str | None = "go", model=None) -> list:
        if model is not None:
            agent._cached_model = None
            with patch("paimon.agent.build_model", return_value=model):
                return [event async for event in agent.run(text)]
        return [event async for event in agent.run(text)]

    @staticmethod
    async def until(condition, timeout: float = 5.0) -> None:
        async def poll() -> None:
            while not condition():
                await asyncio.sleep(0.01)
        await asyncio.wait_for(poll(), timeout)


class SpawnAgentTest(JobsTestCase):
    async def test_the_call_returns_at_once_and_the_answer_arrives_as_a_notice(self) -> None:
        agent = self.agent()
        gate = asyncio.Event()
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["check the parser"], gate=gate,
                                               answer="the parser is fine")):
            events = await self.turn(agent)
            end = next(event for event in events if isinstance(event, ToolEnd))
            (job_id,) = agent.jobs
            self.assertIn(f"Started agent {job_id}", end.result)
            self.assertTrue(agent.jobs[job_id].running, "the turn ended with the child still going")
            self.assertEqual(agent.notices, [])

            gate.set()
            await self.until(lambda: bool(agent.notices))

        self.assertEqual(agent.notices, [f"agent {job_id} finished:\nthe parser is fine"])
        self.assertEqual(agent.jobs, {}, "a finished agent leaves the table")

    async def test_the_notice_reaches_the_model_as_a_message_of_its_own(self) -> None:
        agent = self.agent()
        with patch("paimon.agent.build_model", return_value=spawning_model(["check the parser"])):
            await self.turn(agent)
            await self.until(lambda: bool(agent.notices))
            events = await self.turn(agent, None)

        notices = [event for event in events if isinstance(event, JobNotice)]
        self.assertEqual(len(notices), 1)
        self.assertTrue(notices[0].text.endswith("finished:\ndone"))
        delivered = [message for message in agent.history if is_job_message(message)]
        self.assertEqual([job_text(message) for message in delivered], [notices[0].text])
        self.assertEqual(agent.notices, [], "a notice is delivered once")

    async def test_a_notice_that_arrives_mid_turn_is_picked_up_between_steps(self) -> None:
        agent = self.agent()
        with patch("paimon.agent.build_model", return_value=stub_model("glob", '{"pattern": "*"}')):
            run = agent.run("go")
            events = []
            async for event in run:
                events.append(event)
                if isinstance(event, ToolEnd):
                    agent.notices.append("agent a1f2 finished:\nfound it")
        notices = [event for event in events if isinstance(event, JobNotice)]
        self.assertEqual([notice.text for notice in notices], ["agent a1f2 finished:\nfound it"])
        self.assertGreater(events.index(notices[0]),
                           events.index(next(e for e in events if isinstance(e, ToolEnd))))

    async def test_the_child_is_narrowed_hidden_and_releases_its_session(self) -> None:
        agent = self.agent()
        gate = asyncio.Event()
        opened: list[Agent] = []
        real_open = Agent.open

        def spy(*args, **kwargs):
            child = real_open(*args, **kwargs)
            opened.append(child)
            return child

        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["check the parser"], gate=gate)), \
                patch.object(agent_module.Agent, "open", side_effect=spy):
            await self.turn(agent)
            (child,) = opened
            for name in ("spawn_agent", "list_models", "stop_job", "run_background", "read_job",
                         "ask_user", "start_new_session"):
                self.assertNotIn(name, child.toolset)
            self.assertIn("shell", child.toolset)
            self.assertEqual(child.session.parent_id, agent.session.id)
            self.assertNotIn(child.session.id, [s.id for s in Session.list(self.cwd)],
                             "a child stays out of the session listings")
            self.assertTrue(lockfile.held(child.session.path))

            gate.set()
            await self.until(lambda: bool(agent.notices))
        self.assertFalse(lockfile.held(child.session.path))

    async def test_a_long_answer_is_clipped(self) -> None:
        agent = self.agent()
        answer = "x" * (agent_module.ANSWER_LIMIT + 500)
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["check the parser"], answer=answer)):
            await self.turn(agent)
            await self.until(lambda: bool(agent.notices))

        (notice,) = agent.notices
        self.assertLess(len(notice), agent_module.ANSWER_LIMIT + 300)
        self.assertIn("clipped, 500 more chars", notice)
        self.assertIn("paimon log", notice, "and it says where the rest is")

    async def test_a_child_that_fails_says_so(self) -> None:
        agent = self.agent()
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["check the parser"], fail="the model said no")), \
                patch("paimon.agent.retry.is_transient", return_value=False):
            await self.turn(agent)
            await self.until(lambda: bool(agent.notices))

        (notice,) = agent.notices
        self.assertIn("failed", notice)
        self.assertIn("the model said no", notice)
        self.assertEqual(agent.jobs, {})

    async def test_several_children_run_at_once(self) -> None:
        agent = self.agent()
        gate = asyncio.Event()
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["one", "two", "three"], gate=gate)):
            await self.turn(agent)
            self.assertEqual(len(agent.jobs), 3)
            self.assertTrue(all(job.running for job in agent.jobs.values()))
            gate.set()
            await self.until(lambda: len(agent.notices) == 3)

    async def test_the_cap_refuses_one_more(self) -> None:
        agent = self.agent()
        gate = asyncio.Event()
        prompts = [f"task {index}" for index in range(agent_module.MAX_AGENTS + 1)]
        with patch("paimon.agent.build_model", return_value=spawning_model(prompts, gate=gate)):
            events = await self.turn(agent)
            results = [event.result for event in events if isinstance(event, ToolEnd)]
            self.assertEqual(len(agent.jobs), agent_module.MAX_AGENTS)
            self.assertIn(f"limit {agent_module.MAX_AGENTS}", results[-1])
            self.assertIn("stop one before starting another", results[-1])
            gate.set()

    async def test_a_ui_showing_the_child_gets_every_event_and_hears_the_end(self) -> None:
        agent = self.agent()
        shown: list = []
        seen: list = []
        ended: list = []

        async def open_job(job_id: str, job) -> None:
            shown.append((job_id, job.kind, job.label, job.running))

            async def sink(event) -> None:
                seen.append(event)

            job.sink = sink
            job.on_done = lambda: ended.append(job_id)

        agent.open_job = open_job
        await self.turn(agent, model=spawning_model(["check the parser"], answer="fine"))
        await self.until(lambda: bool(agent.notices))

        (job_id, kind, label, running), = shown
        self.assertEqual((kind, label), ("agent", "check the parser"))
        self.assertTrue(running, "it is on screen before its first step, and counts as live")
        self.assertIsInstance(seen[0], UserInput)
        self.assertEqual(seen[0].text, "check the parser", "its prompt opens the transcript")
        self.assertEqual("".join(event.text for event in seen if isinstance(event, TextDelta)),
                         "fine")
        self.assertEqual(ended, [job_id])

    async def test_a_child_that_cannot_be_shown_is_not_started(self) -> None:
        agent = self.agent()
        before = {session.path for session in Session.list(self.cwd, include_children=True)}

        async def open_job(job_id: str, job) -> None:
            raise RuntimeError("all 8 panes are in use")

        agent.open_job = open_job
        answer = await agent._job_tool("spawn_agent", {"prompt": "check the parser"})

        self.assertIn("could not start an agent: all 8 panes are in use", answer)
        self.assertEqual(agent.jobs, {})
        for session in Session.list(self.cwd, include_children=True):
            if session.path not in before:
                self.assertFalse(lockfile.held(session.path), "its session is released")

    async def test_a_running_child_follows_its_parents_mode(self) -> None:
        agent = self.agent(mode="yolo")
        gate = asyncio.Event()
        await self.turn(agent, model=spawning_model(["check"], gate=gate))
        (job,) = agent.jobs.values()
        self.assertEqual(job.agent.mode, "yolo")

        agent.mode = "read"
        self.assertEqual(job.agent.mode, "read",
                         "tightening the mode must not leave the child on the old one")

    async def test_children_inherit_the_turns_tool_budget(self) -> None:
        agent = self.agent()
        seen: list = []
        real_run = Agent.run

        def spy(self, user_input, **kwargs):
            seen.append((user_input, kwargs.get("max_tool_calls")))
            return real_run(self, user_input, **kwargs)

        with patch("paimon.agent.build_model", return_value=spawning_model(["check"])), \
                patch.object(agent_module.Agent, "run", spy):
            [event async for event in agent.run("go", max_tool_calls=7)]
            await self.until(lambda: bool(agent.notices))
        self.assertIn(("check", 7), seen)


class ListModelsTest(JobsTestCase):
    CATALOG = ["openai:gpt-6.1-sol", "test:stub", "zai:glm-5.3"]

    async def test_it_lists_every_signed_in_provider_under_qualified_names(self) -> None:
        agent = self.agent()
        agent.config.providers = {"zai": {"api_key": "k"}}
        with patch("paimon.llm.known_models", return_value=self.CATALOG):
            listing = await agent._job_tool("list_models", {})
        self.assertIn("running on test:stub", listing)
        self.assertEqual(listing.splitlines()[1:3], ["test:stub", "zai:glm-5.3"])
        self.assertNotIn("openai:", listing, "no credential, not offered")

    async def test_a_provider_signed_in_to_later_shows_up_on_the_next_call(self) -> None:
        agent = self.agent()
        with patch("paimon.llm.known_models", return_value=self.CATALOG):
            self.assertNotIn("zai:glm-5.3", await agent._job_tool("list_models", {}))
            agent.config.providers = {"zai": {"api_key": "k"}}
            self.assertIn("zai:glm-5.3", await agent._job_tool("list_models", {}))
            with patch("paimon.chatgpt.signed_in", return_value=True):
                self.assertIn("chatgpt:gpt-6.1-sol", await agent._job_tool("list_models", {}))
            with patch("paimon.antigravity.signed_in", return_value=True):
                self.assertIn("antigravity:claude-sonnet-4-6", await agent._job_tool("list_models", {}))

    async def test_an_agent_is_started_on_the_model_asked_for(self) -> None:
        agent = self.agent()
        gate = asyncio.Event()
        with patch("paimon.agent.build_model", return_value=spawning_model([], gate=gate)):
            await agent._job_tool("spawn_agent", {"prompt": "go", "model": "zai:glm-5.3"})
            (job,) = agent.jobs.values()
            self.assertEqual(job.agent.model_name, "zai:glm-5.3")
            job.task.cancel()
            await settle()

    async def test_a_model_that_cannot_be_built_starts_nothing(self) -> None:
        agent = self.agent()
        with patch("paimon.agent.build_model", side_effect=ValueError("Unknown provider: nope.")):
            answer = await agent._job_tool("spawn_agent", {"prompt": "go", "model": "nope:x"})
        self.assertIn("cannot start an agent on 'nope:x': Unknown provider: nope.", answer)
        self.assertIn("list_models", answer)
        self.assertEqual(agent.jobs, {})


class StopTest(JobsTestCase):
    async def _running(self, agent: Agent) -> tuple[str, asyncio.Event]:
        gate = asyncio.Event()
        await self.turn(agent, model=spawning_model(["check the parser"], gate=gate))
        (job_id,) = agent.jobs
        return job_id, gate

    async def test_stopping_one_child_leaves_the_others(self) -> None:
        agent = self.agent()
        gate = asyncio.Event()
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["one", "two"], gate=gate)):
            await self.turn(agent)
            first, second = agent.jobs
            self.assertEqual(await agent._job_tool("stop_job", {"job_id": first}),
                             f"Stopped agent {first}. It reports nothing back.")
            await settle()
            self.assertEqual(list(agent.jobs), [second])
            self.assertEqual(agent.notices, [], "the model stopped it, so it is not told again")

            gate.set()
            await self.until(lambda: bool(agent.notices))
        self.assertTrue(agent.notices[0].startswith(f"agent {second} finished"))

    async def test_a_stop_the_user_ordered_is_news_to_the_model(self) -> None:
        agent = self.agent()
        changes: list = []
        agent.on_jobs_changed = lambda: changes.append(list(agent.notices))
        with patch("paimon.agent.build_model", return_value=None):
            job_id, _ = await self._running(agent)
            self.assertTrue(agent.stop_job(job_id, by_user=True))
            await settle()

        self.assertEqual(agent.notices, [f"agent {job_id} was stopped by the user"])
        self.assertEqual(changes[-1], agent.notices, "and the UI hears about it")
        self.assertFalse(agent.stop_job(job_id), "nothing is left under that id")

    async def test_a_child_stopped_before_its_first_step_is_still_released(self) -> None:
        agent = self.agent()
        job_id = await agent._spawn("check the parser", None)
        child = agent.jobs[job_id].agent
        self.assertTrue(lockfile.held(child.session.path))

        self.assertTrue(agent.stop_job(job_id))
        await settle()

        self.assertEqual(agent.jobs, {}, "a task that never ran still leaves the table")
        self.assertFalse(lockfile.held(child.session.path))

    async def test_closing_the_parent_takes_its_children_with_it(self) -> None:
        agent = self.agent()
        paths = [session.path for session in Session.list(self.cwd, include_children=True)]
        job_id, _ = await self._running(agent)
        task = agent.jobs[job_id].task
        (child_path,) = [session.path
                         for session in Session.list(self.cwd, include_children=True)
                         if session.path not in paths and session.parent_id]
        self.assertTrue(lockfile.held(child_path))

        agent.close()
        await settle()

        self.assertTrue(task.cancelled())
        self.assertFalse(lockfile.held(child_path), "a stopped child releases its session")
        self.assertEqual(agent.notices, [], "nobody is left to tell")


class ChildConfirmTest(JobsTestCase):
    """A child asks where it is shown, and through its parent where it is not."""

    def setUp(self) -> None:
        # The project sits one level down, so the child's write lands outside
        # it: auto mode holds that, where a write inside would just run.
        super().setUp()
        self.out = self.cwd / "out.txt"
        self.cwd = self.cwd / "project"
        self.cwd.mkdir()

    async def _child_writes(self, agent: Agent) -> None:
        requests = 0

        async def stream(messages, info):
            nonlocal requests
            asked = [part.content for message in messages for part in message.parts
                     if isinstance(part, UserPromptPart)]
            requests += 1
            if "write it" in asked:
                done = any(getattr(part, "tool_name", None) == "write_file"
                           and part.part_kind == "tool-return"
                           for message in messages for part in message.parts)
                if done:
                    yield "written"
                else:
                    yield {0: DeltaToolCall(name="write_file", tool_call_id="w-1", json_args=json.dumps(
                        {"path": "../out.txt", "content": "hi"}))}
            elif requests == 1:
                yield {0: DeltaToolCall(name="spawn_agent", tool_call_id="s-1",
                                        json_args=json.dumps({"prompt": "write it"}))}
            else:
                yield "ok"

        with patch("paimon.agent.build_model",
                   return_value=FunctionModel(stream_function=stream)):
            await self.turn(agent)
            await self.until(lambda: bool(agent.notices))

    async def test_a_ui_showing_the_child_is_the_one_asked(self) -> None:
        asked: list = []

        async def parent_confirm(name: str, args: dict) -> bool:
            asked.append("parent")
            return False

        agent = self.agent(mode="auto", confirm=parent_confirm)

        async def open_job(job_id: str, job) -> None:
            async def confirm(name: str, args: dict) -> bool:
                asked.append((name, args["path"]))
                return True

            job.agent.confirm = confirm

        agent.open_job = open_job
        await self._child_writes(agent)

        self.assertEqual(asked, [("write_file", "../out.txt")])
        self.assertEqual(self.out.read_text(), "hi")

    async def test_a_child_nobody_shows_asks_through_its_parent(self) -> None:
        asked: list = []

        async def confirm(name: str, args: dict) -> bool:
            asked.append(name)
            return False

        agent = self.agent(mode="auto", confirm=confirm)
        await self._child_writes(agent)

        self.assertEqual(asked, ["write_file"])
        self.assertFalse(self.out.exists())

    async def test_with_nobody_to_ask_it_is_denied(self) -> None:
        agent = self.agent(mode="auto")
        await self._child_writes(agent)
        self.assertFalse(self.out.exists())
        self.assertIn("finished", agent.notices[0], "the child carries on and reports")


class BackgroundCommandTest(JobsTestCase):
    async def _start(self, agent: Agent) -> tuple[str, FakeCommand]:
        command = FakeCommand("npm run dev")
        shown: list = []

        async def open_job(job_id: str, job) -> None:
            shown.append((job_id, job.command, job.label))

        agent.open_job = open_job
        with patch("paimon.tools.start_background", return_value=command):
            answer = await agent._job_tool(
                "run_background", {"command": "npm run dev", "description": "dev server"})
        ((job_id, _, _),) = shown
        self.assertIn(f"Started background command {job_id}", answer)
        self.assertEqual(shown, [(job_id, command, "dev server")], "the UI puts it on screen")
        self.assertIn(job_id, agent.jobs)
        return job_id, command

    async def test_output_is_read_incrementally(self) -> None:
        agent = self.agent()
        job_id, command = await self._start(agent)
        command.output.append(b"listening on 3000\n")

        first = await agent._job_tool("read_job", {"job_id": job_id})
        self.assertIn("[running]", first)
        self.assertIn("listening on 3000", first)
        self.assertIn("no output since your last read",
                      await agent._job_tool("read_job", {"job_id": job_id}))
        self.assertIn("listening on 3000",
                      await agent._job_tool("read_job", {"job_id": job_id, "mode": "all"}))

    async def test_an_exit_is_delivered_and_the_output_stays_readable(self) -> None:
        agent = self.agent()
        job_id, command = await self._start(agent)
        command.output.append(b"built\n")
        command.exit(2)
        await settle()

        self.assertEqual(agent.notices, [f"command {job_id} exited (code 2)"])
        self.assertIn("[exited, code 2]", await agent._job_tool("read_job", {"job_id": job_id}))

    async def test_a_stop_the_model_ordered_is_not_reported_back(self) -> None:
        agent = self.agent()
        job_id, command = await self._start(agent)
        self.assertIn("Stopped", await agent._job_tool("stop_job", {"job_id": job_id}))
        await settle()

        self.assertTrue(command.killed)
        self.assertEqual(agent.notices, [])
        self.assertIn("already ended", await agent._job_tool("stop_job", {"job_id": job_id}))

    async def test_a_tab_the_user_closed_is_reported(self) -> None:
        agent = self.agent()
        job_id, command = await self._start(agent)
        command.kill()  # what closing the tab does
        await settle()
        self.assertEqual(agent.notices, [f"command {job_id} was stopped"])

    async def test_a_command_that_cannot_be_shown_is_killed(self) -> None:
        agent = self.agent()
        command = FakeCommand()

        async def open_job(job_id: str, job) -> None:
            raise RuntimeError("all 8 panes are in use")

        agent.open_job = open_job
        with patch("paimon.tools.start_background", return_value=command):
            answer = await agent._job_tool(
                "run_background", {"command": "npm run dev", "description": "dev"})

        self.assertIn("could not start the command: all 8 panes are in use", answer)
        self.assertTrue(command.killed, "or it would outlive the app in its own group")
        self.assertEqual(agent.jobs, {})

    async def test_agents_at_their_cap_leave_room_for_a_command(self) -> None:
        agent = self.agent()
        gate = asyncio.Event()
        prompts = [f"task {index}" for index in range(agent_module.MAX_AGENTS)]
        await self.turn(agent, model=spawning_model(prompts, gate=gate))
        self.assertEqual(len(agent.jobs), agent_module.MAX_AGENTS)

        job_id, _ = await self._start(agent)
        self.assertEqual(agent.jobs[job_id].kind, "command")
        gate.set()

    async def test_an_agent_has_no_output_to_read(self) -> None:
        agent = self.agent()
        gate = asyncio.Event()
        await self.turn(agent, model=spawning_model(["check"], gate=gate))
        (job_id,) = agent.jobs
        self.assertIn("is an agent", await agent._job_tool("read_job", {"job_id": job_id}))


class HeadlessTest(JobsTestCase):
    """-p has nobody to come back later, so the run waits for its agents."""

    def test_the_run_waits_for_its_agents_and_lets_the_model_react(self) -> None:
        out = io.StringIO()
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["check the parser"], answer="parser is fine")), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = headless.run(prompt="go", piped="", cwd=self.cwd, mode="yolo", session=None,
                                output_format="json", config=Config(model="test:stub"))

        events = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(code, 0)
        (notice,) = [event for event in events if event["type"] == "job"]
        self.assertTrue(notice["text"].endswith("finished:\nparser is fine"))
        self.assertEqual(events[-1]["type"], "result")
        self.assertEqual(events[-1]["subtype"], "success")
        self.assertEqual(events[-1]["text"], "ok\n\nok",
                         "one block per turn: before the agent reported, and after")
        for session in Session.list(self.cwd, include_children=True):
            self.assertFalse(lockfile.held(session.path))


if __name__ == "__main__":
    unittest.main()
