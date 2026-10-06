import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic_ai.messages import (
    ModelRequest,
    ToolReturnPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from paimon import tools
from paimon.agent import (
    Agent,
    SessionHandoff,
    TextDelta,
    TodosUpdate,
    ToolBudgetExhausted,
    ToolEnd,
    ToolStart,
    TurnEnd,
    replay_events,
)
from paimon.config import Config
from paimon.review import MAX_BLOCKS, Verdict
from tests.support.agent import make_session, open_agent, stub_model


class HistoryToolWiringTest(unittest.IsolatedAsyncioTestCase):
    async def test_search_history_reaches_the_agents_own_session_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")

            with patch("paimon.agent.build_model",
                       return_value=stub_model("search_history", '{"query": "avocado"}')):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                events = [event async for event in agent.run("the codeword is avocado")]

            end = next(event for event in events if isinstance(event, ToolEnd))
            self.assertIn("matching part", end.result)
            self.assertIn("avocado", end.result)


class PermissionModeTest(unittest.IsolatedAsyncioTestCase):
    """The agent consults the gate per tool call and puts a held one to whoever
    decides. The gate's full decision table is covered in test_tools_permissions."""

    HELD = '{"path": "../a.txt", "content": "hi"}'  # a write outside the working directory

    def _cwd(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = Path(tmp.name).resolve() / "project"
        cwd.mkdir()
        return cwd

    async def _run_tool_turn(self, agent: Agent, name: str, arguments: str) -> ToolEnd:
        agent._cached_model = None  # the agent caches per config; each turn gets a fresh stub
        with patch("paimon.agent.build_model", return_value=stub_model(name, arguments)):
            events = [event async for event in agent.run("go")]
        return next(event for event in events if isinstance(event, ToolEnd))

    async def test_auto_mode_writes_in_cwd_without_asking(self) -> None:
        cwd = self._cwd()
        confirm = AsyncMock(return_value=False)
        agent = open_agent(cwd, confirm=confirm, mode="auto")

        with patch("paimon.review.judge", AsyncMock()) as judge:
            end = await self._run_tool_turn(agent, "write_file", '{"path": "a.txt", "content": "hi"}')

        judge.assert_not_awaited()
        confirm.assert_not_awaited()
        self.assertFalse(end.denied)
        self.assertEqual((cwd / "a.txt").read_text(), "hi")

    async def test_mode_switch_applies_to_the_next_tool_call(self) -> None:
        cwd = self._cwd()
        confirm = AsyncMock(return_value=True)
        agent = open_agent(cwd, confirm=confirm, mode="read")

        end = await self._run_tool_turn(agent, "write_file", '{"path": "a.txt", "content": "hi"}')
        self.assertTrue(end.denied)
        self.assertEqual(end.result, tools.READ_DENIAL)
        self.assertFalse((cwd / "a.txt").exists())

        agent.mode = "yolo"
        end = await self._run_tool_turn(agent, "write_file", '{"path": "a.txt", "content": "hi"}')
        confirm.assert_not_awaited()
        self.assertFalse(end.denied)
        self.assertEqual((cwd / "a.txt").read_text(), "hi")

    @unittest.skipIf(os.name == "nt", "nothing is auto-allowed under cmd.exe")
    async def test_read_only_commands_run_unless_the_config_says_strict(self) -> None:
        """The safe_commands setting has to reach run_tool, not just the gate."""
        agent = open_agent(self._cwd(), mode="read")

        end = await self._run_tool_turn(agent, "shell", '{"command": "pwd"}')
        self.assertFalse(end.denied)

        agent.config.safe_commands = False
        end = await self._run_tool_turn(agent, "shell", '{"command": "pwd"}')
        self.assertTrue(end.denied)

    async def test_auto_mode_runs_what_the_reviewer_allows(self) -> None:
        cwd = self._cwd()
        confirm = AsyncMock(return_value=False)
        agent = open_agent(cwd, confirm=confirm, mode="auto")

        with patch("paimon.review.judge", AsyncMock(return_value=Verdict(True))) as judge:
            end = await self._run_tool_turn(agent, "write_file", self.HELD)

        _, history, name, args, _ = judge.await_args.args
        self.assertIs(history, agent.history)
        self.assertEqual((name, args["path"]), ("write_file", "../a.txt"))
        confirm.assert_not_awaited()
        self.assertFalse(end.denied)
        self.assertEqual((cwd.parent / "a.txt").read_text(), "hi")

    async def test_a_blocked_call_is_denied_with_the_reason(self) -> None:
        cwd = self._cwd()
        confirm = AsyncMock(return_value=True)
        agent = open_agent(cwd, confirm=confirm, mode="auto")

        with patch("paimon.review.judge", AsyncMock(return_value=Verdict(False, "Not asked for."))):
            end = await self._run_tool_turn(agent, "write_file", self.HELD)

        confirm.assert_not_awaited()
        self.assertTrue(end.denied)
        self.assertIn("Not asked for.", end.result)
        self.assertFalse((cwd.parent / "a.txt").exists())

    async def test_blocks_in_a_row_end_up_with_the_user(self) -> None:
        cwd = self._cwd()
        confirm = AsyncMock(return_value=True)
        agent = open_agent(cwd, confirm=confirm, mode="auto")

        with patch("paimon.review.judge", AsyncMock(return_value=Verdict(False, "No."))):
            for _ in range(MAX_BLOCKS - 1):
                end = await self._run_tool_turn(agent, "write_file", self.HELD)
                self.assertTrue(end.denied)
            confirm.assert_not_awaited()
            end = await self._run_tool_turn(agent, "write_file", self.HELD)

        confirm.assert_awaited_once()
        self.assertFalse(end.denied, "the user overruled the reviewer")
        self.assertEqual((cwd.parent / "a.txt").read_text(), "hi")

    async def test_an_unavailable_reviewer_falls_back_to_the_user(self) -> None:
        cwd = self._cwd()
        confirm = AsyncMock(return_value=False)
        agent = open_agent(cwd, confirm=confirm, mode="auto")

        with patch("paimon.review.judge", AsyncMock(side_effect=RuntimeError("HTTP 500"))):
            end = await self._run_tool_turn(agent, "write_file", self.HELD)

        confirm.assert_awaited_once()
        self.assertTrue(end.denied)
        self.assertEqual(end.result, tools.USER_DENIAL)

    async def test_an_unavailable_reviewer_with_nobody_to_ask_is_a_denial(self) -> None:
        cwd = self._cwd()
        agent = open_agent(cwd, mode="auto")

        with patch("paimon.review.judge", AsyncMock(side_effect=RuntimeError("HTTP 500"))):
            end = await self._run_tool_turn(agent, "write_file", self.HELD)

        self.assertTrue(end.denied)
        self.assertIn("HTTP 500", end.result)
        self.assertFalse((cwd.parent / "a.txt").exists())

    async def test_the_reviewer_can_be_a_model_of_its_own(self) -> None:
        cwd = self._cwd()
        agent = open_agent(cwd, mode="auto")
        agent.config.review_model = "test:reviewer"
        built = []

        def build(name, *rest):
            built.append(name)
            return stub_model("write_file", self.HELD)

        agent._cached_model = None
        with (patch("paimon.agent.build_model", side_effect=build),
              patch("paimon.review.judge", AsyncMock(return_value=Verdict(True)))):
            [event async for event in agent.run("go")]

        self.assertIn("test:reviewer", built)


class TodosEventShapeTest(unittest.IsolatedAsyncioTestCase):
    async def test_write_todos_yields_only_a_todos_update(self) -> None:
        """No ToolStart/ToolEnd for write_todos, matching what replay produces —
        renderers can then treat every ToolStart/ToolEnd the same way."""
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            arguments = '{"todos": [{"content": "x", "status": "pending"}]}'

            with patch("paimon.agent.build_model", return_value=stub_model("write_todos", arguments)):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                events = [event async for event in agent.run("go")]

            self.assertFalse([e for e in events if isinstance(e, (ToolStart, ToolEnd))])
            todos = next(e for e in events if isinstance(e, TodosUpdate))
            self.assertEqual(todos.todos, [{"content": "x", "status": "pending"}])
            self.assertEqual(agent.todos, todos.todos)

    async def test_looping_write_todos_stops_at_the_tool_budget(self) -> None:
        """HEADLESS-1: agent-handled tools count against max_tool_calls, and
        the refusal is persisted explicitly rather than as an interrupted
        placeholder."""
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            arguments = '{"todos": [{"content": "x", "status": "pending"}]}'

            async def stream(messages, info):
                # Calls write_todos on every request, forever.
                yield {0: DeltaToolCall(name="write_todos", json_args=arguments,
                                        tool_call_id=f"call-{len(messages)}")}

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                events = [event async for event in agent.run("go", max_tool_calls=1)]

            self.assertEqual(len([e for e in events if isinstance(e, TodosUpdate)]), 1)
            budget = [e for e in events if isinstance(e, ToolBudgetExhausted)]
            self.assertEqual([b.limit for b in budget], [1])
            last = session.messages()[-1]
            self.assertIn("Not executed", last.parts[0].content)
            self.assertIn("max_tool_calls=1", last.parts[0].content)

    async def test_zero_budget_refuses_every_tool_including_agent_handled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            arguments = '{"todos": [{"content": "x", "status": "pending"}]}'

            with patch("paimon.agent.build_model",
                       return_value=stub_model("write_todos", arguments)):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                events = [event async for event in agent.run("go", max_tool_calls=0)]

            self.assertFalse([e for e in events if isinstance(e, TodosUpdate)])
            self.assertEqual(agent.todos, [])
            self.assertTrue([e for e in events if isinstance(e, ToolBudgetExhausted)])

    async def test_malformed_agent_handled_args_are_a_tool_error(self) -> None:
        """TOOLS-1: agent-handled tools go through the same validation
        contract; missing arguments never raise out of the loop."""
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            with patch("paimon.agent.build_model", return_value=stub_model("read_job", "{}")):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                events = [event async for event in agent.run("go")]

            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertIn("invalid arguments for read_job", end.result)
            self.assertIn("job_id", end.result)
            self.assertTrue([e for e in events if isinstance(e, TurnEnd)])

    async def test_malformed_todos_are_a_tool_error_the_turn_survives(self) -> None:
        """The model writes these arguments, so the wrong shape must reach it as a
        tool error instead of raising out of the agent loop and killing the turn."""
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")

            with patch("paimon.agent.build_model",
                       return_value=stub_model("write_todos", '{"todos": "oops"}')):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                events = [event async for event in agent.run("go")]

            self.assertFalse([e for e in events if isinstance(e, TodosUpdate)])
            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertIn("must be a valid list", end.result)
            self.assertEqual(agent.todos, [], "the bad list is not adopted")
            self.assertTrue([e for e in events if isinstance(e, TurnEnd)],
                            "the model gets the error and finishes the turn")
            # ...and a resume of that session shows the same failed call.
            replayed = replay_events(session.messages())
            self.assertFalse([e for e in replayed if isinstance(e, TodosUpdate)])
            self.assertIn("must be a valid list",
                          next(e for e in replayed if isinstance(e, ToolEnd)).result)


class AgentToolsetTest(unittest.IsolatedAsyncioTestCase):
    """A per-agent toolset narrows both what the model sees and what may run."""

    async def test_only_the_toolset_is_offered_and_runnable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"), mode="yolo",
                               toolset={"read_file": tools.REGISTRY["read_file"]})

            offered: list[list[str]] = []
            requests = 0

            async def stream(messages, info):
                nonlocal requests
                requests += 1
                offered.append([tool.name for tool in info.function_tools])
                if requests == 1:
                    yield {0: DeltaToolCall(name="shell", json_args='{"command": "echo hi"}',
                                            tool_call_id="call-1")}
                else:
                    yield "done"

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                events = [event async for event in agent.run("go")]

            self.assertEqual(offered[0], ["read_file"])
            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertIn("unknown tool", end.result)
            self.assertIsInstance(events[-1], TurnEnd)


class AskUserTest(unittest.IsolatedAsyncioTestCase):
    """ask_user hands the question to the ask hook and feeds the answer back
    as the tool result; without a hook it fails with a readable error."""

    @staticmethod
    async def _run(agent: Agent, arguments: str) -> list:
        with patch("paimon.agent.build_model", return_value=stub_model("ask_user", arguments)):
            return [event async for event in agent.run("go")]

    async def test_answer_becomes_the_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            ask = AsyncMock(return_value="Postgres")
            agent = open_agent(cwd, ask=ask, mode="read")

            events = await self._run(agent, '{"question": "Which db?", "options": ["Postgres", "SQLite"]}')

            ask.assert_awaited_once_with("Which db?", ["Postgres", "SQLite"])
            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertFalse(end.denied)
            self.assertEqual(end.result, "User answered: Postgres")
            self.assertIsInstance(events[-1], TurnEnd)
            returns = [part for message in agent.session.messages()
                       if isinstance(message, ModelRequest)
                       for part in message.parts if isinstance(part, ToolReturnPart)]
            self.assertEqual(returns[-1].content, "User answered: Postgres")

    async def test_dismissal_tells_the_model_to_carry_on(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            agent = open_agent(cwd, ask=AsyncMock(return_value=None), mode="read")

            events = await self._run(agent, '{"question": "Which db?"}')

            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertIn("dismissed", end.result)
            self.assertTrue([e for e in events if isinstance(e, TextDelta)], "the turn continued")

    async def test_without_ask_hook_the_call_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            agent = open_agent(cwd, mode="yolo")

            events = await self._run(agent, '{"question": "Which db?"}')

            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertTrue(end.result.startswith("Error:"))
            self.assertIn("continue", end.result)

    async def test_blank_question_is_rejected_before_asking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            ask = AsyncMock(return_value="x")
            agent = open_agent(cwd, ask=ask, mode="read")

            events = await self._run(agent, '{"question": "  "}')

            ask.assert_not_awaited()
            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertEqual(end.result, "Error: question is required.")


class SessionHandoffTest(unittest.IsolatedAsyncioTestCase):
    """start_new_session ends the turn on approval without another model
    request; without a confirm hook it is denied even in yolo mode."""

    @staticmethod
    async def _run(agent: Agent, arguments: str = '{"prompt": "next phase"}') -> list:
        with patch("paimon.agent.build_model",
                   return_value=stub_model("start_new_session", arguments)):
            return [event async for event in agent.run("go")]

    async def test_approval_ends_the_turn_with_a_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            confirm = AsyncMock(return_value=True)
            agent = open_agent(cwd, confirm=confirm, mode="yolo")

            events = await self._run(agent)

            confirm.assert_awaited_once()
            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertFalse(end.denied)
            self.assertIsInstance(events[-1], SessionHandoff)
            self.assertEqual(events[-1].prompt, "next phase")
            self.assertFalse([e for e in events if isinstance(e, (TurnEnd, TextDelta))],
                             "no second model request after the handoff")
            returns = [part for message in agent.session.messages()
                       if isinstance(message, ModelRequest)
                       for part in message.parts if isinstance(part, ToolReturnPart)]
            self.assertIn("Handoff accepted", returns[-1].content)

    async def test_denial_continues_the_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            confirm = AsyncMock(return_value=False)
            agent = open_agent(cwd, confirm=confirm, mode="read")

            events = await self._run(agent)

            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertTrue(end.denied)
            self.assertFalse([e for e in events if isinstance(e, SessionHandoff)])
            self.assertTrue([e for e in events if isinstance(e, TextDelta)])
            self.assertIsInstance(events[-1], TurnEnd)

    async def test_without_confirm_hook_denied_even_in_yolo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            agent = open_agent(cwd, mode="yolo")

            events = await self._run(agent)

            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertTrue(end.denied)
            self.assertFalse([e for e in events if isinstance(e, SessionHandoff)])
            self.assertIsInstance(events[-1], TurnEnd)

    async def test_empty_prompt_is_an_error_without_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            confirm = AsyncMock(return_value=True)
            agent = open_agent(cwd, confirm=confirm, mode="read")

            events = await self._run(agent, '{"prompt": "  "}')

            confirm.assert_not_awaited()
            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertIn("prompt is required", end.result)
            self.assertFalse([e for e in events if isinstance(e, SessionHandoff)])
            self.assertIsInstance(events[-1], TurnEnd)


class JobToolsTest(unittest.IsolatedAsyncioTestCase):
    """The job tools as the agent loop sees them; test_agent_jobs covers the jobs."""

    async def _call(self, agent: Agent, name: str, arguments: str) -> ToolEnd:
        with patch("paimon.agent.build_model", return_value=stub_model(name, arguments)):
            events = [event async for event in agent.run("do it")]
        return next(event for event in events if isinstance(event, ToolEnd))

    async def test_a_background_command_needs_somewhere_to_show(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with open_agent(Path(directory)) as agent:
                end = await self._call(agent, "run_background",
                                       '{"command": "npm run dev", "description": "dev"}')
            self.assertIn("only work in the interactive UI", end.result)

    async def test_a_narrowed_toolset_disables_them_entirely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            toolset = tools.without(tools.REGISTRY, tools.SUBAGENT_DENIED)
            with open_agent(Path(directory), toolset=toolset) as agent:
                end = await self._call(agent, "spawn_agent", '{"prompt": "go"}')
                names = [schema["function"]["name"] for schema in agent.tool_schemas]
            self.assertIn("unknown tool", end.result)
            self.assertNotIn("spawn_agent", names, "and the model is never offered it")

    async def test_an_id_that_names_nothing_is_an_error_the_model_can_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with open_agent(Path(directory)) as agent:
                for name in ("read_job", "stop_job"):
                    agent._cached_model = None
                    end = await self._call(agent, name, '{"job_id": "a1f2"}')
                    self.assertIn("no agent or background command a1f2", end.result)
