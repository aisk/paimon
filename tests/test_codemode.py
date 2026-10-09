import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from textual.widgets import Static
from typing_extensions import TypedDict

from paimon import codemode, headless, tools
from paimon.agent import Agent, ToolBudgetExhausted, ToolEnd, ToolStart, replay_events
from paimon.config import Config
from paimon.session import Session
from paimon.transcript import EventRenderer
from tests.support.agent import make_session, session_records
from tests.support.app import AppTestCase

WAIT = 5.0  # a script that hangs fails its test instead of the suite


class _NoArgs(TypedDict):
    pass


def _script_model(*scripts: str) -> FunctionModel:
    """Model stub: one run_code call per script, a request each, then text."""
    requests = 0

    async def stream(messages, info: AgentInfo):
        nonlocal requests
        requests += 1
        if requests <= len(scripts):
            yield {0: DeltaToolCall(name="run_code", tool_call_id=f"call-{requests}",
                                    json_args=json.dumps({"code": scripts[requests - 1]}))}
        else:
            yield "done"

    return FunctionModel(stream_function=stream)


class CodeModeTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve() / "project"
        self.cwd.mkdir()

    def _agent(self, *, code_mode: bool = True, extra: dict | None = None, **kwargs) -> Agent:
        session = make_session(self.cwd)
        session.append_system_prompt("snapshot")
        toolset = {**tools.REGISTRY, **extra} if extra else None
        agent = Agent.open(cwd=self.cwd, session=session, toolset=toolset,
                           config=Config(model="test:stub", code_mode=code_mode), **kwargs)
        self.addCleanup(agent.close)
        return agent

    async def _events(self, agent: Agent, *scripts: str, **kwargs) -> list:
        agent._cached_model = None
        with patch("paimon.agent.build_model", return_value=_script_model(*scripts)):
            return await asyncio.wait_for(self._collect(agent, **kwargs), WAIT)

    @staticmethod
    async def _collect(agent: Agent, **kwargs) -> list:
        return [event async for event in agent.run("go", **kwargs)]

    async def _run(self, agent: Agent, script: str, **kwargs) -> ToolEnd:
        events = await self._events(agent, script, **kwargs)
        return next(event for event in events if isinstance(event, ToolEnd))


class ScriptTest(CodeModeTestCase):
    async def test_sequential_calls_chain_and_only_the_output_comes_back(self) -> None:
        (self.cwd / "a.txt").write_text("b.txt\n")
        (self.cwd / "b.txt").write_text("secret payload\nneedle here\n")
        agent = self._agent()

        end = await self._run(agent, (
            "first = await read_file(path='a.txt')\n"
            "name = first.split()[-1]\n"
            "second = await read_file(path=name)\n"
            "print('read', name)\n"
            "[line for line in second.splitlines() if 'needle' in line]\n"))

        header, _, body = end.result.partition("\n")
        self.assertRegex(header, r"^Script completed in \d+\.\d\ds$")
        self.assertEqual(body, 'read b.txt\n[\n "2  needle here"\n]')
        self.assertNotIn("secret payload", end.result)
        self.assertEqual([(call["name"], call["detail"], call["status"]) for call in end.calls],
                         [("read_file", "a.txt", "ok"), ("read_file", "b.txt", "ok")])

    async def test_gathered_calls_run_concurrently(self) -> None:
        # Each call waits for the other to have started, so run one after the
        # other they would never finish.
        started = {"left": asyncio.Event(), "right": asyncio.Event()}

        def waiting(mine: str, other: str) -> tools.Tool:
            async def run(args, cwd, mode, ctx):
                started[mine].set()
                await started[other].wait()
                return mine
            return tools.Tool(description="Wait for the other.", params=_NoArgs, run=run)

        agent = self._agent(extra={"left": waiting("left", "right"),
                                   "right": waiting("right", "left")})

        end = await self._run(agent, "import asyncio\nawait asyncio.gather(left(), right())\n")

        self.assertIn('[\n "left",\n "right"\n]', end.result)
        self.assertEqual([call["status"] for call in end.calls], ["ok", "ok"])

    async def test_a_failing_script_keeps_what_it_printed(self) -> None:
        agent = self._agent()

        end = await self._run(agent, "print('partial')\nitems = [1]\nitems[3]\nprint('never')\n")

        self.assertRegex(end.result, r"^Script failed after ")
        self.assertIn("partial", end.result)
        self.assertNotIn("never", end.result)
        self.assertIn("line 3", end.result)
        self.assertIn("IndexError", end.result)
        self.assertEqual(agent.history[-2].parts[0].outcome, "failed")

    async def test_a_script_that_does_not_type_check_calls_nothing(self) -> None:
        agent = self._agent()

        end = await self._run(agent, (
            "await write_file(path='a.txt', content='x')\n"
            "await read_file(file='a.txt')\n"))

        self.assertRegex(end.result, r"^Script failed after ")
        self.assertIn("script.py:2", end.result)
        self.assertEqual(end.calls, [])
        self.assertFalse((self.cwd / "a.txt").exists())

    async def test_output_is_capped_and_the_error_survives_the_cap(self) -> None:
        agent = self._agent()

        end = await self._run(agent, "print('x' * 50000)\nraise ValueError('the reason')\n")

        self.assertLess(len(end.result), tools.MAX_OUTPUT + 100)
        self.assertIn("more chars)", end.result)
        self.assertTrue(end.result.endswith("ValueError: the reason"))

    async def test_the_sandbox_reaches_nothing_but_the_tools(self) -> None:
        (self.cwd / "a.txt").write_text("x")
        agent = self._agent()
        for script in ("open('a.txt').read()", "import os\nos.environ['HOME']",
                       "import time\ntime.time()", "await spawn_agent(prompt='x')",
                       "await run_code(code='1')"):
            with self.subTest(script=script):
                end = await self._run(agent, script)
                self.assertRegex(end.result, r"^Script failed after ")


class GatingTest(CodeModeTestCase):
    async def test_a_denied_call_raises_in_the_script(self) -> None:
        asked = []

        async def confirm(name: str, args: dict) -> bool:
            asked.append((name, args["path"]))
            return False

        agent = self._agent(mode="auto", confirm=confirm)

        with patch("paimon.review.judge", side_effect=RuntimeError("down")):
            end = await self._run(agent, (
                "await write_file(path='in.txt', content='ok')\n"
                "try:\n"
                "    await write_file(path='../out.txt', content='no')\n"
                "except Exception as exc:\n"
                "    print('refused:', exc)\n"))

        self.assertIn(f"refused: {tools.USER_DENIAL}", end.result)
        self.assertEqual(asked, [("write_file", "../out.txt")])
        self.assertTrue((self.cwd / "in.txt").exists())
        self.assertFalse((self.cwd.parent / "out.txt").exists())
        self.assertEqual([call["status"] for call in end.calls], ["ok", "denied"])

    async def test_read_mode_refuses_a_write_from_a_script(self) -> None:
        agent = self._agent(mode="read")

        end = await self._run(agent, "await write_file(path='a.txt', content='x')\n")

        self.assertRegex(end.result, r"^Script failed after ")
        self.assertIn("read mode only allows reading", end.result)
        self.assertFalse((self.cwd / "a.txt").exists())

    async def test_invalid_arguments_raise_without_running_the_tool(self) -> None:
        agent = self._agent()

        # Parsed at run time, so the type check cannot see the wrong type.
        end = await self._run(agent, "import json\n"
                                     "args = json.loads('{\"path\": \"a.txt\", \"content\": 1}')\n"
                                     "await write_file(**args)\n")

        self.assertIn("invalid arguments for write_file", end.result)
        self.assertEqual([call["status"] for call in end.calls], ["error"])
        self.assertFalse((self.cwd / "a.txt").exists())

    async def test_parallel_calls_ask_one_at_a_time(self) -> None:
        asking = peak = 0

        async def confirm(name: str, args: dict) -> bool:
            nonlocal asking, peak
            asking += 1
            peak = max(peak, asking)
            await asyncio.sleep(0.01)
            asking -= 1
            return True

        agent = self._agent(mode="auto", confirm=confirm)

        with patch("paimon.review.judge", side_effect=RuntimeError("down")):
            end = await self._run(agent, (
                "import asyncio\n"
                "await asyncio.gather(*[write_file(path='../' + name, content='x')\n"
                "                       for name in ('a.txt', 'b.txt', 'c.txt')])\n"))

        self.assertRegex(end.result, r"^Script completed in ")
        self.assertEqual(peak, 1)
        self.assertEqual(len(list(self.cwd.parent.glob("?.txt"))), 3)


class BudgetTest(CodeModeTestCase):
    async def test_nested_calls_count_and_exhaustion_ends_the_turn(self) -> None:
        (self.cwd / "a.txt").write_text("x")
        agent = self._agent()

        # The script itself is the first call, so two reads fit and one does not.
        events = await self._events(agent, (
            "for index in range(3):\n"
            "    await read_file(path='a.txt')\n"
            "    print('read', index)\n"), max_tool_calls=3)

        end = next(event for event in events if isinstance(event, ToolEnd))
        self.assertIn("read 1", end.result)
        self.assertNotIn("read 2", end.result)
        self.assertIn("Not executed: the run reached its tool call budget (max_tool_calls=3).",
                      end.result)
        self.assertEqual([call["status"] for call in end.calls], ["ok", "ok", "error"])
        self.assertIsInstance(events[-1], ToolBudgetExhausted)
        self.assertEqual(session_records(agent.session)[-1]["outcome"], "max_tool_calls")

    async def test_a_script_within_budget_leaves_the_turn_running(self) -> None:
        (self.cwd / "a.txt").write_text("x")
        agent = self._agent()

        events = await self._events(agent, "await read_file(path='a.txt')\n", max_tool_calls=2)

        self.assertFalse(any(isinstance(event, ToolBudgetExhausted) for event in events))


class CancellationTest(CodeModeTestCase):
    def _hanging(self, state: dict) -> tools.Tool:
        async def run(args, cwd, mode, ctx):
            state["started"].set()
            try:
                await asyncio.Event().wait()
            finally:
                state["cancelled"] = True
        return tools.Tool(description="Never return.", params=_NoArgs, run=run)

    async def test_interrupt_cancels_running_calls_and_keeps_the_placeholder(self) -> None:
        state = {"started": asyncio.Event(), "cancelled": False}
        agent = self._agent(extra={"hang": self._hanging(state)})

        turn = asyncio.ensure_future(self._events(agent, "await hang()\n"))
        await asyncio.wait_for(state["started"].wait(), WAIT)
        turn.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await turn

        self.assertTrue(state["cancelled"])
        slot = agent.history[-1].parts[0]
        self.assertEqual((slot.content, slot.outcome), ("Interrupted by user.", "interrupted"))
        self.assertEqual(session_records(agent.session)[-1]["outcome"], "interrupted")

    async def test_a_call_the_script_never_awaited_ends_with_it(self) -> None:
        state = {"started": asyncio.Event(), "cancelled": False}

        def wait_for_start() -> tools.Tool:
            async def run(args, cwd, mode, ctx):
                await state["started"].wait()
                return "started"
            return tools.Tool(description="Wait.", params=_NoArgs, run=run)

        agent = self._agent(extra={"hang": self._hanging(state), "wait_for_start": wait_for_start()})

        end = await self._run(agent, "pending = hang()\nawait wait_for_start()\n'over'\n")

        self.assertIn("over", end.result)
        self.assertTrue(state["cancelled"])
        self.assertEqual([(call["name"], call["status"]) for call in end.calls],
                         [("hang", "cancelled"), ("wait_for_start", "ok")])


class ReviewFixesTest(CodeModeTestCase):
    async def test_a_value_small_in_the_sandbox_is_not_written_out_whole(self) -> None:
        import tracemalloc

        tracemalloc.start()
        try:
            end = await self._run(self._agent(), "x = ['y' * 100]\nfor i in range(40):\n    x = [x, x]\nx")
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        self.assertTrue(end.result.startswith("Script completed"))
        self.assertIn("... (truncated)", end.result)
        self.assertLessEqual(len(end.result), tools.MAX_OUTPUT + 100)
        self.assertLess(peak, 20 * 1024 * 1024)

    def test_rendering_matches_json_until_it_is_cut(self) -> None:
        value = {"a": [1, "é", None, {"b": []}], "c": {}, 3: (1, 2)}
        self.assertEqual(codemode._render(value),
                         (json.dumps(value, ensure_ascii=False, indent=1), False))
        text, clipped = codemode._render([["x" * 10] * 100] * 100, limit=500)
        self.assertTrue(clipped)
        self.assertLess(len(text), 600)

    async def test_a_command_that_fails_is_not_recorded_as_ok(self) -> None:
        end = await self._run(self._agent(), "await shell(command='exit 3')")

        self.assertEqual([(call["name"], call["status"]) for call in end.calls], [("shell", "error")])

    def test_failure_reads_the_status_a_command_ends_with(self) -> None:
        self.assertIsNone(tools.failure("out\n(exit code 0)"))
        self.assertEqual(tools.failure("out\n(exit code 1)"), "exit 1")
        self.assertEqual(tools.failure("out\n(timed out after 120s)"), "timed out")
        self.assertEqual(tools.failure("Error: unknown tool 'x'"), "failed")
        self.assertIsNone(tools.failure("1:Error handling\n"))

    async def test_headless_counts_a_call_denied_inside_a_script(self) -> None:
        from paimon.headless import ResultRenderer

        agent = self._agent(mode="read")
        renderer = ResultRenderer(agent.config)
        for event in await self._events(agent, "await write_file(path='a.txt', content='x')"):
            await renderer.handle(event)

        self.assertEqual(renderer._denied, 1)
        self.assertFalse((self.cwd / "a.txt").exists())

    def test_a_log_without_durations_still_renders(self) -> None:
        from paimon.ui import nested_calls

        calls = [{"name": "shell", "detail": "ls", "status": "ok", "seconds": None}, {"name": "grep"}]
        self.assertIn("0.00s", nested_calls(calls).plain)

    def test_an_optional_argument_is_still_checked(self) -> None:
        from typing import Optional, TypedDict

        class Args(TypedDict):
            a: Optional[str]
            b: "int | None"

        entry = tools.Tool(description="x", params=Args, run=lambda *a: "")
        self.assertEqual(codemode._signature("f", entry),
                         "async def f(*, a: str | None, b: int | None) -> str: ...")


class ToggleTest(CodeModeTestCase):
    async def test_turned_off_the_tool_is_gone(self) -> None:
        self.assertTrue(Config().code_mode)
        agent = self._agent(code_mode=False)

        self.assertNotIn("run_code", agent.toolset)
        self.assertNotIn("run_code", [schema["function"]["name"] for schema in agent.tool_schemas])
        end = await self._run(agent, "1 + 1")
        self.assertEqual(end.result, "Error: unknown tool 'run_code'")

    def test_the_config_key_turns_it_off(self) -> None:
        path = Path(tempfile.mkdtemp(dir=self.cwd)) / "config.json"
        path.write_text(json.dumps({"code_mode": False}))
        with patch.dict("os.environ", {"PAIMON_CONFIG_HOME": str(path.parent)}):
            self.assertFalse(Config.load().code_mode)

    def test_the_prompt_mentions_it_only_when_it_is_on(self) -> None:
        for code_mode in (True, False):
            with patch("paimon.agent.Session.create", return_value=make_session(self.cwd)):
                agent = Agent.open(cwd=self.cwd, config=Config(model="test:stub", code_mode=code_mode))
            agent.close()
            self.assertEqual("run_code" in agent.system_prompt, code_mode)

    def test_the_description_names_what_this_agent_can_call(self) -> None:
        agent = self._agent()
        description = agent.toolset["run_code"].description
        named = description.rsplit("Callable: ", 1)[1].split(", ")

        # Names only: the arguments are in each tool's own declaration.
        self.assertNotIn("async def", description)
        self.assertEqual(named, list(codemode.callable_tools(agent.toolset)))
        self.assertIn("read_file", named)
        # Tools the loop runs itself, run_code among them, stay direct-only.
        for name in ("write_todos", "spawn_agent", "run_code"):
            self.assertNotIn(name, named)
        self.assertEqual(description,
                         next(schema["function"]["description"] for schema in agent.tool_schemas
                              if schema["function"]["name"] == "run_code"))

        narrowed = Agent(agent.session, "prompt", cwd=self.cwd, config=agent.config,
                         toolset=tools.without(agent.toolset, ("shell", "write_file")))
        self.assertNotIn("shell", narrowed.toolset["run_code"].description.rsplit("Callable: ", 1)[1])
        self.assertIn("shell", named)
        self.assertEqual(set(codemode.callable_tools(narrowed.toolset)),
                         {"read_file", "edit_file", "glob", "grep", "web_search",
                          "search_history", "read_history"})


class ReplayTest(CodeModeTestCase):
    SCRIPT = "await read_file(path='a.txt')\nawait read_file(path='missing.txt')\n'ok'\n"

    async def _session_with_a_script(self) -> Session:
        (self.cwd / "a.txt").write_text("x")
        agent = self._agent()
        await self._events(agent, self.SCRIPT)
        agent.close()
        return agent.session

    async def test_nested_calls_are_persisted_beside_the_result_not_in_it(self) -> None:
        session = await self._session_with_a_script()

        end = next(event for event in replay_events(session.messages())
                   if isinstance(event, ToolEnd))

        self.assertEqual([(call["name"], call["detail"], call["status"]) for call in end.calls],
                         [("read_file", "a.txt", "ok"), ("read_file", "missing.txt", "error")])
        self.assertTrue(all(isinstance(call["seconds"], float) for call in end.calls))
        self.assertNotIn("a.txt", end.result)

    async def test_headless_shows_them_under_the_script(self) -> None:
        session = await self._session_with_a_script()
        events = replay_events(session.messages())

        err = io.StringIO()
        text = headless.TextRenderer(io.StringIO(), err, Config())
        out = io.StringIO()
        machine = headless.JsonRenderer(out, Config())
        for event in events:
            await text.handle(event)
            await machine.handle(event)

        lines = err.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("· run_code  await read_file(path='a.txt')"))
        self.assertRegex(lines[1], r"^    read_file  a\.txt  → ok \d+\.\d\ds$")
        self.assertRegex(lines[2], r"^    read_file  missing\.txt  → error \d+\.\d\ds$")
        result = next(json.loads(line) for line in out.getvalue().splitlines()
                      if json.loads(line)["type"] == "tool_result")
        self.assertEqual([call["name"] for call in result["calls"]], ["read_file", "read_file"])


class TranscriptTest(AppTestCase):
    async def test_nested_calls_are_listed_under_the_script(self) -> None:
        calls = [{"name": "read_file", "detail": "a.txt", "status": "ok", "seconds": 0.01},
                 {"name": "shell", "detail": "rm -rf /", "status": "denied", "seconds": 0.5}]
        app = self.make_app()
        async with app.run_test() as pilot:
            renderer = EventRenderer(app.pane.transcript, app.pane.agent)
            await renderer.handle(ToolStart("c1", "run_code", {"code": "await read_file(path='a.txt')"}))
            await renderer.handle(ToolEnd("c1", "run_code", "Script completed in 0.51s\nok", calls=calls))
            await pilot.pause()

            nested = str(app.query_one(".tool-entry .tool-nested", Static).render())
            self.assertIn("read  a.txt  ✓ 0.01s", nested)
            self.assertIn("shell  rm -rf /  denied 0.50s", nested)
