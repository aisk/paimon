import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from helpers import SILENT_EVENTS, agent_events, make_session, stub_model
from pydantic_ai.messages import ModelResponse, TextPart

from paimon import headless
from paimon.agent import (
    Agent,
    ContextCompacted,
    ReasoningDelta,
    TextDelta,
    TodosUpdate,
    ToolEnd,
    ToolStart,
)
from paimon.config import Config


def _config(**kwargs) -> Config:
    return Config(model="test:stub", **kwargs)


async def _feed(renderer, *events) -> None:
    for event in events:
        await renderer.handle(event)
    await renderer.close()


class TextRendererTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.out, self.err = io.StringIO(), io.StringIO()

    def _renderer(self, **kwargs) -> headless.TextRenderer:
        return headless.TextRenderer(self.out, self.err, _config(**kwargs))

    async def test_text_goes_to_stdout_alone(self) -> None:
        renderer = self._renderer()
        await _feed(renderer, TextDelta("26 Python "), TextDelta("files."))
        self.assertEqual(self.out.getvalue(), "26 Python files.\n")
        self.assertEqual(self.err.getvalue(), "")

    async def test_trailing_newline_is_not_doubled(self) -> None:
        renderer = self._renderer()
        await _feed(renderer, TextDelta("done\n"))
        self.assertEqual(self.out.getvalue(), "done\n")

    async def test_tool_call_summary_goes_to_stderr_on_one_line(self) -> None:
        renderer = self._renderer()
        await _feed(
            renderer,
            ToolStart("call-1", "bash", {"command": "find . \\\n  -name '*.py'"}),
            ToolEnd("call-1", "bash", "26"),
        )
        self.assertEqual(self.out.getvalue(), "")
        self.assertEqual(self.err.getvalue(), "· bash  find . \\ -name '*.py'\n")

    async def test_long_detail_is_truncated(self) -> None:
        renderer = self._renderer()
        await _feed(renderer, ToolStart("call-1", "bash", {"command": "x" * 500}))
        line = self.err.getvalue().strip()
        self.assertTrue(line.endswith("…"))
        self.assertLess(len(line), headless.DETAIL_WIDTH + 20)

    async def test_denied_call_is_reported_and_advises_once(self) -> None:
        renderer = self._renderer()
        await _feed(
            renderer,
            ToolStart("call-1", "edit_file", {"path": "README.md"}),
            ToolEnd("call-1", "edit_file", "User denied this operation.", denied=True),
            ToolStart("call-2", "bash", {"command": "ls"}),
            ToolEnd("call-2", "bash", "User denied this operation.", denied=True),
        )
        err = self.err.getvalue()
        self.assertEqual(err.count("→ denied"), 2)  # both outcomes reported
        self.assertEqual(err.count("--mode yolo"), 1)  # advised only once
        self.assertEqual(self.out.getvalue(), "")

    async def test_reasoning_is_hidden_by_default_and_shown_when_enabled(self) -> None:
        hidden = self._renderer()
        await _feed(hidden, ReasoningDelta("pondering"))
        self.assertEqual(self.err.getvalue(), "")

        self.err = io.StringIO()
        shown = self._renderer(show_reasoning=True)
        await _feed(shown, ReasoningDelta("pondering"))
        self.assertIn("pondering", self.err.getvalue())
        self.assertEqual(self.out.getvalue(), "")

    async def test_resume_command_is_reported_on_stderr(self) -> None:
        renderer = self._renderer()
        renderer.begin("3f2a8b1c-0000-0000-0000-000000000000", "test:stub", "read", Path("/tmp"))
        await _feed(renderer)
        renderer.finish()
        self.assertIn("paimon -r 3f2a8b1c", self.err.getvalue())
        self.assertEqual(self.out.getvalue(), "")

    async def test_broken_stdout_does_not_raise(self) -> None:
        self.out.close()
        renderer = self._renderer()
        await _feed(renderer, TextDelta("gone"))


class JsonRendererTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.out = io.StringIO()

    def _lines(self) -> list[dict]:
        return [json.loads(line) for line in self.out.getvalue().splitlines()]

    async def test_stream_starts_with_init_and_ends_with_result(self) -> None:
        renderer = headless.JsonRenderer(self.out)
        renderer.begin("sid", "test:stub", "read", Path("/tmp"))
        await _feed(renderer, TextDelta("hi"))
        renderer.finish()

        lines = self._lines()
        self.assertEqual(lines[0]["type"], "init")
        self.assertEqual(lines[0]["session_id"], "sid")
        self.assertEqual(lines[-1], {
            "type": "result", "subtype": "success", "is_error": False,
            "session_id": "sid", "text": "hi", "denied": 0, "error": None,
        })

    async def test_event_types_and_text_blocks(self) -> None:
        renderer = headless.JsonRenderer(self.out)
        renderer.begin("sid")
        await _feed(
            renderer,
            ReasoningDelta("think"),
            TextDelta("Looking."),
            ToolStart("call-1", "bash", {"command": "ls"}),
            ToolEnd("call-1", "bash", "denied", denied=True),
            TodosUpdate([{"content": "a", "status": "pending"}]),
            ContextCompacted(100, 20),
            TextDelta("Done."),
        )
        renderer.finish()

        lines = self._lines()
        self.assertEqual(
            [line["type"] for line in lines],
            ["init", "reasoning_delta", "text_delta", "tool_use", "tool_result",
             "todos", "compacted", "text_delta", "result"],
        )
        # Text blocks are split where a tool call interrupts them.
        self.assertEqual(lines[-1]["text"], "Looking.\n\nDone.")
        self.assertEqual(lines[-1]["denied"], 1)

    async def test_reasoning_is_emitted_even_when_display_is_off(self) -> None:
        renderer = headless.JsonRenderer(self.out, _config(show_reasoning=False))
        renderer.begin("sid")
        await _feed(renderer, ReasoningDelta("think"))
        renderer.finish()
        self.assertIn("reasoning_delta", [line["type"] for line in self._lines()])

    async def test_early_failure_still_produces_init_and_result(self) -> None:
        with patch("sys.stdout", self.out):
            code = headless.fail("no model configured; run 'paimon' once to log in", "json")
        self.assertEqual(code, 1)
        lines = self._lines()
        self.assertEqual([line["type"] for line in lines], ["init", "result"])
        self.assertIsNone(lines[0]["session_id"])
        self.assertTrue(lines[-1]["is_error"])
        self.assertEqual(lines[-1]["subtype"], "error")
        self.assertIn("log in", lines[-1]["error"])


class EventCoverageTest(unittest.IsolatedAsyncioTestCase):
    """Every agent event must be renderable by both headless renderers.

    Adding an event to paimon.agent without teaching the renderers about it
    should fail here rather than crash a run. The TUI renderer is held to the
    same list in test_app.
    """

    async def test_every_event_renders(self) -> None:
        events = agent_events()
        for renderer in (headless.TextRenderer(io.StringIO(), io.StringIO(), _config()),
                         headless.JsonRenderer(io.StringIO(), _config())):
            renderer.begin("sid")
            await _feed(renderer, *events)
            renderer.finish()

    async def test_json_emits_a_line_for_every_visible_event(self) -> None:
        """A machine consumer should not silently lose an event type."""
        out = io.StringIO()
        renderer = headless.JsonRenderer(out, _config())
        renderer.begin("sid")
        for event in agent_events():
            if type(event).__name__ in SILENT_EVENTS | {"UserInput", "CompactionNotice"}:
                continue  # replay-only: never reaches a headless run
            before = out.getvalue()
            await renderer.handle(event)
            self.assertNotEqual(out.getvalue(), before,
                                f"JsonRenderer emitted nothing for {type(event).__name__}")


class PromptBuildingTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name)

    def test_prompt_only(self) -> None:
        self.assertEqual(headless.build_prompt("hello", "", self.cwd), "hello")

    def test_piped_only(self) -> None:
        built = headless.build_prompt("", "log line", self.cwd)
        self.assertEqual(built, "<piped_input>\nlog line\n</piped_input>")

    def test_piped_content_comes_before_the_instruction(self) -> None:
        built = headless.build_prompt("summarize", "log line", self.cwd)
        self.assertLess(built.index("log line"), built.index("summarize"))
        self.assertIn("</piped_input>", built)

    def test_both_empty(self) -> None:
        self.assertEqual(headless.build_prompt("  ", "", self.cwd), "")

    def test_mentions_expand_in_the_prompt_but_not_in_piped_input(self) -> None:
        (self.cwd / "note.txt").write_text("secret contents")
        built = headless.build_prompt("read @note.txt", "@note.txt", self.cwd)
        self.assertEqual(built.count("secret contents"), 1)
        self.assertIn("<piped_input>\n@note.txt\n</piped_input>", built)


class ReadStdinTest(unittest.TestCase):
    def _stdin(self, raw: bytes):
        return SimpleNamespace(buffer=io.BytesIO(raw))

    def test_decodes_utf8(self) -> None:
        self.assertEqual(headless.read_stdin(self._stdin("日志\n".encode())), "日志")

    def test_invalid_bytes_do_not_raise(self) -> None:
        self.assertIn("�", headless.read_stdin(self._stdin(b"\xff\xfe bad")))

    def test_oversized_input_is_truncated(self) -> None:
        text = headless.read_stdin(self._stdin(b"x" * (headless.MAX_STDIN_CHARS + 500)))
        self.assertIn("truncated", text)
        self.assertLess(len(text), headless.MAX_STDIN_CHARS + 100)


class DriveTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name)
        self.session = make_session(self.cwd)
        self.session.append_system_prompt("sys")
        self.addCleanup(self.session.unlock)
        self.out, self.err = io.StringIO(), io.StringIO()

    def _agent(self, model) -> Agent:
        # The model is built lazily on the first request, so the patch has to
        # outlive construction.
        patcher = patch("paimon.agent.build_model", return_value=model)
        patcher.start()
        self.addCleanup(patcher.stop)
        return Agent.open(cwd=self.cwd, session=self.session, config=_config())

    def _renderer(self) -> headless.TextRenderer:
        return headless.TextRenderer(self.out, self.err, _config())

    async def test_successful_turn_exits_zero(self) -> None:
        agent = self._agent(stub_model())
        code = await headless._drive(agent, self._renderer(), "hi")
        self.assertEqual(code, 0)
        self.assertEqual(self.out.getvalue(), "done\n")

    async def test_model_failure_exits_one_and_reports(self) -> None:
        async def explode(messages, info):
            raise RuntimeError("boom")
            yield  # pragma: no cover - makes this an async generator

        from pydantic_ai.models.function import FunctionModel

        agent = self._agent(FunctionModel(stream_function=explode))
        renderer = headless.JsonRenderer(self.out, _config())
        renderer.begin("sid")
        self.assertEqual(await headless._drive(agent, renderer, "hi"), 1)
        result = json.loads(self.out.getvalue().splitlines()[-1])
        self.assertEqual(result["subtype"], "error")
        self.assertIn("boom", result["error"])

    async def test_interrupt_exits_130_and_keeps_history_loadable(self) -> None:
        from pydantic_ai.models.function import FunctionModel

        async def stall(messages, info):
            yield "partial answer"
            await asyncio.sleep(3600)

        agent = self._agent(FunctionModel(stream_function=stall))
        # _install_interrupt receives the turn task; cancelling it is exactly
        # what the SIGINT handler does, without going near real signals.
        captured: list = []
        with patch.object(headless, "_install_interrupt", captured.append):
            drive = asyncio.ensure_future(headless._drive(agent, self._renderer(), "hi"))
            for _ in range(100):  # let the stream deliver its first chunk
                await asyncio.sleep(0)
                if self.out.getvalue():
                    break
            self.assertTrue(captured, "_drive never installed an interrupt handler")
            captured[0].cancel()
            code = await drive

        self.assertEqual(code, 130)
        # The partial response must still be persisted, or the session log
        # would end on a request with no answer (see Agent.run's cancel path).
        last = self.session.messages()[-1]
        self.assertIsInstance(last, ModelResponse)
        self.assertIsInstance(last.parts[0], TextPart)
        self.assertIn("partial", last.parts[0].content)


if __name__ == "__main__":
    unittest.main()
