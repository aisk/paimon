import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from helpers import stub_model
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from paimon import cli
from paimon.config import Config
from paimon.session import Session, _project_dir


class CliTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = patch.dict("os.environ", {"PAIMON_DATA_HOME": tmp.name})
        env.start()
        self.addCleanup(env.stop)
        self.cwd = Path(tmp.name) / "project"
        self.cwd.mkdir()
        cwd = patch("paimon.cli.Path.cwd", return_value=self.cwd)
        cwd.start()
        self.addCleanup(cwd.stop)
        # A terminal by default, so tests opt into headless explicitly rather
        # than inheriting it from however the suite was launched.
        self._set_stdin(tty=True)

    def _set_stdin(self, *, tty: bool, data: bytes = b"") -> None:
        stdin = patch("sys.stdin", SimpleNamespace(isatty=lambda: tty, buffer=io.BytesIO(data)))
        stdin.start()
        self.addCleanup(stdin.stop)

    def _session(self, session_id: str) -> Session:
        directory = _project_dir(self.cwd)
        directory.mkdir(parents=True, exist_ok=True)
        session = Session(directory / f"{session_id[:8]}.jsonl", session_id, self.cwd)
        session.append({"type": "session", "id": session_id,
                        "cwd": str(self.cwd), "created_at": "2026-07-24T00:00:00+00:00"})
        session.append_message(ModelRequest(parts=[UserPromptPart(content="hi")]))
        return session

    def _main_exit(self, *argv: str) -> tuple[int, str]:
        stderr = io.StringIO()
        with patch("sys.argv", ["paimon", *argv]), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                cli.main()
        return ctx.exception.code, stderr.getvalue()

    def _main_output(self, *argv: str, model: str | None = "test:stub",
                     tool: str | None = None, stub=None) -> tuple[int, str, str]:
        """Run main() headless against a stubbed model, capturing both streams."""
        out, err = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["paimon", *argv]), \
                patch("paimon.headless.Config.load", return_value=Config(model=model)), \
                patch("paimon.agent.build_model", return_value=stub or stub_model(tool)), \
                patch("paimon.agent.build_system_prompt", return_value="sys"), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                cli.main()
        return ctx.exception.code, out.getvalue(), err.getvalue()


class ResumeResolutionTest(CliTestCase):
    def test_unknown_prefix_exits_1(self) -> None:
        code, stderr = self._main_exit("--resume", "ffffffff")
        self.assertEqual(code, 1)
        self.assertIn("no session matching", stderr)

    def test_ambiguous_prefix_exits_1(self) -> None:
        self._session("abc11111-0000-0000-0000-000000000000")
        self._session("abc22222-0000-0000-0000-000000000000")
        code, stderr = self._main_exit("--resume", "abc")
        self.assertEqual(code, 1)
        self.assertIn("ambiguous", stderr)


class WebForwardingTest(CliTestCase):
    def _serve_command(self, *argv: str) -> str:
        captured: dict = {}

        class FakeServer:
            def __init__(self, command: str, port: int = 8000) -> None:
                captured["command"] = command

            def serve(self) -> None:
                pass

        fake_module = SimpleNamespace(Server=FakeServer)
        with patch("sys.argv", ["paimon", *argv]), \
                patch.dict("sys.modules", {"textual_serve.server": fake_module}):
            cli.main()
        return captured["command"]

    def test_forwards_resume_id(self) -> None:
        self.assertIn("--resume 3f2a", self._serve_command("--web", "--resume", "3f2a"))

    def test_forwards_bare_resume(self) -> None:
        self.assertTrue(self._serve_command("--web", "--resume").endswith("--resume"))

    def test_forwards_tui_flag(self) -> None:
        # textual-serve gives the child a piped stdin, which would otherwise
        # be read as "run headless" and exit immediately.
        self.assertIn("--tui", self._serve_command("--web"))

    def test_forwards_model_override(self) -> None:
        self.assertIn("--model test:stub", self._serve_command("--web", "--model", "test:stub"))

    def test_forwards_continue(self) -> None:
        self.assertIn("--continue", self._serve_command("--web", "-c"))


class TuiExitTest(CliTestCase):
    """The UI also has to tell the user how to get the conversation back."""

    def _run_tui(self, *argv: str, history: list | None = None, crash: bool = False) -> str:
        session = SimpleNamespace(id="9c1e40aa-0000-0000-0000-000000000000")
        agent = SimpleNamespace(session=session, history=[1] if history is None else history)

        class FakeApp:
            def __init__(self, opened, **kwargs) -> None:
                self.agent = opened

            def run(self) -> None:
                if crash:
                    raise RuntimeError("boom")

        stderr = io.StringIO()
        with patch("sys.argv", ["paimon", *argv]), patch("paimon.cli.PaimonApp", FakeApp), \
                patch("paimon.cli.Agent.open", return_value=agent), \
                contextlib.redirect_stderr(stderr):
            if crash:
                with self.assertRaises(RuntimeError):
                    cli.main()
            else:
                cli.main()
        return stderr.getvalue()

    def test_resume_command_is_printed_on_exit(self) -> None:
        self.assertIn("paimon -r 9c1e40aa", self._run_tui())

    def test_untouched_session_prints_nothing(self) -> None:
        self.assertEqual(self._run_tui(history=[]), "")

    def test_resume_command_survives_a_crash(self) -> None:
        self.assertIn("paimon -r 9c1e40aa", self._run_tui(crash=True))

    def test_nothing_is_printed_under_tui_flag(self) -> None:
        # --tui means textual-serve owns these streams.
        self.assertEqual(self._run_tui("--tui"), "")


class HeadlessArgumentTest(CliTestCase):
    def test_web_and_print_conflict(self) -> None:
        code, stderr = self._main_exit("--web", "-p", "hi")
        self.assertEqual(code, 2)
        self.assertIn("cannot be combined", stderr)

    def test_output_format_without_print_is_rejected(self) -> None:
        code, stderr = self._main_exit("--output-format", "json")
        self.assertEqual(code, 2)
        self.assertIn("only applies to --print", stderr)

    def test_bare_resume_with_print_is_rejected(self) -> None:
        code, stderr = self._main_exit("-p", "hi", "-r")
        self.assertEqual(code, 2)
        self.assertIn("needs a session id", stderr)

    def test_empty_prompt_is_rejected(self) -> None:
        code, stderr = self._main_exit("-p", "   ")
        self.assertEqual(code, 2)
        self.assertIn("nothing to do", stderr)

    def test_unknown_resume_prefix_still_exits_1(self) -> None:
        code, stderr = self._main_exit("-p", "hi", "-r", "ffffffff")
        self.assertEqual(code, 1)
        self.assertIn("no session matching", stderr)

    def test_continue_and_resume_conflict(self) -> None:
        code, stderr = self._main_exit("-c", "-r", "abcd")
        self.assertEqual(code, 2)
        self.assertIn("cannot be combined", stderr)

    def test_unqualified_model_is_a_usage_error(self) -> None:
        code, stderr = self._main_exit("--model", "gpt-5")
        self.assertEqual(code, 2)
        self.assertIn("provider:model", stderr)

    def test_invalid_profile_name_is_a_usage_error(self) -> None:
        code, stderr = self._main_exit("--profile", "../evil")
        self.assertEqual(code, 2)
        self.assertIn("invalid profile name", stderr)

    def test_append_system_prompt_without_print_is_rejected(self) -> None:
        code, stderr = self._main_exit("--append-system-prompt", "role")
        self.assertEqual(code, 2)
        self.assertIn("only applies to --print", stderr)

    def test_append_system_prompt_with_continue_is_rejected(self) -> None:
        code, stderr = self._main_exit("-p", "hi", "-c", "--append-system-prompt", "role")
        self.assertEqual(code, 2)
        self.assertIn("cannot be combined", stderr)

    def test_append_system_prompt_with_resume_is_rejected(self) -> None:
        code, stderr = self._main_exit("-p", "hi", "-r", "abcd", "--append-system-prompt", "role")
        self.assertEqual(code, 2)
        self.assertIn("cannot be combined", stderr)

    def test_blank_append_system_prompt_is_rejected(self) -> None:
        code, stderr = self._main_exit("-p", "hi", "--append-system-prompt", "   ")
        self.assertEqual(code, 2)
        self.assertIn("non-empty", stderr)


def _slow_model() -> FunctionModel:
    async def stream(messages, info):
        await asyncio.sleep(30)
        yield "late"

    return FunctionModel(stream_function=stream)


class GuardRailTest(CliTestCase):
    def test_timeout_exits_124(self) -> None:
        code, out, err = self._main_output("-p", "hi", "--output-format", "json",
                                           "--timeout", "0.2", stub=_slow_model())
        self.assertEqual(code, 124)
        result = json.loads(out.splitlines()[-1])
        self.assertEqual(result["subtype"], "timeout")
        self.assertTrue(result["is_error"])
        self.assertTrue(result["session_id"])

    def test_tool_budget_exits_4_before_the_tool_runs(self) -> None:
        code, out, err = self._main_output("-p", "run it", "--output-format", "json",
                                           "--max-tool-calls", "0", tool="shell")
        self.assertEqual(code, 4)
        lines = [json.loads(line) for line in out.splitlines()]
        self.assertEqual(lines[-1]["subtype"], "max_tool_calls")
        types = [line["type"] for line in lines]
        self.assertIn("tool_use", types)  # the caller sees which call blew the budget
        self.assertNotIn("tool_result", types)  # ... and that it never executed

    def test_session_is_resumable_after_a_budget_stop(self) -> None:
        self._main_output("-p", "run it", "--max-tool-calls", "0", tool="shell")
        session = Session.list(self.cwd)[0]
        code, out, err = self._main_output("-p", "again", "-c")
        self.assertEqual(code, 0)
        self.assertEqual([s.id for s in Session.list(self.cwd)], [session.id])

    def test_timeout_without_print_is_rejected(self) -> None:
        code, stderr = self._main_exit("--timeout", "5")
        self.assertEqual(code, 2)
        self.assertIn("only applies to --print", stderr)

    def test_negative_tool_budget_is_rejected(self) -> None:
        code, stderr = self._main_exit("-p", "hi", "--max-tool-calls", "-1")
        self.assertEqual(code, 2)
        self.assertIn("must be positive", stderr)

    def test_zero_timeout_is_rejected(self) -> None:
        code, stderr = self._main_exit("-p", "hi", "--timeout", "0")
        self.assertEqual(code, 2)
        self.assertIn("must be positive", stderr)


class HeadlessRunTest(CliTestCase):
    def test_missing_model_exits_1_without_creating_a_session(self) -> None:
        code, out, err = self._main_output("-p", "hi", model=None)
        self.assertEqual(code, 1)
        self.assertIn("log in", err)
        self.assertEqual(out, "")
        self.assertEqual(Session.list(self.cwd), [])

    def test_answer_goes_to_stdout_and_session_is_persisted(self) -> None:
        code, out, err = self._main_output("-p", "hi")
        self.assertEqual(code, 0)
        self.assertEqual(out, "done\n")
        sessions = Session.list(self.cwd)
        self.assertEqual(len(sessions), 1)
        self.assertIn(f"paimon -r {sessions[0].id[:8]}", err)

    def test_json_output_is_one_object_per_line(self) -> None:
        code, out, err = self._main_output("-p", "hi", "--output-format", "json")
        self.assertEqual(code, 0)
        lines = [json.loads(line) for line in out.splitlines()]
        self.assertEqual(lines[0]["type"], "init")
        self.assertEqual(lines[0]["model"], "test:stub")
        self.assertEqual(lines[0]["mode"], "read")
        self.assertEqual(lines[-1]["type"], "result")
        self.assertEqual(lines[-1]["text"], "done")
        self.assertEqual(err, "")

    def test_result_output_is_exactly_one_object_with_log_end(self) -> None:
        code, out, err = self._main_output("-p", "hi", "--output-format", "result")
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(len(lines), 1)
        result = json.loads(lines[0])
        self.assertEqual(result["type"], "result")
        self.assertEqual(result["text"], "done")
        sessions = Session.list(self.cwd)
        self.assertEqual(result["session_id"], sessions[0].id)
        # log_end is the session log's line count, the cursor for `paimon log --after`.
        with sessions[0].path.open(encoding="utf-8") as file:
            self.assertEqual(result["log_end"], sum(1 for _ in file))
        self.assertEqual(err, "")

    def test_denied_tool_still_exits_0(self) -> None:
        code, out, err = self._main_output(
            "-p", "run it", tool="shell", model="test:stub",
        )
        self.assertEqual(code, 0)
        self.assertIn("denied", err)
        self.assertIn("--mode yolo", err)

    def test_resume_appends_to_the_same_session(self) -> None:
        self._main_output("-p", "hi")
        session = Session.list(self.cwd)[0]
        before = len(session.messages())

        code, out, err = self._main_output("-p", "again", "-r", session.id[:4])
        self.assertEqual(code, 0)
        self.assertEqual([s.id for s in Session.list(self.cwd)], [session.id])
        self.assertGreater(len(Session.list(self.cwd)[0].messages()), before)

    def test_model_override_reaches_the_run_without_saving(self) -> None:
        code, out, err = self._main_output("-p", "hi", "--output-format", "json",
                                           "--model", "test:other")
        self.assertEqual(code, 0)
        init = json.loads(out.splitlines()[0])
        self.assertEqual(init["model"], "test:other")

    def test_continue_resumes_the_latest_session(self) -> None:
        self._main_output("-p", "hi")
        session = Session.list(self.cwd)[0]
        before = len(session.messages())

        code, out, err = self._main_output("-p", "again", "-c")
        self.assertEqual(code, 0)
        self.assertEqual([s.id for s in Session.list(self.cwd)], [session.id])
        self.assertGreater(len(Session.list(self.cwd)[0].messages()), before)

    def test_continue_without_sessions_exits_1(self) -> None:
        code, out, err = self._main_output("-p", "hi", "-c")
        self.assertEqual(code, 1)
        self.assertIn("no session in this directory", err)

    def test_profile_config_is_read_by_a_headless_run(self) -> None:
        config_home = self.cwd / "cfghome"
        profile_config = config_home / "work" / "config.json"
        profile_config.parent.mkdir(parents=True)
        profile_config.write_text(json.dumps({"model": "test:profile"}), encoding="utf-8")

        out, err = io.StringIO(), io.StringIO()
        argv = ["paimon", "-p", "hi", "--profile", "work", "--output-format", "json"]
        with patch.dict("os.environ", {"PAIMON_CONFIG_HOME": str(config_home)}), \
                patch("sys.argv", argv), \
                patch("paimon.agent.build_model", return_value=stub_model()), \
                patch("paimon.agent.build_system_prompt", return_value="sys"), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                cli.main()
        self.assertEqual(ctx.exception.code, 0)
        init = json.loads(out.getvalue().splitlines()[0])
        self.assertEqual(init["model"], "test:profile")

    def test_busy_session_exits_1(self) -> None:
        session = self._session("abc11111-0000-0000-0000-000000000000")
        session.append_system_prompt("sys")
        # The lock is refcounted per process, so a same-process lock would not
        # collide; refuse it outright to stand in for another process.
        with patch("paimon.session.lockfile.acquire", return_value=False):
            code, out, err = self._main_output("-p", "hi", "-r", "abc11111")
        self.assertEqual(code, 1)
        self.assertIn("already active", err)

    def test_append_system_prompt_is_persisted_with_the_session(self) -> None:
        code, out, err = self._main_output("-p", "hi", "--append-system-prompt", "Be a reviewer.")
        self.assertEqual(code, 0)
        session = Session.list(self.cwd)[0]
        self.assertEqual(session.system_prompt(), "sys\n\nBe a reviewer.")

        # A resume without the flag keeps the appended role.
        code, out, err = self._main_output("-p", "again", "-c")
        self.assertEqual(code, 0)
        self.assertEqual(Session.list(self.cwd)[0].system_prompt(), "sys\n\nBe a reviewer.")

    def test_piped_stdin_without_print_becomes_the_prompt(self) -> None:
        self._set_stdin(tty=False, data="日志内容".encode())
        captured: dict = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return 0

        with patch("sys.argv", ["paimon"]), patch("paimon.headless.run", fake_run):
            with self.assertRaises(SystemExit) as ctx:
                cli.main()
        self.assertEqual(ctx.exception.code, 0)
        self.assertEqual(captured["prompt"], "")
        self.assertEqual(captured["piped"], "日志内容")

    def test_piped_stdin_is_combined_with_the_prompt(self) -> None:
        self._set_stdin(tty=False, data=b"log line")
        code, out, err = self._main_output("-p", "summarize")
        self.assertEqual(code, 0)
        session = Session.list(self.cwd)[0]
        prompt = session.messages()[0].parts[0].content
        self.assertIn("<piped_input>\nlog line\n</piped_input>", prompt)
        self.assertLess(prompt.index("log line"), prompt.index("summarize"))


if __name__ == "__main__":
    unittest.main()
