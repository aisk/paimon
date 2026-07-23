import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from paimon import cli
from paimon.session import FORMAT_VERSION, Session, _project_dir


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

    def _session(self, session_id: str) -> Session:
        directory = _project_dir(self.cwd)
        directory.mkdir(parents=True, exist_ok=True)
        session = Session(directory / f"{session_id[:8]}.jsonl", session_id, self.cwd)
        session.append({"type": "session", "version": FORMAT_VERSION, "id": session_id,
                        "cwd": str(self.cwd), "created_at": "2026-07-24T00:00:00+00:00"})
        session.append_message({"role": "user", "content": "hi"})
        return session

    def _main_exit(self, *argv: str) -> tuple[int, str]:
        stderr = io.StringIO()
        with patch("sys.argv", ["paimon", *argv]), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                cli.main()
        return ctx.exception.code, stderr.getvalue()


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


if __name__ == "__main__":
    unittest.main()
