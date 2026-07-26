import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic_ai.messages import ModelRequest, UserPromptPart

from paimon import cli, commands
from paimon.config import config_path
from paimon.session import Session


class CommandTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        env = patch.dict("os.environ", {
            "PAIMON_CONFIG_HOME": str(self.home / "config"),
            "PAIMON_DATA_HOME": str(self.home / "data"),
        })
        env.start()
        self.addCleanup(env.stop)

    def _write_config(self, **data) -> None:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def _run(self, *argv: str, stdin: str = "") -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        fake_stdin = SimpleNamespace(read=lambda: stdin, isatty=lambda: False)
        with patch("sys.argv", ["paimon", *argv]), patch("sys.stdin", fake_stdin), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                cli.main()
        return ctx.exception.code, out.getvalue(), err.getvalue()


class StatusTest(CommandTestCase):
    def test_unconfigured_exits_1(self) -> None:
        code, out, err = self._run("status")
        self.assertEqual(code, 1)
        self.assertIn("not logged in", out)

    def test_configured_exits_0(self) -> None:
        self._write_config(model="zai:glm-4.7", api_key="sk-secret")
        code, out, err = self._run("status")
        self.assertEqual(code, 0)
        self.assertIn("zai:glm-4.7", out)
        self.assertIn("api key set", out)

    def test_json_reports_state_without_the_key(self) -> None:
        self._write_config(model="zai:glm-4.7", api_key="sk-secret", api_base="https://x/v1")
        code, out, err = self._run("status", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(payload["logged_in"])
        self.assertEqual(payload["model"], "zai:glm-4.7")
        self.assertEqual(payload["api_base"], "https://x/v1")
        self.assertTrue(payload["api_key_set"])
        self.assertEqual(payload["sessions_here"], 0)
        self.assertNotIn("sk-secret", out)

    def test_json_unconfigured(self) -> None:
        code, out, err = self._run("status", "--json")
        self.assertEqual(code, 1)
        payload = json.loads(out)
        self.assertFalse(payload["logged_in"])
        self.assertIsNone(payload["model"])
        self.assertFalse(payload["api_key_set"])


class LoginTest(CommandTestCase):
    def test_key_from_environment_variable(self) -> None:
        with patch.dict("os.environ", {"MY_KEY": "sk-abc"}):
            code, out, err = self._run("login", "--model", "openai:gpt-5",
                                       "--api-key-env", "MY_KEY")
        self.assertEqual(code, 0)
        data = json.loads(config_path().read_text())
        self.assertEqual(data["model"], "openai:gpt-5")
        self.assertEqual(data["api_key"], "sk-abc")

    def test_key_from_stdin(self) -> None:
        code, out, err = self._run("login", "--model", "openai:gpt-5",
                                   "--api-key-stdin", stdin="sk-piped\n")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(config_path().read_text())["api_key"], "sk-piped")

    def test_unset_env_var_is_refused(self) -> None:
        code, out, err = self._run("login", "--model", "openai:gpt-5",
                                   "--api-key-env", "NO_SUCH_VAR")
        self.assertEqual(code, 1)
        self.assertIn("NO_SUCH_VAR", err)
        self.assertFalse(config_path().exists())

    def test_empty_stdin_is_refused(self) -> None:
        code, out, err = self._run("login", "--model", "openai:gpt-5",
                                   "--api-key-stdin", stdin="")
        self.assertEqual(code, 1)
        self.assertIn("stdin", err)

    def test_unqualified_model_is_refused(self) -> None:
        code, out, err = self._run("login", "--model", "gpt-5")
        self.assertEqual(code, 1)
        self.assertIn("provider:model", err)

    def test_unknown_provider_is_refused(self) -> None:
        code, out, err = self._run("login", "--model", "nosuchprovider:m")
        self.assertEqual(code, 1)
        self.assertIn("nosuchprovider", err)

    def test_unpassed_fields_keep_their_values(self) -> None:
        self._write_config(model="zai:glm-4.7", api_key="sk-old", theme="dark")
        code, out, err = self._run("login", "--model", "openai:gpt-5")
        self.assertEqual(code, 0)
        data = json.loads(config_path().read_text())
        self.assertEqual(data["model"], "openai:gpt-5")
        self.assertEqual(data["api_key"], "sk-old")
        self.assertEqual(data["theme"], "dark")


class SessionsTest(CommandTestCase):
    def _make_session(self, text: str) -> Session:
        session = Session.create(Path.cwd())
        session.append_message(ModelRequest(parts=[UserPromptPart(content=text)]))
        return session

    def test_empty_json_is_an_empty_array(self) -> None:
        code, out, err = self._run("sessions", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), [])

    def test_empty_text_keeps_stdout_clean(self) -> None:
        code, out, err = self._run("sessions")
        self.assertEqual(code, 0)
        self.assertEqual(out, "")
        self.assertIn("no sessions", err)

    def test_json_lists_newest_first_with_preview(self) -> None:
        older = self._make_session("first   question\nwith a newline")
        newer = self._make_session("second question")
        past = older.path.stat().st_mtime - 60
        os.utime(older.path, (past, past))

        code, out, err = self._run("sessions", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual([entry["id"] for entry in payload], [newer.id, older.id])
        self.assertEqual(payload[1]["preview"], "first question with a newline")
        self.assertTrue(payload[0]["created_at"])
        self.assertTrue(payload[0]["path"].endswith(".jsonl"))

    def test_text_lists_short_ids(self) -> None:
        session = self._make_session("hi")
        code, out, err = self._run("sessions")
        self.assertEqual(code, 0)
        self.assertIn(session.id[:8], out)
        self.assertIn("hi", out)


class VersionTest(unittest.TestCase):
    def test_version_is_a_string(self) -> None:
        self.assertIsInstance(commands.version(), str)


if __name__ == "__main__":
    unittest.main()
