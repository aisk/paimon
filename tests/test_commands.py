import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

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


class ProfileTest(CommandTestCase):
    def test_login_and_status_share_a_profile(self) -> None:
        with patch.dict("os.environ", {"WORK_KEY": "sk-work"}):
            code, out, err = self._run("login", "--profile", "work", "--model",
                                       "openai:gpt-5", "--api-key-env", "WORK_KEY")
        self.assertEqual(code, 0)
        profile_config = self.home / "config" / "work" / "config.json"
        self.assertEqual(json.loads(profile_config.read_text())["api_key"], "sk-work")

        code, out, err = self._run("status", "--profile", "work", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["model"], "openai:gpt-5")

    def test_default_profile_is_untouched(self) -> None:
        with patch.dict("os.environ", {"WORK_KEY": "sk-work"}):
            self._run("login", "--profile", "work", "--model", "openai:gpt-5",
                      "--api-key-env", "WORK_KEY")
        code, out, err = self._run("status", "--json")
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(out)["logged_in"])

    def test_traversal_in_profile_name_is_rejected(self) -> None:
        code, out, err = self._run("status", "--profile", "../evil")
        self.assertEqual(code, 2)
        self.assertIn("invalid profile name", err)

    def test_default_profile_lives_in_its_own_directory(self) -> None:
        with patch.dict("os.environ", {"KEY": "sk-x"}):
            code, out, err = self._run("login", "--model", "openai:gpt-5", "--api-key-env", "KEY")
        self.assertEqual(code, 0)
        path = self.home / "config" / "default" / "config.json"
        self.assertEqual(json.loads(path.read_text())["model"], "openai:gpt-5")


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


class LogTest(CommandTestCase):
    """Seq numbers are physical line numbers; Session.create writes the header
    record on line 1, so the first appended message lands at seq 2."""

    def _make_turn(self, session: Session, prompt: str, answer: str) -> None:
        session.append_message(ModelRequest(parts=[UserPromptPart(content=prompt)]))
        session.append_message(ModelResponse(parts=[
            TextPart(content=answer),
            ToolCallPart(tool_name="bash", args={"command": "ls -la"}, tool_call_id="c1"),
        ]))
        session.append_message(ModelRequest(parts=[
            ToolReturnPart(tool_name="bash", content="file-a\nfile-b", tool_call_id="c1"),
        ]))

    def test_renders_seq_prefixed_compact_lines(self) -> None:
        session = Session.create(Path.cwd())
        self._make_turn(session, "list the files", "Listing.")
        code, out, err = self._run("log")
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertIn(f"[1] session {session.id[:8]}", lines[0])
        self.assertIn("[2] user  list the files", out)
        self.assertIn("[3] assistant  Listing.", out)
        self.assertIn("[3] tool_call bash  ls -la", out)
        self.assertIn("[4] tool_result bash", out)
        self.assertIn("file-a file-b", out)  # collapsed onto one line

    def test_corrupt_line_keeps_its_seq(self) -> None:
        session = Session.create(Path.cwd())
        session.append_message(ModelRequest(parts=[UserPromptPart(content="hi")]))
        with session.path.open("a", encoding="utf-8") as file:
            file.write("not json\n")
        session.append_message(ModelRequest(parts=[UserPromptPart(content="again")]))

        code, out, err = self._run("log")
        self.assertEqual(code, 0)
        self.assertIn("[3] <corrupt>", out)
        self.assertIn("[4] user  again", out)

        code, out, err = self._run("log", "--json")
        payload = [json.loads(line) for line in out.splitlines()]
        self.assertEqual(payload[2], {"seq": 3, "corrupt": True})
        self.assertEqual([entry["seq"] for entry in payload], [1, 2, 3, 4])

    def test_after_returns_only_newer_records(self) -> None:
        session = Session.create(Path.cwd())
        self._make_turn(session, "first", "one")
        self._make_turn(session, "second", "two")
        code, out, err = self._run("log", "--after", "4")
        self.assertEqual(code, 0)
        self.assertNotIn("first", out)
        self.assertIn("[5] user  second", out)

    def test_turns_starts_at_the_last_user_prompt(self) -> None:
        session = Session.create(Path.cwd())
        self._make_turn(session, "first question", "one")
        self._make_turn(session, "second question", "two")
        code, out, err = self._run("log", "--turns", "1")
        self.assertEqual(code, 0)
        self.assertNotIn("first question", out)
        self.assertIn("second question", out)
        self.assertIn("tool_result", out)

    def test_turns_larger_than_history_shows_everything(self) -> None:
        session = Session.create(Path.cwd())
        self._make_turn(session, "only turn", "done")
        code, out, err = self._run("log", "--turns", "5")
        self.assertEqual(code, 0)
        self.assertIn("only turn", out)

    def test_tail_returns_the_last_records(self) -> None:
        session = Session.create(Path.cwd())
        self._make_turn(session, "first", "one")
        code, out, err = self._run("log", "--tail", "1")
        self.assertEqual(code, 0)
        self.assertNotIn("user", out)
        self.assertIn("[4] tool_result bash", out)

    def test_json_merges_seq_into_the_record(self) -> None:
        session = Session.create(Path.cwd())
        session.append_message(ModelRequest(parts=[UserPromptPart(content="hi")]))
        code, out, err = self._run("log", "--json", "--tail", "1")
        self.assertEqual(code, 0)
        record = json.loads(out)
        self.assertEqual(record["seq"], 2)
        self.assertEqual(record["type"], "message")

    def test_id_prefix_selects_a_session_and_default_is_latest(self) -> None:
        older = Session.create(Path.cwd())
        older.append_message(ModelRequest(parts=[UserPromptPart(content="old prompt")]))
        newer = Session.create(Path.cwd())
        newer.append_message(ModelRequest(parts=[UserPromptPart(content="new prompt")]))
        past = older.path.stat().st_mtime - 60
        os.utime(older.path, (past, past))

        code, out, err = self._run("log")
        self.assertIn("new prompt", out)
        code, out, err = self._run("log", older.id[:8])
        self.assertIn("old prompt", out)

    def test_no_session_exits_1(self) -> None:
        code, out, err = self._run("log")
        self.assertEqual(code, 1)
        self.assertIn("no session", err)

    def test_unknown_prefix_exits_1(self) -> None:
        session = Session.create(Path.cwd())
        session.append_message(ModelRequest(parts=[UserPromptPart(content="hi")]))
        code, out, err = self._run("log", "ffffffff")
        self.assertEqual(code, 1)
        self.assertIn("no session matching", err)

    def test_window_flags_are_mutually_exclusive(self) -> None:
        code, out, err = self._run("log", "--after", "1", "--tail", "2")
        self.assertEqual(code, 2)

    def test_full_expands_the_system_prompt(self) -> None:
        session = Session.create(Path.cwd())
        session.append_system_prompt("You are terse.\nAlways.")
        session.append_message(ModelRequest(parts=[UserPromptPart(content="hi")]))
        code, out, err = self._run("log")
        self.assertIn("[2] system_prompt (22 chars)", out)
        self.assertNotIn("You are terse.", out)
        code, out, err = self._run("log", "--full")
        self.assertIn("You are terse.", out)


class InstallSkillTest(CommandTestCase):
    def test_dest_overrides_the_target_directory(self) -> None:
        dest = self.home / "anywhere"
        code, out, err = self._run("install-skill", "--dest", str(dest))
        self.assertEqual(code, 0)
        content = (dest / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("name: paimon", content)
        self.assertIn(str(dest / "SKILL.md"), out)

    def test_default_target_is_claude(self) -> None:
        with patch("paimon.commands.Path.home", return_value=self.home):
            code, out, err = self._run("install-skill")
        self.assertEqual(code, 0)
        self.assertTrue((self.home / ".claude" / "skills" / "paimon" / "SKILL.md").is_file())

    def test_codex_target(self) -> None:
        with patch("paimon.commands.Path.home", return_value=self.home):
            code, out, err = self._run("install-skill", "--target", "codex")
        self.assertEqual(code, 0)
        self.assertTrue((self.home / ".codex" / "skills" / "paimon" / "SKILL.md").is_file())


class VersionTest(unittest.TestCase):
    def test_version_is_a_string(self) -> None:
        self.assertIsInstance(commands.version(), str)


if __name__ == "__main__":
    unittest.main()
