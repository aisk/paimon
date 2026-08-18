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
        self._write_config(model="zai:glm-4.7",
                           providers={"zai": {"api_key": "sk-secret"}})
        code, out, err = self._run("status")
        self.assertEqual(code, 0)
        self.assertIn("zai:glm-4.7", out)
        self.assertIn("api key set", out)

    def test_json_reports_state_without_the_key(self) -> None:
        self._write_config(model="zai:glm-4.7",
                           providers={"zai": {"api_key": "sk-secret",
                                              "api_base": "https://x/v1"}})
        code, out, err = self._run("status", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(payload["configured"])
        self.assertTrue(payload["ready"])
        self.assertIsNone(payload["error"])
        self.assertEqual(payload["model"], "zai:glm-4.7")
        self.assertEqual(payload["api_base"], "https://x/v1")
        self.assertTrue(payload["api_key_set"])
        self.assertEqual(payload["sessions_here"], 0)
        self.assertNotIn("sk-secret", out)

    def test_status_reports_the_current_providers_credentials_only(self) -> None:
        """AUTH-1: a key stored for another provider is not this model's key."""
        self._write_config(model="zai:glm-4.7",
                           providers={"openai": {"api_key": "sk-openai",
                                                 "api_base": "https://o/v1"}})
        code, out, err = self._run("status", "--json")
        payload = json.loads(out)
        self.assertFalse(payload["api_key_set"])
        self.assertIsNone(payload["api_base"])

    def test_json_unconfigured(self) -> None:
        code, out, err = self._run("status", "--json")
        self.assertEqual(code, 1)
        payload = json.loads(out)
        self.assertFalse(payload["configured"])
        self.assertFalse(payload["ready"])
        self.assertIsNone(payload["model"])
        self.assertFalse(payload["api_key_set"])

    def test_configured_model_without_credentials_is_not_ready(self) -> None:
        """CLI-1: a model whose credentials cannot resolve must not exit 0."""
        self._write_config(model="zai:glm-4.7")
        with patch.dict("os.environ", {"ZAI_API_KEY": ""}):
            code, out, err = self._run("status", "--json")
        self.assertEqual(code, 1)
        payload = json.loads(out)
        self.assertTrue(payload["configured"])
        self.assertFalse(payload["ready"])
        self.assertIn("ZAI_API_KEY", payload["error"])

    def test_environment_credentials_make_it_ready_without_leaking(self) -> None:
        self._write_config(model="zai:glm-4.7")
        with patch.dict("os.environ", {"ZAI_API_KEY": "sk-env-secret"}):
            code, out, err = self._run("status", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(payload["ready"])
        self.assertFalse(payload["api_key_set"], "the key came from env, not the config")
        self.assertNotIn("sk-env-secret", out)

    def test_unconstructible_provider_is_reported_in_text_mode(self) -> None:
        self._write_config(model="nosuchprovider:m")
        code, out, err = self._run("status")
        self.assertEqual(code, 1)
        self.assertIn("not ready:", out)


class LoginTest(CommandTestCase):
    def test_key_from_environment_variable(self) -> None:
        with patch.dict("os.environ", {"MY_KEY": "sk-abc"}):
            code, out, err = self._run("login", "--model", "openai:gpt-5",
                                       "--api-key-env", "MY_KEY")
        self.assertEqual(code, 0)
        data = json.loads(config_path().read_text())
        self.assertEqual(data["model"], "openai:gpt-5")
        self.assertEqual(data["providers"]["openai"]["api_key"], "sk-abc")

    def test_key_from_stdin(self) -> None:
        code, out, err = self._run("login", "--model", "openai:gpt-5",
                                   "--api-key-stdin", stdin="sk-piped\n")
        self.assertEqual(code, 0)
        data = json.loads(config_path().read_text())
        self.assertEqual(data["providers"]["openai"]["api_key"], "sk-piped")

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
        self._write_config(model="zai:glm-4.7", theme="dark",
                           providers={"zai": {"api_key": "sk-old"}})
        code, out, err = self._run("login", "--model", "openai:gpt-5")
        self.assertEqual(code, 0)
        data = json.loads(config_path().read_text())
        self.assertEqual(data["model"], "openai:gpt-5")
        self.assertEqual(data["providers"]["zai"]["api_key"], "sk-old")
        self.assertEqual(data["theme"], "dark")

    def test_corrupt_config_is_refused_with_a_force_hint(self) -> None:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"model": "zai:glm')
        code, out, err = self._run("login", "--model", "openai:gpt-5")
        self.assertEqual(code, 1)
        self.assertIn("--force", err)
        self.assertEqual(path.read_text(), '{"model": "zai:glm')

    def test_force_replaces_a_corrupt_config(self) -> None:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"model": "zai:glm')
        code, out, err = self._run("login", "--model", "openai:gpt-5", "--force")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(path.read_text()), {"model": "openai:gpt-5"})

    def test_empty_api_base_clears_the_stored_override(self) -> None:
        self._write_config(model="zai:glm-4.7",
                           providers={"zai": {"api_base": "https://old/v1"}})
        code, out, err = self._run("login", "--model", "zai:glm-5.2", "--api-base", "")
        self.assertEqual(code, 0)
        self.assertNotIn("providers", json.loads(config_path().read_text()))

    def test_login_leaves_other_providers_credentials_alone(self) -> None:
        self._write_config(model="zai:glm-4.7",
                           providers={"zai": {"api_key": "sk-zai"}})
        with patch.dict("os.environ", {"MY_KEY": "sk-openai"}):
            code, out, err = self._run("login", "--model", "openai:gpt-5",
                                       "--api-key-env", "MY_KEY")
        self.assertEqual(code, 0)
        data = json.loads(config_path().read_text())
        self.assertEqual(data["providers"]["zai"], {"api_key": "sk-zai"})
        self.assertEqual(data["providers"]["openai"], {"api_key": "sk-openai"})


class ProfileTest(CommandTestCase):
    def test_login_and_status_share_a_profile(self) -> None:
        with patch.dict("os.environ", {"WORK_KEY": "sk-work"}):
            code, out, err = self._run("login", "--profile", "work", "--model",
                                       "openai:gpt-5", "--api-key-env", "WORK_KEY")
        self.assertEqual(code, 0)
        profile_config = self.home / "config" / "work" / "config.json"
        stored = json.loads(profile_config.read_text())
        self.assertEqual(stored["providers"]["openai"]["api_key"], "sk-work")

        code, out, err = self._run("status", "--profile", "work", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["model"], "openai:gpt-5")

    def test_default_profile_is_untouched(self) -> None:
        with patch.dict("os.environ", {"WORK_KEY": "sk-work"}):
            self._run("login", "--profile", "work", "--model", "openai:gpt-5",
                      "--api-key-env", "WORK_KEY")
        code, out, err = self._run("status", "--json")
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(out)["configured"])

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

    def test_subagent_sessions_stay_out_of_the_listing_and_of_continue(self) -> None:
        """`paimon -c` goes through latest_session, so an unfiltered list would
        silently resume some agent's session after an afternoon of them."""
        mine = self._make_session("my work")
        child = Session.create(Path.cwd(), parent=mine.id)
        child.append_message(ModelRequest(parts=[UserPromptPart(content="their work")]))

        code, out, err = self._run("sessions", "--json")
        self.assertEqual([entry["id"] for entry in json.loads(out)], [mine.id])
        self.assertEqual(commands.latest_session().id, mine.id)
        # Naming one is still allowed: that is a deliberate act, not a default.
        self.assertEqual(commands.resolve_session(child.id[:8]).id, child.id)

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
            ToolCallPart(tool_name="shell", args={"command": "ls -la"}, tool_call_id="c1"),
        ]))
        session.append_message(ModelRequest(parts=[
            ToolReturnPart(tool_name="shell", content="file-a\nfile-b", tool_call_id="c1"),
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
        self.assertIn("[3] tool_call shell  ls -la", out)
        self.assertIn("[4] tool_result shell", out)
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
        self.assertIn("[4] tool_result shell", out)

    def _make_replaced_tool_turn(self, session: Session, *, replace: bool = True,
                                 twice: bool = False) -> None:
        """A turn the agent's placeholder-then-replace persistence produces."""
        session.append_message(ModelRequest(parts=[UserPromptPart(content="run it")]))
        session.append_message(ModelResponse(parts=[
            ToolCallPart(tool_name="shell", args={"command": "printf ok"}, tool_call_id="c1"),
        ]))
        record_id = session.append_message(ModelRequest(parts=[
            ToolReturnPart(tool_name="shell", content="Interrupted by user.", tool_call_id="c1"),
        ]))
        if replace:
            if twice:
                session.append_message(ModelRequest(parts=[
                    ToolReturnPart(tool_name="shell", content="halfway snapshot",
                                   tool_call_id="c1"),
                ]), replaces=record_id)
            session.append_message(ModelRequest(parts=[
                ToolReturnPart(tool_name="shell", content="ok (exit code 0)", tool_call_id="c1"),
            ]), replaces=record_id)

    def test_replaced_placeholder_is_folded_out_of_the_default_log(self) -> None:
        """LOG-1: a successful tool shows one result, not a fake interruption."""
        session = Session.create(Path.cwd())
        self._make_replaced_tool_turn(session)
        code, out, err = self._run("log")
        self.assertEqual(code, 0)
        self.assertNotIn("Interrupted by user.", out)
        results = [line for line in out.splitlines() if "tool_result" in line]
        self.assertEqual(len(results), 1)
        self.assertIn("ok (exit code 0)", results[0])

    def test_a_real_interruption_still_shows(self) -> None:
        session = Session.create(Path.cwd())
        self._make_replaced_tool_turn(session, replace=False)
        code, out, err = self._run("log")
        self.assertIn("Interrupted by user.", out)

    def test_intermediate_snapshots_are_folded_too(self) -> None:
        session = Session.create(Path.cwd())
        self._make_replaced_tool_turn(session, twice=True)
        code, out, err = self._run("log")
        self.assertNotIn("halfway snapshot", out)
        self.assertNotIn("Interrupted by user.", out)
        self.assertIn("ok (exit code 0)", out)

    def test_json_keeps_every_revision_for_audit(self) -> None:
        session = Session.create(Path.cwd())
        self._make_replaced_tool_turn(session)
        code, out, err = self._run("log", "--json")
        self.assertIn("Interrupted by user.", out)
        self.assertIn("ok (exit code 0)", out)

    def test_after_window_still_folds_across_the_cursor(self) -> None:
        """The placeholder may precede the --after cursor while its replacement
        follows it; the map has to be computed over the whole log."""
        session = Session.create(Path.cwd())
        self._make_replaced_tool_turn(session)
        # Cursor right on the placeholder record (seq 4): only the replacement
        # is in the window, and it renders as the one real result.
        code, out, err = self._run("log", "--after", "4")
        self.assertNotIn("Interrupted by user.", out)
        self.assertIn("ok (exit code 0)", out)

    def test_turn_end_records_render_with_error_and_partial(self) -> None:
        session = Session.create(Path.cwd())
        session.append_message(ModelRequest(parts=[UserPromptPart(content="hi")]))
        session.append_meta("turn_end", outcome="error",
                            error="RuntimeError: boom", partial_text="half an answer")
        code, out, err = self._run("log")
        self.assertEqual(code, 0)
        self.assertIn("turn_end error  RuntimeError: boom", out)
        self.assertIn("partial", out)
        code, out, err = self._run("log", "--full")
        self.assertIn("half an answer", out)

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
