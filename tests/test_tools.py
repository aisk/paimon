import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart

from paimon.tools import (
    MAX_OUTPUT,
    MODES,
    REGISTRY,
    ToolContext,
    _read_history,
    _search_history,
    _shell,
    gate,
    run_tool,
    validate_args,
)
from tests.support.agent import make_session


class RunToolTest(unittest.IsolatedAsyncioTestCase):
    """run_tool is the enforcement point: gating cannot be bypassed by omitting the hook."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()

    async def test_without_confirm_hook_dangerous_calls_are_denied(self) -> None:
        result, denied = await run_tool("write_file", {"path": "a.txt", "content": "hi"}, self.cwd, "read")
        self.assertTrue(denied)
        self.assertEqual(result, "User denied this operation.")
        self.assertFalse((self.cwd / "a.txt").exists())

    async def test_confirm_hook_allows_execution(self) -> None:
        confirm = AsyncMock(return_value=True)
        result, denied = await run_tool("write_file", {"path": "a.txt", "content": "hi"}, self.cwd, "read", confirm)
        confirm.assert_awaited_once()
        self.assertFalse(denied)
        self.assertIn("Wrote", result)
        self.assertEqual((self.cwd / "a.txt").read_text(), "hi")

    async def test_allowed_calls_skip_the_hook(self) -> None:
        (self.cwd / "a.txt").write_text("hi")
        confirm = AsyncMock(return_value=False)
        result, denied = await run_tool("read_file", {"path": "a.txt"}, self.cwd, "read", confirm)
        confirm.assert_not_awaited()
        self.assertFalse(denied)
        self.assertIn("hi", result)

    async def test_safe_command_runs_without_confirm_hook(self) -> None:
        result, denied = await run_tool("shell", {"command": "ls"}, self.cwd, "read")
        self.assertFalse(denied)
        self.assertNotEqual(result, "User denied this operation.")

    async def test_strict_denies_safe_command_without_hook(self) -> None:
        result, denied = await run_tool("shell", {"command": "ls"}, self.cwd, "read", safe_commands=False)
        self.assertTrue(denied)
        self.assertEqual(result, "User denied this operation.")

    async def test_shell_timeout_terminates_and_reaps_process_tree(self) -> None:
        with (
            patch("paimon.tools._COMMAND_TIMEOUT", 0.05),
            patch("paimon.tools._KILL_GRACE", 0.05),
            patch("paimon.tools._KILL_TIMEOUT", 0.5),
        ):
            result = await _shell({"command": "trap '' TERM; sleep 30"}, self.cwd)

        self.assertIn("(timed out after 0.05s)", result)


class LineEndingPreservationTest(unittest.TestCase):
    """TOOLS-2: editing changes the target span only — line endings, a missing
    final newline and a UTF-8 BOM all survive."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()
        self.path = self.cwd / "f.txt"

    def _edit(self, old: str, new: str) -> str:
        from paimon.tools import _edit_file
        return _edit_file({"path": "f.txt", "old_string": old, "new_string": new}, self.cwd)

    def test_crlf_file_stays_crlf(self) -> None:
        self.path.write_bytes(b"a\r\nb\r\nc\r\n")
        self.assertIn("Edited", self._edit("b", "B"))
        self.assertEqual(self.path.read_bytes(), b"a\r\nB\r\nc\r\n")

    def test_lf_file_stays_lf(self) -> None:
        self.path.write_bytes(b"a\nb\nc\n")
        self._edit("b", "B")
        self.assertEqual(self.path.read_bytes(), b"a\nB\nc\n")

    def test_missing_final_newline_is_not_added(self) -> None:
        self.path.write_bytes(b"a\nb")
        self._edit("b", "B")
        self.assertEqual(self.path.read_bytes(), b"a\nB")

    def test_utf8_bom_survives(self) -> None:
        self.path.write_bytes(b"\xef\xbb\xbfx\ny\n")
        self._edit("x", "X")
        self.assertEqual(self.path.read_bytes(), b"\xef\xbb\xbfX\ny\n")

    def test_lf_old_string_matches_a_crlf_file(self) -> None:
        """The model reads LF (read_file normalizes), so its old_string spans
        lines with LF; the match must still land on a CRLF file."""
        self.path.write_bytes(b"a\r\nb\r\nc\r\n")
        self._edit("b\nc", "B\nC")
        self.assertEqual(self.path.read_bytes(), b"a\r\nB\r\nC\r\n")

    def test_write_file_keeps_an_existing_files_conventions(self) -> None:
        from paimon.tools import _write_file
        self.path.write_bytes(b"\xef\xbb\xbfold\r\n")
        _write_file({"path": "f.txt", "content": "p\nq\n"}, self.cwd)
        self.assertEqual(self.path.read_bytes(), b"\xef\xbb\xbfp\r\nq\r\n")

    def test_write_file_creates_new_files_as_plain_lf_utf8(self) -> None:
        from paimon.tools import _write_file
        _write_file({"path": "f.txt", "content": "p\nq\n"}, self.cwd)
        self.assertEqual(self.path.read_bytes(), b"p\nq\n")

    def test_read_file_shows_crlf_files_without_stray_carriage_returns(self) -> None:
        from paimon.tools import _read_file
        self.path.write_bytes(b"a\r\nb\r\n")
        result = _read_file({"path": "f.txt"}, self.cwd)
        self.assertNotIn("\r", result)
        self.assertIn("a", result)


class ValidateArgsTest(unittest.TestCase):
    """TOOLS-1: one schema check gates every tool call, agent-handled included."""

    def test_valid_args_pass(self) -> None:
        self.assertIsNone(validate_args("shell", {"command": "ls"}, REGISTRY))
        self.assertIsNone(validate_args(
            "write_todos",
            {"todos": [{"content": "x", "status": "pending"}]}, REGISTRY))

    def test_missing_required_argument(self) -> None:
        self.assertIn("missing required argument 'command'",
                      validate_args("shell", {}, REGISTRY))
        self.assertIn("missing required argument 'job_id'",
                      validate_args("read_job", {}, REGISTRY))

    def test_wrong_top_level_type(self) -> None:
        self.assertIn("'todos' must be an array",
                      validate_args("write_todos", {"todos": "oops"}, REGISTRY))
        self.assertIn("'command' must be a string",
                      validate_args("shell", {"command": 42}, REGISTRY))

    def test_wrong_array_item_shape(self) -> None:
        self.assertIn("'todos[0]' must be an object",
                      validate_args("write_todos", {"todos": ["x"]}, REGISTRY))
        self.assertIn("missing required argument 'status'",
                      validate_args("write_todos", {"todos": [{"content": "x"}]}, REGISTRY))

    def test_wrong_enum_value(self) -> None:
        error = validate_args(
            "write_todos", {"todos": [{"content": "x", "status": "bogus"}]}, REGISTRY)
        self.assertIn("must be one of", error)

    def test_unknown_extra_keys_pass(self) -> None:
        self.assertIsNone(validate_args("shell", {"command": "ls", "stray": 1}, REGISTRY))

    def test_unknown_tool_is_reported(self) -> None:
        self.assertIn("unknown tool", validate_args("nope", {}, REGISTRY))


class HistoryToolsTest(unittest.TestCase):
    """search_history/read_history see the whole log, compaction included."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()
        self.session = make_session(self.cwd)
        self.ctx = ToolContext(session=self.session)

    def _user(self, text: str) -> str:
        return self.session.append_message(ModelRequest(parts=[UserPromptPart(content=text)]))

    def _tool_return(self, tool_name: str, content: str) -> str:
        part = ToolReturnPart(tool_name=tool_name, content=content, tool_call_id="call-1")
        return self.session.append_message(ModelRequest(parts=[part]))

    def test_search_finds_content_from_before_a_compaction(self) -> None:
        self._user("the database password lives in vault.yaml")
        self.session.append_compaction("earlier work summarized", [], 1000)
        self._user("carry on")
        result = _search_history({"query": "vault"}, self.ctx)
        self.assertIn("vault.yaml", result)
        self.assertIn("[2] user", result, "the hit keeps its pre-compaction seq")

    def test_compaction_summaries_match_but_kept_messages_do_not_double_count(self) -> None:
        message = ModelRequest(parts=[UserPromptPart(content="unique-marker in a kept message")])
        self.session.append_message(message)
        self.session.append_compaction("summary mentions unique-marker too", [message], 1000)
        result = _search_history({"query": "unique-marker"}, self.ctx)
        self.assertIn("2 matching parts", result,
                      "one hit for the original record, one for the summary, none for kept_messages")
        self.assertIn("compaction summary", result)

    def test_a_replaced_record_is_searched_only_in_its_final_form(self) -> None:
        record_id = self._tool_return("shell", "first-draft output")
        replacement = ModelRequest(parts=[ToolReturnPart(
            tool_name="shell", content="final output", tool_call_id="call-1")])
        self.session.append_message(replacement, replaces=record_id)
        self.assertIn("No matches", _search_history({"query": "first-draft"}, self.ctx))
        result = _search_history({"query": "final output"}, self.ctx)
        self.assertIn("[3]", result, "the hit carries the replacement line's seq")

    def test_the_history_tools_own_output_is_not_searched(self) -> None:
        self._tool_return("search_history", "[2] user: needle from an earlier search")
        self._tool_return("shell", "a real needle")
        result = _search_history({"query": "needle"}, self.ctx)
        self.assertIn("1 matching part", result)
        self.assertIn("tool_result shell", result)

    def test_an_invalid_regex_falls_back_to_a_literal_search(self) -> None:
        self._user("weird chars: a[b")
        result = _search_history({"query": "a[b"}, self.ctx)
        self.assertIn("searched literally", result)
        self.assertIn("a[b", result)

    def test_a_corrupt_line_keeps_later_seq_numbers_in_place(self) -> None:
        self._user("before the corruption")
        with self.session.path.open("a", encoding="utf-8") as file:
            file.write("this is not json\n")
        self._user("after the corruption")
        result = _search_history({"query": "after the corruption"}, self.ctx)
        self.assertIn("[4] user", result)
        self.assertIn("<corrupt>", _read_history({"seq": 3}, self.ctx))

    def test_max_results_truncation_reports_the_total(self) -> None:
        for i in range(5):
            self._user(f"needle number {i}")
        result = _search_history({"query": "needle", "max_results": 2}, self.ctx)
        self.assertIn("5 matching parts", result)
        self.assertIn("showing the first 2", result)
        self.assertEqual(len(result.splitlines()), 3)

    def test_read_history_returns_full_records_by_seq(self) -> None:
        self._user("short question")
        self._tool_return("shell", "line one\nline two")
        result = _read_history({"seq": 2, "count": 2}, self.ctx)
        self.assertIn("[2] user  short question", result)
        self.assertIn("line one\nline two", result, "full mode keeps newlines")

    def test_read_history_does_not_return_a_superseded_revision(self) -> None:
        """LOG-1: the pre-seeded interrupted placeholder must not be handed to
        the model as if the interruption really happened."""
        record_id = self._tool_return("shell", "Interrupted by user.")
        replacement = ModelRequest(parts=[ToolReturnPart(
            tool_name="shell", content="ok (exit code 0)", tool_call_id="call-1")])
        self.session.append_message(replacement, replaces=record_id)
        result = _read_history({"seq": 2}, self.ctx)
        self.assertNotIn("Interrupted by user.", result)
        self.assertIn("superseded revision", result)
        self.assertIn("seq 3", result)
        self.assertIn("ok (exit code 0)", _read_history({"seq": 3}, self.ctx))

    def test_read_history_rejects_a_seq_outside_the_log(self) -> None:
        self._user("only entry")
        self.assertIn("out of range", _read_history({"seq": 99}, self.ctx))
        self.assertIn("out of range", _read_history({"seq": 0}, self.ctx))

    def test_read_history_stops_at_its_output_budget_with_a_note(self) -> None:
        self._user("x" * (MAX_OUTPUT - 2000))
        self._user("y" * (MAX_OUTPUT - 2000))
        result = _read_history({"seq": 2, "count": 2}, self.ctx)
        self.assertLess(len(result), MAX_OUTPUT)
        self.assertIn("stopped before seq 3", result)

    def test_a_single_oversized_record_is_truncated_rather_than_dropped(self) -> None:
        self._user("z" * (MAX_OUTPUT + 5000))
        result = _read_history({"seq": 2}, self.ctx)
        self.assertLess(len(result), MAX_OUTPUT)
        self.assertIn("record truncated", result)

    def test_both_tools_error_readably_without_a_session(self) -> None:
        ctx = ToolContext()
        self.assertEqual(_search_history({"query": "x"}, ctx), "Error: no session log available")
        self.assertEqual(_read_history({"seq": 1}, ctx), "Error: no session log available")

    def test_history_tools_are_always_allowed_reads(self) -> None:
        for mode in MODES:
            self.assertEqual(gate("search_history", {"query": "x"}, mode, self.cwd), "allow")
            self.assertEqual(gate("read_history", {"seq": 1}, mode, self.cwd), "allow")
