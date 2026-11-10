import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart

from paimon.session import job_message
from paimon.tools.base import MAX_OUTPUT, ToolContext
from paimon.tools.permissions import MODES, READ_DENIAL, USER_DENIAL, gate, run_tool
from paimon.tools.registry import (REGISTRY, _read_history, _search_history, execute_tool,
                                   validate_args)
from paimon.tools.result import ToolError
from paimon.tools.shell import _shell
from tests.support.agent import make_session
from tests.support.shell import sleeper


class RunToolTest(unittest.IsolatedAsyncioTestCase):
    """run_tool is the enforcement point: gating cannot be bypassed by omitting the hooks."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.outside = Path(tmp.name).resolve()
        self.cwd = self.outside / "project"
        self.cwd.mkdir()
        # A write auto mode does not let through on its own.
        self.held = {"path": "../a.txt", "content": "hi"}

    async def test_read_mode_refuses_without_asking_anyone(self) -> None:
        confirm = AsyncMock(return_value=True)
        review = AsyncMock(return_value=None)
        result = await run_tool("write_file", {"path": "a.txt", "content": "hi"},
                                self.cwd, "read", confirm, review=review)
        confirm.assert_not_awaited()
        review.assert_not_awaited()
        self.assertEqual(result.status, "denied")
        self.assertEqual(result.text, READ_DENIAL)
        self.assertFalse((self.cwd / "a.txt").exists())

    async def test_auto_mode_runs_what_the_reviewer_lets_through(self) -> None:
        confirm = AsyncMock(return_value=False)
        review = AsyncMock(return_value=None)
        result = await run_tool("write_file", self.held, self.cwd, "auto", confirm,
                                review=review)
        review.assert_awaited_once_with("write_file", self.held)
        confirm.assert_not_awaited()
        self.assertEqual(result.status, "success")
        self.assertEqual((self.outside / "a.txt").read_text(), "hi")

    async def test_auto_mode_reports_the_reviewers_refusal(self) -> None:
        review = AsyncMock(return_value="Denied by the reviewer: no.")
        result = await run_tool("write_file", self.held, self.cwd, "auto", review=review)
        self.assertEqual(result.status, "denied")
        self.assertEqual(result.text, "Denied by the reviewer: no.")
        self.assertFalse((self.outside / "a.txt").exists())

    async def test_without_reviewer_or_confirm_hook_held_calls_are_denied(self) -> None:
        result = await run_tool("write_file", self.held, self.cwd, "auto")
        self.assertEqual(result.status, "denied")
        self.assertEqual(result.text, USER_DENIAL)
        self.assertFalse((self.outside / "a.txt").exists())

    async def test_without_a_reviewer_the_confirm_hook_decides(self) -> None:
        confirm = AsyncMock(return_value=True)
        result = await run_tool("write_file", self.held, self.cwd, "auto", confirm)
        confirm.assert_awaited_once()
        self.assertEqual(result.status, "success")
        self.assertIn("Wrote", result.text)
        self.assertEqual((self.outside / "a.txt").read_text(), "hi")

    async def test_allowed_calls_skip_the_hooks(self) -> None:
        (self.cwd / "a.txt").write_text("hi")
        confirm = AsyncMock(return_value=False)
        review = AsyncMock(return_value="no")
        result = await run_tool("read_file", {"path": "a.txt"}, self.cwd, "read", confirm,
                                review=review)
        confirm.assert_not_awaited()
        review.assert_not_awaited()
        self.assertEqual(result.status, "success")
        self.assertIn("hi", result.text)

    @unittest.skipIf(os.name == "nt", "nothing is auto-allowed under cmd.exe")
    async def test_safe_command_runs_in_read_mode(self) -> None:
        result = await run_tool("shell", {"command": "ls"}, self.cwd, "read")
        self.assertEqual(result.status, "success")
        self.assertNotEqual(result.text, READ_DENIAL)

    async def test_strict_denies_safe_command(self) -> None:
        result = await run_tool("shell", {"command": "ls"}, self.cwd, "read", safe_commands=False)
        self.assertEqual(result.status, "denied")
        self.assertEqual(result.text, READ_DENIAL)

    async def test_shell_timeout_terminates_and_reaps_process_tree(self) -> None:
        with (
            patch("paimon.tools.shell._COMMAND_TIMEOUT", 0.05),
            patch("paimon.tools.shell._KILL_GRACE", 0.05),
            patch("paimon.tools.shell._KILL_TIMEOUT", 0.5),
        ):
            result = await _shell({"command": sleeper(ignore_term=True)}, self.cwd)

        self.assertIn("(timed out after 0.05s)", result.text)
        self.assertEqual((result.status, result.detail), ("failed", "timed out"))


class LineEndingPreservationTest(unittest.TestCase):
    """TOOLS-2: editing changes the target span only — line endings, a missing
    final newline and a UTF-8 BOM all survive."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()
        self.path = self.cwd / "f.txt"

    def _edit(self, old: str, new: str) -> str:
        from paimon.tools.files import _edit_file
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
        from paimon.tools.files import _write_file
        self.path.write_bytes(b"\xef\xbb\xbfold\r\n")
        _write_file({"path": "f.txt", "content": "p\nq\n"}, self.cwd)
        self.assertEqual(self.path.read_bytes(), b"\xef\xbb\xbfp\r\nq\r\n")

    def test_write_file_creates_new_files_as_plain_lf_utf8(self) -> None:
        from paimon.tools.files import _write_file
        _write_file({"path": "f.txt", "content": "p\nq\n"}, self.cwd)
        self.assertEqual(self.path.read_bytes(), b"p\nq\n")

    def test_edit_refuses_a_file_that_is_not_utf8(self) -> None:
        """Decoded lossily and written back, every GBK byte outside the edit
        would turn into U+FFFD."""
        original = "中文\nb\n".encode("gbk")
        self.path.write_bytes(original)
        with self.assertRaises(ToolError) as caught:
            self._edit("b", "B")
        result = caught.exception.text
        self.assertTrue(result.startswith("Error:"), result)
        self.assertIn("UTF-8", result)
        self.assertIn("shell", result)
        self.assertEqual(self.path.read_bytes(), original)

    def test_write_file_replaces_a_file_that_is_not_utf8(self) -> None:
        from paimon.tools.files import _write_file
        self.path.write_bytes("café\r\n".encode("latin-1"))
        _write_file({"path": "f.txt", "content": "p\nq\n"}, self.cwd)
        self.assertEqual(self.path.read_bytes(), b"p\r\nq\r\n")

    def test_read_file_shows_a_file_that_is_not_utf8(self) -> None:
        from paimon.tools.files import _read_file
        self.path.write_bytes("café\nb\n".encode("latin-1"))
        result = _read_file({"path": "f.txt"}, self.cwd)
        self.assertIn("caf�", result)
        self.assertIn("b", result)

    def test_read_file_shows_crlf_files_without_stray_carriage_returns(self) -> None:
        from paimon.tools.files import _read_file
        self.path.write_bytes(b"a\r\nb\r\n")
        result = _read_file({"path": "f.txt"}, self.cwd)
        self.assertNotIn("\r", result)
        self.assertIn("a", result)


class ReadFileTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()
        (self.cwd / "f.txt").write_text("a\nb\nc\n")

    def _read(self, **args) -> str:
        from paimon.tools.files import _read_file
        return _read_file({"path": "f.txt", **args}, self.cwd)

    def test_offset_and_limit_select_a_window(self) -> None:
        self.assertEqual(self._read(offset=2, limit=1), "2  b")

    def test_a_limit_of_zero_is_an_error_not_the_whole_file(self) -> None:
        for limit in (0, -1):
            with self.assertRaisesRegex(ToolError, "limit must be at least 1"):
                self._read(limit=limit)

    def test_an_offset_below_one_is_an_error_not_clamped(self) -> None:
        for offset in (0, -3):
            with self.assertRaisesRegex(ToolError, "1-indexed"):
                self._read(offset=offset)


class GlobTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()

    def write(self, relative: str) -> Path:
        path = self.cwd / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n")
        return path

    def _glob(self, pattern: str) -> list[str]:
        from paimon.tools.files import _glob
        return _glob({"pattern": pattern}, self.cwd).splitlines()

    def test_ignored_directories_are_skipped_but_files_of_those_names_are_not(self) -> None:
        self.write("build/a.txt")
        self.write("src/node_modules/b.txt")
        build = self.write("scripts/build")
        out = self.write("out")
        self.assertEqual(sorted(self._glob("**/*")), sorted([str(build), str(out)]))

    def test_newest_first(self) -> None:
        old, new = self.write("old.txt"), self.write("new.txt")
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))
        self.assertEqual(self._glob("*.txt"), [str(new), str(old)])

    def test_a_file_deleted_while_listing_is_left_out(self) -> None:
        kept, gone = self.write("kept.txt"), self.write("gone.txt")
        stat = Path.stat

        def vanishing(path, *args, **kwargs):
            # Still there when the listing checks it, gone when it is timed.
            if path == gone and sys._getframe(1).f_code.co_name == "_glob":
                raise FileNotFoundError(path)
            return stat(path, *args, **kwargs)

        with patch.object(Path, "stat", vanishing):
            self.assertEqual(self._glob("*.txt"), [str(kept)])


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
        self.assertIn("'todos' must be a valid list",
                      validate_args("write_todos", {"todos": "oops"}, REGISTRY))
        self.assertIn("'command' must be a valid string",
                      validate_args("shell", {"command": 42}, REGISTRY))

    def test_wrong_array_item_shape(self) -> None:
        self.assertIn("'todos[0]' must be a valid dictionary",
                      validate_args("write_todos", {"todos": ["x"]}, REGISTRY))
        self.assertIn("missing required argument 'status'",
                      validate_args("write_todos", {"todos": [{"content": "x"}]}, REGISTRY))

    def test_wrong_enum_value(self) -> None:
        error = validate_args(
            "write_todos", {"todos": [{"content": "x", "status": "bogus"}]}, REGISTRY)
        self.assertIn("'todos[0].status' must be 'pending'", error)

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

    def test_a_settled_slot_is_searched_and_read_at_its_own_seq(self) -> None:
        request = ModelRequest(parts=[
            ToolReturnPart(tool_name="shell", content="Interrupted by user.", tool_call_id=call_id)
            for call_id in ("call-1", "call-2")])
        record_id = self.session.append_message(request)
        for index in (0, 1):
            request.parts[index].content = f"settled output {index}"
            self.session.append_tool_result(record_id, index, request)

        self.assertIn("No matches", _search_history({"query": "Interrupted"}, self.ctx))
        result = _search_history({"query": "settled output 1"}, self.ctx)
        self.assertIn("[4] tool_result shell", result)
        placeholder = _read_history({"seq": 2}, self.ctx)
        self.assertNotIn("Interrupted by user.", placeholder)
        self.assertIn("seq 3, 4", placeholder)
        self.assertIn("settled output 0", _read_history({"seq": 3}, self.ctx))

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
        for seq in (99, 0):
            with self.assertRaisesRegex(ToolError, "out of range"):
                _read_history({"seq": seq}, self.ctx)

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
        for name, args in (("search_history", {"query": "x"}), ("read_history", {"seq": 1})):
            result = asyncio.run(execute_tool(name, args, self.cwd, ctx=ctx))
            self.assertEqual((result.text, result.status),
                             ("Error: no session log available", "failed"))

    def test_a_job_notice_is_labelled_as_one_not_as_the_user(self) -> None:
        self.session.append_message(job_message("agent ab12 finished:\nneedle found"))
        self._user("[job] needle typed by the user")
        result = _search_history({"query": "needle"}, self.ctx)
        self.assertIn("[2] job: ", result)
        self.assertIn("[3] user: ", result)

    def test_history_tools_are_always_allowed_reads(self) -> None:
        for mode in MODES:
            self.assertEqual(gate("search_history", {"query": "x"}, mode, self.cwd), "allow")
            self.assertEqual(gate("read_history", {"seq": 1}, mode, self.cwd), "allow")
