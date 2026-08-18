import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from paimon.session import Session, SessionBusyError, _project_dir


class SessionScanTestCase(unittest.TestCase):
    """Sessions created via the real API in an isolated PAIMON_DATA_HOME."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = patch.dict("os.environ", {"PAIMON_DATA_HOME": tmp.name})
        env.start()
        self.addCleanup(env.stop)
        self.cwd = Path(tmp.name) / "project"
        self.cwd.mkdir()

    def _session_with_message(self, content: str, mtime: float) -> Session:
        session = Session.create(self.cwd)
        session.append_message(ModelRequest(parts=[UserPromptPart(content=content)]))
        os.utime(session.path, (mtime, mtime))
        return session


class ListTest(SessionScanTestCase):
    def test_newest_first_and_empty_sessions_excluded(self) -> None:
        old = self._session_with_message("old", mtime=1_000)
        new = self._session_with_message("new", mtime=2_000)
        empty = Session.create(self.cwd)

        listed = Session.list(self.cwd)

        self.assertEqual([session.id for session in listed], [new.id, old.id])
        self.assertNotIn(empty.id, [session.id for session in listed])

    def test_invalid_files_are_skipped(self) -> None:
        session = self._session_with_message("hi", mtime=1_000)
        (_project_dir(self.cwd) / "garbage.jsonl").write_text("not json\n")
        (_project_dir(self.cwd) / "missing-id.jsonl").write_text(
            '{"type": "session"}\n'
        )

        self.assertEqual([s.id for s in Session.list(self.cwd)], [session.id])

    def test_no_project_dir_gives_empty_list(self) -> None:
        self.assertEqual(Session.list(self.cwd / "elsewhere"), [])


class PreviewTest(SessionScanTestCase):
    def test_created_at_from_header(self) -> None:
        session = Session.create(self.cwd)
        created = session.created_at()
        self.assertIsNotNone(created)
        self.assertIn("T", created)

    def test_first_user_text_skips_assistant_messages(self) -> None:
        session = Session.create(self.cwd)
        session.append_message(ModelResponse(parts=[TextPart(content="hello!")]))
        session.append_message(ModelRequest(parts=[UserPromptPart(content="fix the bug")]))
        session.append_message(ModelRequest(parts=[UserPromptPart(content="second")]))

        self.assertEqual(session.first_user_text(), "fix the bug")

    def test_first_user_text_none_for_empty_session(self) -> None:
        self.assertIsNone(Session.create(self.cwd).first_user_text())


class EntriesTest(SessionScanTestCase):
    def test_seq_is_the_physical_line_number_and_corrupt_lines_hold_their_place(self) -> None:
        session = Session.create(self.cwd)
        session.append_message(ModelRequest(parts=[UserPromptPart(content="hi")]))
        with session.path.open("a", encoding="utf-8") as file:
            file.write("not json\n")
        session.append_message(ModelResponse(parts=[TextPart(content="hello!")]))

        entries = session.entries()

        self.assertEqual([seq for seq, _ in entries], [1, 2, 3, 4])
        self.assertEqual(entries[0][1]["type"], "session")
        self.assertIsNone(entries[2][1], "the corrupt line stays as a placeholder")
        self.assertEqual(entries[3][1]["type"], "message")

    def test_unreadable_file_raises_instead_of_yielding_nothing(self) -> None:
        session = Session.create(self.cwd)
        session.path.unlink()
        with self.assertRaises(OSError):
            session.entries()


class OrphanToolCallTest(SessionScanTestCase):
    """SESSION-4: a crash between the tool-call response and its pre-seeded
    results must not leave a history providers reject."""

    def _tool_call_response(self, *ids: str) -> ModelResponse:
        return ModelResponse(parts=[
            ToolCallPart(tool_name="shell", args={"command": "ls"}, tool_call_id=call_id)
            for call_id in ids
        ])

    def test_a_log_ending_on_a_tool_call_gets_a_synthesized_result(self) -> None:
        session = Session.create(self.cwd)
        session.append_message(ModelRequest(parts=[UserPromptPart(content="go")]))
        session.append_message(self._tool_call_response("c1", "c2"))

        messages = session.messages()

        last = messages[-1]
        self.assertIsInstance(last, ModelRequest)
        returns = [p for p in last.parts if isinstance(p, ToolReturnPart)]
        self.assertEqual([r.tool_call_id for r in returns], ["c1", "c2"])
        self.assertIn("Interrupted", returns[0].content)
        # Deterministic: a second load produces the same repair shape.
        again = session.messages()
        self.assertEqual(len(again), len(messages))
        self.assertEqual([p.tool_call_id for p in again[-1].parts], ["c1", "c2"])

    def test_a_user_prompt_after_the_orphan_stays_after_the_repair(self) -> None:
        session = Session.create(self.cwd)
        session.append_message(self._tool_call_response("c1"))
        session.append_message(ModelRequest(parts=[UserPromptPart(content="resumed")]))

        messages = session.messages()

        self.assertIsInstance(messages[1], ModelRequest)
        self.assertEqual(messages[1].parts[0].tool_call_id, "c1")
        self.assertEqual(messages[2].parts[0].content, "resumed")

    def test_a_partially_answered_batch_is_completed_in_place(self) -> None:
        session = Session.create(self.cwd)
        session.append_message(self._tool_call_response("c1", "c2"))
        session.append_message(ModelRequest(parts=[
            ToolReturnPart(tool_name="shell", content="ok", tool_call_id="c1"),
        ]))

        messages = session.messages()

        self.assertEqual(len(messages), 2, "no extra request is inserted")
        answered = {p.tool_call_id: p.content for p in messages[1].parts
                    if isinstance(p, ToolReturnPart)}
        self.assertEqual(answered["c1"], "ok")
        self.assertIn("Interrupted", answered["c2"])

    def test_a_complete_batch_is_untouched(self) -> None:
        session = Session.create(self.cwd)
        session.append_message(self._tool_call_response("c1"))
        session.append_message(ModelRequest(parts=[
            ToolReturnPart(tool_name="shell", content="fine", tool_call_id="c1"),
        ]))

        messages = session.messages()

        self.assertEqual(len(messages), 2)
        self.assertEqual([p.content for p in messages[1].parts], ["fine"])


class ForkTest(SessionScanTestCase):
    def test_fork_copies_log_under_a_fresh_id(self) -> None:
        source = Session.create(self.cwd)
        source.append_system_prompt("be helpful")
        source.append_message(ModelRequest(parts=[UserPromptPart(content="hi")]))
        source.append_message(ModelResponse(parts=[TextPart(content="hello!")]))

        fork = source.fork()

        self.assertNotEqual(fork.id, source.id)
        self.assertNotEqual(fork.path, source.path)
        self.assertEqual(fork.path.parent, source.path.parent)
        header = Session._read_records(fork.path)[0]
        self.assertEqual(header["type"], "session")
        self.assertEqual(header["id"], fork.id)
        self.assertEqual(fork.system_prompt(), "be helpful")
        self.assertEqual(fork.messages(), source.messages())

    def test_fork_leaves_the_source_untouched(self) -> None:
        source = Session.create(self.cwd)
        source.append_message(ModelRequest(parts=[UserPromptPart(content="hi")]))
        before = source.path.read_text(encoding="utf-8")

        source.fork()

        self.assertEqual(source.path.read_text(encoding="utf-8"), before)

    def test_fork_keeps_line_numbers_aligned(self) -> None:
        source = Session.create(self.cwd)
        source.append_message(ModelRequest(parts=[UserPromptPart(content="hi")]))
        with source.path.open("a", encoding="utf-8") as file:
            file.write("not json\n")
        source.append_message(ModelResponse(parts=[TextPart(content="hello!")]))

        fork = source.fork()

        source_lines = source.path.read_text(encoding="utf-8").splitlines()
        fork_lines = fork.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(fork_lines[1:], source_lines[1:])

    def test_fork_preserves_compaction_checkpoints(self) -> None:
        source = Session.create(self.cwd)
        source.append_message(ModelRequest(parts=[UserPromptPart(content="hi")]))
        kept = ModelResponse(parts=[TextPart(content="hello!")])
        source.append_compaction("earlier talk", [kept], tokens_before=100)

        # The summary message is synthesized on replay with a fresh timestamp,
        # so compare part contents rather than whole messages.
        contents = [[part.content for part in message.parts] for message in source.fork().messages()]
        self.assertEqual(contents, [[part.content for part in message.parts]
                                    for message in source.messages()])
        self.assertIn("earlier talk", contents[0][0])


class LockTest(SessionScanTestCase):
    """Lock mechanics live in test_lockfile; this covers the session semantics."""

    def test_lock_raises_busy_when_another_process_holds_it(self) -> None:
        session = Session.create(self.cwd)
        with patch("paimon.session.lockfile.acquire", return_value=False):
            with self.assertRaisesRegex(SessionBusyError, "another process"):
                session.lock()

    def test_lock_and_unlock_round_trip(self) -> None:
        session = Session.create(self.cwd)
        session.lock()
        session.unlock()
        session.unlock()  # extra unlock is a no-op

    def test_a_session_already_open_here_is_refused_too(self) -> None:
        """The lock refcounts per process, so two panes would both be let in —
        and their histories would interleave into one append-only log."""
        session = Session.create(self.cwd)
        session.lock()
        self.addCleanup(session.unlock)

        with self.assertRaisesRegex(SessionBusyError, "already open"):
            Session(session.path, session.id, self.cwd).lock()


class ChildSessionTest(SessionScanTestCase):
    """Subagent sessions share the project directory but not the listings."""

    def test_children_are_hidden_unless_asked_for(self) -> None:
        mine = self._session_with_message("mine", mtime=1_000)
        child = Session.create(self.cwd, parent=mine.id)
        child.append_message(ModelRequest(parts=[UserPromptPart(content="theirs")]))

        self.assertEqual([session.id for session in Session.list(self.cwd)], [mine.id])
        listed = Session.list(self.cwd, include_children=True)
        self.assertEqual(sorted(session.id for session in listed), sorted([mine.id, child.id]))
        self.assertEqual(next(s for s in listed if s.id == child.id).parent, mine.id)

    def test_a_fork_of_a_child_is_still_a_child(self) -> None:
        parent = Session.create(self.cwd)
        child = Session.create(self.cwd, parent=parent.id)
        child.append_message(ModelRequest(parts=[UserPromptPart(content="theirs")]))

        forked = child.fork()

        self.assertEqual(forked.parent, parent.id)
        self.assertNotIn(forked.id, [session.id for session in Session.list(self.cwd)])


if __name__ == "__main__":
    unittest.main()
