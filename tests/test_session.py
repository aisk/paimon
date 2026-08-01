import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

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


if __name__ == "__main__":
    unittest.main()
