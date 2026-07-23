import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paimon.session import Session, _project_dir


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
        session.append_message({"role": "user", "content": content})
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
        (_project_dir(self.cwd) / "wrong-version.jsonl").write_text(
            '{"type": "session", "version": 999, "id": "x"}\n'
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
        session.append_message({"role": "assistant", "content": "hello!"})
        session.append_message({"role": "user", "content": "fix the bug"})
        session.append_message({"role": "user", "content": "second"})

        self.assertEqual(session.first_user_text(), "fix the bug")

    def test_first_user_text_none_for_empty_session(self) -> None:
        self.assertIsNone(Session.create(self.cwd).first_user_text())


if __name__ == "__main__":
    unittest.main()
