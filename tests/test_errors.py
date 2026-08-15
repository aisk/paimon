"""Every deliberate failure hangs off one root, and none off RuntimeError."""

import unittest

from paimon import PaimonError
from paimon.aside import AsideError
from paimon.compaction import CompactionError
from paimon.llm import NoModelError
from paimon.session import SessionBusyError, SessionError, SessionIncompleteError
from paimon.supervisor import SupervisorError

ERRORS = [AsideError, CompactionError, NoModelError, SessionError,
          SessionBusyError, SessionIncompleteError, SupervisorError]


class ErrorTreeTest(unittest.TestCase):
    def test_everything_paimon_raises_is_a_paimon_error(self) -> None:
        for error in ERRORS:
            self.assertTrue(issubclass(error, PaimonError), error.__name__)

    def test_nothing_borrows_a_builtin_meaning(self) -> None:
        # RuntimeError means the interpreter hit a runtime problem, and code
        # catching it around Path.resolve or a signal handler must not swallow
        # a session that is merely busy.
        for error in ERRORS:
            self.assertNotIsInstance(error("x"), (RuntimeError, ValueError, OSError), error.__name__)

    def test_the_session_family_stays_catchable_as_one(self) -> None:
        for error in (SessionBusyError, SessionIncompleteError):
            self.assertTrue(issubclass(error, SessionError), error.__name__)


if __name__ == "__main__":
    unittest.main()
