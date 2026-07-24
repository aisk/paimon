import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from paimon import lockfile


class LockfileTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "target.jsonl"
        self.path.touch()

    def test_same_process_reacquire_is_refcounted(self) -> None:
        self.assertTrue(lockfile.acquire(self.path))
        self.assertTrue(lockfile.acquire(self.path))

        lockfile.release(self.path)
        self.assertTrue(lockfile.held(self.path), "still claimed by the second holder")
        lockfile.release(self.path)
        self.assertFalse(lockfile.held(self.path))

    def test_release_without_acquire_is_a_noop(self) -> None:
        lockfile.release(self.path)

    def _probe(self) -> int:
        """Try to acquire self.path from a separate process; its exit code
        is 0 when the lock was free, 3 when it was held."""
        script = textwrap.dedent("""
            import sys
            from pathlib import Path
            from paimon import lockfile
            sys.exit(0 if lockfile.acquire(Path(sys.argv[1])) else 3)
        """)
        result = subprocess.run([sys.executable, "-c", script, str(self.path)],
                                capture_output=True, text=True)
        self.assertIn(result.returncode, (0, 3), result.stderr)
        return result.returncode

    def test_other_process_is_locked_out_until_release(self) -> None:
        self.assertTrue(lockfile.acquire(self.path))
        self.addCleanup(lockfile.release, self.path)
        self.assertEqual(self._probe(), 3)

        lockfile.release(self.path)
        self.assertEqual(self._probe(), 0)
        lockfile.acquire(self.path)  # keep the addCleanup release balanced

    def test_other_fds_on_the_file_do_not_release_the_lock(self) -> None:
        # Appends open and close their own fds; flock must survive that
        # (POSIX record locks would be dropped by the first close).
        self.assertTrue(lockfile.acquire(self.path))
        self.addCleanup(lockfile.release, self.path)

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("appended\n")

        self.assertEqual(self._probe(), 3)


if __name__ == "__main__":
    unittest.main()
