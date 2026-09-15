import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from paimon.tools import (
    _grep,
    _grep_async,
    run_tool,
)


class GrepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cwd = Path(self.tmp.name).resolve()

    def write(self, relative: str, content: str) -> Path:
        path = self.cwd / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_matches_come_back_as_path_line_text(self) -> None:
        self.write("a.py", "x = 1\nneedle here\n")
        self.write("sub/b.py", "also a needle\n")
        result = _grep({"pattern": "needle"}, self.cwd)
        lines = result.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn(f"{self.cwd / 'a.py'}:2:needle here", lines)
        self.assertIn(f"{self.cwd / 'sub' / 'b.py'}:1:also a needle", lines)

    def test_a_single_file_can_be_searched(self) -> None:
        path = self.write("a.txt", "one\ntwo\n")
        result = _grep({"pattern": "two", "path": str(path)}, self.cwd)
        self.assertEqual(result, f"{path}:2:two")

    def test_glob_filters_by_file_name(self) -> None:
        self.write("a.py", "needle\n")
        self.write("a.md", "needle\n")
        result = _grep({"pattern": "needle", "glob": "*.py"}, self.cwd)
        self.assertIn("a.py", result)
        self.assertNotIn("a.md", result)

    def test_noise_directories_and_binary_files_are_skipped(self) -> None:
        self.write("node_modules/dep.js", "needle\n")
        (self.cwd / "blob.bin").write_bytes(b"needle\0needle")
        self.write("a.py", "needle\n")
        result = _grep({"pattern": "needle"}, self.cwd)
        self.assertEqual(len(result.splitlines()), 1)
        self.assertIn("a.py", result)

    def test_the_result_is_capped_with_a_note(self) -> None:
        self.write("a.txt", "needle\n" * 10)
        result = _grep({"pattern": "needle", "max_results": 3}, self.cwd)
        lines = result.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("first 3 matches shown", lines[-1])

    def test_a_bad_regex_is_an_error_not_an_exception(self) -> None:
        self.assertIn("invalid regular expression", _grep({"pattern": "("}, self.cwd))

    def test_a_missing_base_is_an_error(self) -> None:
        result = _grep({"pattern": "x", "path": "nowhere"}, self.cwd)
        self.assertIn("no such file or directory", result)

    def test_no_matches_says_so(self) -> None:
        self.write("a.txt", "quiet\n")
        self.assertEqual(_grep({"pattern": "needle"}, self.cwd), "(no matches)")

    def test_sandboxed_grep_skips_symlink_escapes(self) -> None:
        with tempfile.TemporaryDirectory() as outer:
            outer = Path(outer).resolve()
            cwd = outer / "project"
            cwd.mkdir()
            (cwd / "a.py").write_text("needle\n")
            secret = outer / "secret.py"
            secret.write_text("needle\n")
            (cwd / "b.py").symlink_to(secret)

            sandboxed = _grep({"pattern": "needle"}, cwd, sandboxed=True)
            self.assertNotIn("b.py", sandboxed)
            free = _grep({"pattern": "needle"}, cwd, sandboxed=False)
            self.assertIn("b.py", free)

    def test_exact_limit_is_not_truncated(self) -> None:
        path = self.write("a.txt", "needle\n" * 3)
        result = _grep({"pattern": "needle", "max_results": 3}, self.cwd)
        self.assertEqual(result.splitlines(), [f"{path}:{i}:needle" for i in range(1, 4)])

    def test_truncation_detects_match_in_next_file(self) -> None:
        self.write("a.txt", "needle\n")
        self.write("b.txt", "needle\n")
        result = _grep({"pattern": "needle", "max_results": 1}, self.cwd)
        self.assertIn("first 1 matches shown", result)
        self.assertNotIn("b.txt", result)

    def test_only_lf_and_crlf_change_line_numbers(self) -> None:
        path = self.cwd / "a.txt"
        path.write_bytes(b"\xef\xbb\xbfhead\xe2\x80\xa8needle\r\nnext\vneedle\nlast needle")
        result = _grep({"pattern": "needle"}, self.cwd)
        self.assertEqual(result, f"{path}:1:head\u2028needle\n{path}:2:next\vneedle\n{path}:3:last needle")
        self.assertEqual(_grep({"pattern": "^head"}, self.cwd), f"{path}:1:head\u2028needle")

    def test_ignored_directories_are_not_entered(self) -> None:
        self.write("node_modules/nested/a.txt", "needle\n")
        self.write("sub/.git/a.txt", "needle\n")
        self.write("sub/b.txt", "needle\n")
        visited = []
        scandir = os.scandir

        def track(path):
            visited.append(Path(path))
            return scandir(path)

        with patch("os.scandir", side_effect=track):
            result = _grep({"pattern": "needle"}, self.cwd)
        self.assertIn("b.txt", result)
        self.assertEqual(visited, [self.cwd, self.cwd / "sub"])

    def test_stops_reading_after_one_extra_match(self) -> None:
        path = self.write("a.txt", "needle\n" * 10000)
        with path.open("rb") as source:
            tracked = MagicMock(wraps=source)
            tracked.__enter__.return_value = tracked
            with patch.object(Path, "open", return_value=tracked):
                result = _grep({"pattern": "needle", "max_results": 1}, self.cwd)
            self.assertEqual(tracked.readline.call_count, 2)
            tracked.read.assert_called_once_with(8192)
        self.assertIn("first 1 matches shown", result)

    def test_large_files_are_reported(self) -> None:
        self.write("a.txt", "needle\n" * 10)
        with patch("paimon.tools._GREP_MAX_FILE_BYTES", 20):
            self.assertEqual(_grep({"pattern": "needle"}, self.cwd), "(no matches) (1 large files skipped)")

    def test_directory_symlink_cycles_are_not_followed(self) -> None:
        self.write("sub/a.txt", "needle\n")
        (self.cwd / "sub/back").symlink_to(self.cwd, target_is_directory=True)
        self.assertEqual(len(_grep({"pattern": "needle"}, self.cwd).splitlines()), 1)


class GrepProcessTest(unittest.IsolatedAsyncioTestCase):
    async def test_registered_tool_searches_in_requested_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            path = cwd / "a.txt"
            path.write_text("needle\n")
            result, denied = await run_tool("grep", {"pattern": "needle"}, cwd, mode="read")
            self.assertFalse(denied)
            self.assertEqual(result, f"{path}:1:needle")

    async def test_pathological_regex_times_out_without_blocking_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "a.txt").write_text("a" * 100 + "!\n")
            processes = []
            spawn = asyncio.create_subprocess_exec

            async def track(*args, **kwargs):
                proc = await spawn(*args, **kwargs)
                processes.append(proc)
                return proc

            with patch("paimon.tools._GREP_TIMEOUT", 1.0), patch(
                "paimon.tools.asyncio.create_subprocess_exec", side_effect=track,
            ):
                task = asyncio.create_task(_grep_async({"pattern": "(a+)+$"}, cwd))
                await asyncio.sleep(0.1)
                self.assertFalse(task.done())
                result = await asyncio.wait_for(task, 5)
            self.assertIn("timed out", result)
            self.assertIsNotNone(processes[0].returncode)

    async def test_cancellation_reaps_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "a.txt").write_text("a" * 100 + "!\n")
            processes = []
            started = asyncio.Event()
            spawn = asyncio.create_subprocess_exec

            async def track(*args, **kwargs):
                proc = await spawn(*args, **kwargs)
                processes.append(proc)
                started.set()
                return proc

            with patch("paimon.tools.asyncio.create_subprocess_exec", side_effect=track):
                task = asyncio.create_task(_grep_async({"pattern": "(a+)+$"}, cwd))
                await asyncio.wait_for(started.wait(), 5)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertIsNotNone(processes[0].returncode)
