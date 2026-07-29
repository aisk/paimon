import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from paimon.tools import MODES, _glob, _inside, _shell, gate, run_tool


class GateTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()

    def test_yolo_allows_everything(self) -> None:
        for name in ("read_file", "glob", "write_file", "edit_file", "shell", "write_todos"):
            self.assertEqual(gate(name, {"path": "/etc/hosts"}, "yolo", self.cwd), "allow")

    def test_reads_inside_cwd_are_free_outside_confirm(self) -> None:
        for mode in ("read", "edit"):
            self.assertEqual(gate("read_file", {"path": "a.py"}, mode, self.cwd), "allow")
            self.assertEqual(gate("read_file", {"path": "/etc/hosts"}, mode, self.cwd), "confirm")
            self.assertEqual(gate("read_file", {"path": "../x"}, mode, self.cwd), "confirm")
            self.assertEqual(gate("glob", {"pattern": "*.py"}, mode, self.cwd), "allow")
            self.assertEqual(gate("glob", {"pattern": "*", "path": "/tmp"}, mode, self.cwd), "confirm")

    def test_read_mode_confirms_all_dangerous_tools(self) -> None:
        self.assertEqual(gate("write_file", {"path": "a.py", "content": "x"}, "read", self.cwd), "confirm")
        self.assertEqual(gate("edit_file", {"path": "a.py"}, "read", self.cwd), "confirm")
        self.assertEqual(gate("shell", {"command": "ls"}, "read", self.cwd), "confirm")

    def test_edit_mode_auto_approves_writes_inside_cwd(self) -> None:
        self.assertEqual(gate("write_file", {"path": "a.py", "content": "x"}, "edit", self.cwd), "allow")
        self.assertEqual(gate("edit_file", {"path": "sub/a.py"}, "edit", self.cwd), "allow")
        self.assertEqual(gate("write_file", {"path": "/tmp/a.py", "content": "x"}, "edit", self.cwd), "confirm")
        self.assertEqual(gate("edit_file", {"path": "../a.py"}, "edit", self.cwd), "confirm")
        self.assertEqual(gate("shell", {"command": "ls"}, "edit", self.cwd), "confirm")

    def test_write_todos_and_missing_path_are_allowed(self) -> None:
        for mode in MODES:
            self.assertEqual(gate("write_todos", {"todos": []}, mode, self.cwd), "allow")
        self.assertEqual(gate("read_file", {}, "read", self.cwd), "allow")

    def test_start_new_session_always_confirms(self) -> None:
        for mode in MODES:
            self.assertEqual(gate("start_new_session", {"prompt": "x"}, mode, self.cwd), "confirm")


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

    async def test_shell_timeout_terminates_and_reaps_process_tree(self) -> None:
        with (
            patch("paimon.tools._COMMAND_TIMEOUT", 0.05),
            patch("paimon.tools._KILL_GRACE", 0.05),
            patch("paimon.tools._KILL_TIMEOUT", 0.5),
        ):
            result = await _shell({"command": "trap '' TERM; sleep 30"}, self.cwd)

        self.assertEqual(result, "Error: command timed out after 0.05s.")


class InsideTest(unittest.TestCase):
    def test_symlink_escape_is_outside(self) -> None:
        with tempfile.TemporaryDirectory() as outer:
            outer = Path(outer).resolve()
            cwd = outer / "project"
            cwd.mkdir()
            secret = outer / "secret.txt"
            secret.write_text("secret")
            link = cwd / "link.txt"
            link.symlink_to(secret)

            self.assertTrue(_inside(cwd / "a.py", cwd))
            self.assertFalse(_inside(secret, cwd))
            self.assertFalse(_inside(link, cwd))


class GlobSandboxTest(unittest.TestCase):
    def test_sandboxed_glob_filters_symlink_escapes(self) -> None:
        with tempfile.TemporaryDirectory() as outer:
            outer = Path(outer).resolve()
            cwd = outer / "project"
            cwd.mkdir()
            (cwd / "a.py").write_text("a")
            secret = outer / "secret.py"
            secret.write_text("secret")
            (cwd / "b.py").symlink_to(secret)

            sandboxed = _glob({"pattern": "*.py"}, cwd, sandboxed=True)
            self.assertIn("a.py", sandboxed)
            self.assertNotIn("b.py", sandboxed)

            free = _glob({"pattern": "*.py"}, cwd, sandboxed=False)
            self.assertIn("a.py", free)
            self.assertIn("b.py", free)


if __name__ == "__main__":
    unittest.main()
