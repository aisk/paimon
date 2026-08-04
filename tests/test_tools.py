import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from paimon.tools import MODES, _glob, _inside, _shell, gate, run_tool, safe_command


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
        self.assertEqual(gate("shell", {"command": "rm -rf x"}, "read", self.cwd), "confirm")

    def test_edit_mode_auto_approves_writes_inside_cwd(self) -> None:
        self.assertEqual(gate("write_file", {"path": "a.py", "content": "x"}, "edit", self.cwd), "allow")
        self.assertEqual(gate("edit_file", {"path": "sub/a.py"}, "edit", self.cwd), "allow")
        self.assertEqual(gate("write_file", {"path": "/tmp/a.py", "content": "x"}, "edit", self.cwd), "confirm")
        self.assertEqual(gate("edit_file", {"path": "../a.py"}, "edit", self.cwd), "confirm")
        self.assertEqual(gate("shell", {"command": "rm -rf x"}, "edit", self.cwd), "confirm")

    def test_write_todos_and_missing_path_are_allowed(self) -> None:
        for mode in MODES:
            self.assertEqual(gate("write_todos", {"todos": []}, mode, self.cwd), "allow")
        self.assertEqual(gate("read_file", {}, "read", self.cwd), "allow")

    def test_start_new_session_always_confirms(self) -> None:
        for mode in MODES:
            self.assertEqual(gate("start_new_session", {"prompt": "x"}, mode, self.cwd), "confirm")

    def test_safe_shell_commands_auto_allowed(self) -> None:
        for mode in ("read", "edit"):
            self.assertEqual(gate("shell", {"command": "ls"}, mode, self.cwd), "allow")
            self.assertEqual(gate("shell", {"command": "git status"}, mode, self.cwd), "allow")
        self.assertEqual(gate("shell", {}, "read", self.cwd), "confirm")

    def test_strict_disables_safe_commands(self) -> None:
        for mode in ("read", "edit"):
            self.assertEqual(gate("shell", {"command": "ls"}, mode, self.cwd, safe_commands=False), "confirm")
        self.assertEqual(gate("shell", {"command": "ls"}, "yolo", self.cwd, safe_commands=False), "allow")


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

        self.assertEqual(result, "Error: command timed out after 0.05s.")


class SafeCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()

    def test_recognized_read_only_commands(self) -> None:
        for cmd in (
            "ls", "ls -la", "pwd", "cat sub/a.txt", "head -n 20 a.py",
            "tail -n 5 a.log", "wc -l *.py", "stat a.py", "file a.py",
            "which python3", "df", "du -sh .", "grep TODO src", "grep -rn foo .",
            "rg TODO", "rg -n foo src", "find . -name *.py -type f",
            "tree -L 2", "diff a.txt b.txt", "readlink -f a.py",
            "realpath a.py", "echo done", "uname -a", "whoami", "date", "nproc",
            "git status", "git status --porcelain", "git log --oneline -20",
            "git log HEAD~3..", "git diff --stat", "git show HEAD~1",
            "git branch -a", "git branch --show-current",
            "git blame cli.py", "git shortlog -sn", "git describe --tags",
            "git rev-parse HEAD", "git ls-files", "git ls-tree -r HEAD",
            "git remote -v", "git tag -l",
        ):
            self.assertTrue(safe_command(cmd, self.cwd), cmd)

    def test_globs_that_could_expand_to_parent_reject(self) -> None:
        # ".*" expands to "." and "..", so the literal path check is not enough.
        for cmd in ("ls .*", "grep -r SECRET .*", "cat .*/*", "grep -r x .*/.*",
                    "ls ..*", "ls .?", "wc -l sub/.*"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_attached_flag_values_reject(self) -> None:
        # A path attached to a flag never reaches the containment check.
        for cmd in ("diff --from-file=/etc/passwd x", "diff --to-file=/etc/passwd x",
                    "grep -f/etc/passwd -r .", "grep --file=/etc/passwd x",
                    "file -m/etc/passwd .", "du --exclude-from=/etc/passwd .",
                    "date -f/etc/passwd", "date -f /etc/passwd"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_shell_metacharacters_reject(self) -> None:
        for cmd in (
            "ls; rm x", "ls | wc -l", "cat $(pwd)/x", "echo `id`",
            "ls > out.txt", "ls & rm x", "cat < /etc/passwd", "(ls)",
            "ls \\\n x", "cat {..,x}/y", "ls [.][.]",
        ):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_tilde_and_parse_failures_reject(self) -> None:
        for cmd in ("cat ~/secrets", "grep --include=~/x foo", "ls 'unclosed", ""):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_unlisted_commands_reject(self) -> None:
        for cmd in ("rm x", "sed -i s/a/b/ f", "sort -o out in", "env",
                    "./ls", "/bin/ls", "FOO=1 ls"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_denied_flags_reject(self) -> None:
        for cmd in ("tail -f log", "tail -20f log", "tail --follow log",
                    "file -C -m magic src", "rg --pre sh foo",
                    "rg --hostname-bin=x foo", "find . -delete",
                    "find . -name x -exec rm {} +", "find . -fprintf out %p",
                    "tree -o out", "date -s 12:00"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_paths_outside_cwd_reject(self) -> None:
        for cmd in ("cat /etc/passwd", "cat ../x", "grep foo /etc/passwd", "du -sh /"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_symlink_escape_rejects(self) -> None:
        (self.cwd / "link").symlink_to("/etc")
        self.assertFalse(safe_command("cat link/passwd", self.cwd))

    def test_unsafe_git_rejects(self) -> None:
        for cmd in (
            "git", "git push", "git commit -m x", "git -c core.pager=sh log",
            "git -C /tmp status", "git log --output=/tmp/x",
            "git log --output-directory=/tmp", "git diff --no-index a b",
            "git diff --ext-diff", "git show --textconv HEAD:f",
            "git branch foo", "git branch -D foo", "git branch --set-upstream-to=x",
            "git remote add origin url", "git remote prune origin",
            "git tag v1", "git tag -d v1", "git stash", "git reflog expire --all",
            # git grep -O runs a command, and bundling (-nOsh) hides it.
            "git grep -n foo", "git grep -nOsh foo", "git grep -Ovim foo",
            # Options that read a file the cwd check never sees.
            "git blame --contents /etc/passwd -- x", "git blame --contents=/etc/passwd x",
            "git ls-files -X /etc/passwd", "git ls-files -cX /etc/passwd",
            "git ls-files --exclude-from=/etc/passwd",
            "git log --show-signature", "git show --textconv HEAD:f",
        ):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)


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
