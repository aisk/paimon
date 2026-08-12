import asyncio
import os
import re
import signal
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from paimon.tools import (
    MAX_OUTPUT,
    MODES,
    ToolContext,
    _glob,
    _inside,
    _shell,
    gate,
    run_tool,
    safe_command,
    shell_output_dir,
)


class GateTest(unittest.IsolatedAsyncioTestCase):
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

    def test_only_the_agents_own_overflow_files_are_read_without_confirmation(self) -> None:
        """A command's own overflow file is readable back; nothing else moves.

        The exemption is per agent: the directory is shared by every session
        and project on the machine, so a blanket one would be a free sideways
        read into another agent's command output.
        """
        with tempfile.TemporaryDirectory() as data_home:
            with patch.dict(os.environ, {"PAIMON_DATA_HOME": data_home}):
                directory = shell_output_dir()
                directory.mkdir(parents=True, exist_ok=True)
                mine = directory / "20260811-120000-1-abc.log"
                theirs = directory / "20260811-120000-2-def.log"
                mine.write_text("x")
                theirs.write_text("x")
                ctx = ToolContext(shell_outputs={mine.resolve()})

                self.assertEqual(gate("read_file", {"path": str(mine)}, "read", self.cwd, ctx=ctx), "allow")
                self.assertEqual(gate("read_file", {"path": str(theirs)}, "read", self.cwd, ctx=ctx), "confirm")
                self.assertEqual(gate("read_file", {"path": str(mine)}, "read", self.cwd), "confirm")
                self.assertEqual(gate("read_file", {"path": "/etc/hosts"}, "read", self.cwd, ctx=ctx), "confirm")
                self.assertEqual(gate("write_file", {"path": str(mine)}, "edit", self.cwd, ctx=ctx), "confirm")

    async def test_a_command_records_the_overflow_file_it_wrote(self) -> None:
        """The gate exemption above is only reachable through this."""
        with tempfile.TemporaryDirectory() as data_home:
            with patch.dict(os.environ, {"PAIMON_DATA_HOME": data_home}):
                ctx = ToolContext()
                result = await _shell({"command": _filler(3000, "line")}, self.cwd, ctx)
                self.assertEqual(ctx.shell_outputs, {_overflow_path(result).resolve()})
                self.assertEqual(gate("read_file", {"path": str(_overflow_path(result))},
                                      "read", self.cwd, ctx=ctx), "allow")

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
        with patch.dict(os.environ, {"CDPATH": ""}):
            self.assertEqual(gate("shell", {"command": "cd sub && ls"}, "read", self.cwd), "allow")
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

        self.assertIn("(timed out after 0.05s)", result)


def _filler(lines: int, payload: str, newline: bool = True) -> str:
    """A POSIX sh loop printing ``payload`` ``lines`` times (/bin/sh, not bash)."""
    fmt = "%s\\n" if newline else "%s"
    return f'i=0; while [ $i -lt {lines} ]; do printf "{fmt}" "{payload}"; i=$((i+1)); done'


def _overflow_path(result: str) -> Path:
    match = re.search(r"full output: (\S+?)\]", result)
    assert match, f"no overflow path in result: {result[-300:]!r}"
    return Path(match.group(1))


class ShellOutputTest(unittest.IsolatedAsyncioTestCase):
    """The model has to see how a command ended, which is what the tail holds."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()
        data = tempfile.TemporaryDirectory()
        self.addCleanup(data.cleanup)
        env = patch.dict(os.environ, {"PAIMON_DATA_HOME": data.name})
        env.start()
        self.addCleanup(env.stop)

    def _reap(self, pid: int) -> None:
        """Kill a process this test started, whatever the assertions did."""
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    @staticmethod
    def _pid_from(result: str) -> int:
        match = re.search(r"pid (\d+)", result)
        assert match, f"no pid in result: {result!r}"
        return int(match.group(1))

    async def test_tail_survives_a_large_output_and_the_head_is_kept_on_disk(self) -> None:
        # 300 * 101 bytes overshoots the byte budget while staying well under
        # the line budget, so this is byte truncation alone.
        command = f'{_filler(300, "x" * 100)}; echo FINAL-ERROR; exit 7'
        result = await _shell({"command": command}, self.cwd)

        self.assertIn("FINAL-ERROR", result, "the tail is what the model needs")
        self.assertIn("(exit code 7)", result)
        self.assertIn("showing last", result)
        self.assertLessEqual(len(result), MAX_OUTPUT, "execute_tool must never re-cut this")

        path = _overflow_path(result)
        self.assertEqual(path.parent, shell_output_dir())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        full = path.read_text()
        self.assertTrue(full.startswith("x" * 100), "the head is recoverable")
        self.assertIn("FINAL-ERROR", full)

    async def test_line_budget_truncates_before_the_byte_budget(self) -> None:
        command = _filler(3000, "line")  # ~15KB, far under the byte cap
        result = await _shell({"command": command}, self.cwd)

        body = result.split("\n\n[")[0]
        self.assertEqual(len(body.splitlines()), 2_000)
        self.assertIn("of 3,000 lines", result)
        self.assertIn("full output:", result)
        self.assertIn("(exit code 0)", result)

    async def test_multibyte_output_stays_within_the_result_budget(self) -> None:
        command = f'{_filler(400, "😀" * 40)}; echo TAIL-OK'
        result = await _shell({"command": command}, self.cwd)

        self.assertIn("TAIL-OK", result, "the cut must not eat the last line")
        self.assertLessEqual(len(result), MAX_OUTPUT)

    async def test_one_enormous_line_keeps_its_end(self) -> None:
        """No newline to cut on, so the end of the line is the whole answer."""
        result = await _shell({"command": _filler(400, "x" * 100, newline=False)}, self.cwd)

        self.assertIn("of line 1 (line is ", result)
        self.assertLessEqual(len(result), MAX_OUTPUT)
        self.assertIn("full output:", result)
        self.assertEqual(len(_overflow_path(result).read_bytes()), 40_000)

    async def test_timeout_keeps_what_the_command_already_printed(self) -> None:
        command = "printf 'partial-output\\n'; trap '' TERM; sleep 30"
        with (
            patch("paimon.tools._COMMAND_TIMEOUT", 0.3),
            patch("paimon.tools._KILL_GRACE", 0.05),
            patch("paimon.tools._KILL_TIMEOUT", 0.5),
        ):
            result = await _shell({"command": command}, self.cwd)

        self.assertIn("partial-output", result, "a timeout is when output matters most")
        self.assertIn("(timed out after 0.3s)", result)

    async def test_a_backgrounded_descendant_does_not_hold_the_turn(self) -> None:
        """The command is finished even though something it started holds the pipe."""
        with patch("paimon.tools._COMMAND_TIMEOUT", 10.0):
            started = time.monotonic()
            result = await _shell({"command": "sleep 10 & echo pid $!"}, self.cwd)
            elapsed = time.monotonic() - started
        self.addCleanup(self._reap, self._pid_from(result))

        self.assertIn("(exit code 0)", result)
        self.assertLess(elapsed, 3.0, "waiting for stdout EOF would have taken the full timeout")

    async def test_timeout_kills_the_descendants_too(self) -> None:
        command = "sleep 30 & echo pid $!; trap '' TERM; sleep 30"
        with (
            patch("paimon.tools._COMMAND_TIMEOUT", 0.3),
            patch("paimon.tools._KILL_GRACE", 0.05),
            patch("paimon.tools._KILL_TIMEOUT", 0.5),
        ):
            result = await _shell({"command": command}, self.cwd)
        pid = self._pid_from(result)
        self.addCleanup(self._reap, pid)

        for _ in range(40):
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, PermissionError):
                return
            await asyncio.sleep(0.05)
        self.fail(f"backgrounded descendant {pid} outlived the timeout")


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

    def test_hard_metacharacters_reject(self) -> None:
        # Substitution, redirection, escaping and expansion are rejected even
        # inside quotes; only ";", "&" and "|" get position-aware handling.
        for cmd in (
            "cat $(pwd)/x", "echo `id`", "ls > out.txt", "cat < /etc/passwd",
            "(ls)", "ls \\\n x", "cat {..,x}/y", "ls [.][.]",
            'echo "$HOME"', "echo \"`id`\"",
        ):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_compound_commands_allowed(self) -> None:
        for cmd in ("ls | wc -l", "git log --oneline | head -5",
                    "grep -rn foo . | wc -l", "ls && pwd",
                    "git log; git status", "ls || pwd", "echo a&&pwd"):
            self.assertTrue(safe_command(cmd, self.cwd), cmd)

    def test_compound_unsafe_segment_rejects(self) -> None:
        for cmd in ("ls; rm x", "ls | rm x", "ls && rm -rf .",
                    "git status; git push", "cat f | sed -i s/a/b/ f",
                    'echo "&&" && rm x'):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_quoted_operators_allowed(self) -> None:
        for cmd in ('grep "a|b" f', "grep -E 'foo|bar' .", "echo 'a && b'",
                    "grep ';' f"):
            self.assertTrue(safe_command(cmd, self.cwd), cmd)

    def test_operator_edge_cases_reject(self) -> None:
        for cmd in ("ls &&", "&& ls", "| wc", "ls |", "ls ;; pwd",
                    "ls && && pwd", "ls & pwd", "ls & rm x", "ls &",
                    "ls |& pwd", "ls 'x && pwd"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_cd_chains_allowed(self) -> None:
        with patch.dict(os.environ, {"CDPATH": ""}):
            for cmd in ("cd sub && ls", "cd sub && cat a.txt",
                        "cd sub && cd deeper && ls", "cd sub && ls ../",
                        "cd 'sub' && ls", f"cd {self.cwd}/sub && ls", "cd sub"):
                self.assertTrue(safe_command(cmd, self.cwd), cmd)

    def test_cd_forms_reject(self) -> None:
        for cmd in ("cd", "cd -", "cd -P sub", "cd a b", "cd ..", "cd ../x",
                    "cd a/../b && ls", "cd .* && ls", "cd sub* && ls",
                    "cd s?b && ls"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_cd_glob_target_cannot_escape_via_symlink(self) -> None:
        # A glob's literal form hides the symlink it matches, so the target's
        # containment check would otherwise pass while the shell cd's out.
        (self.cwd / "sublink").symlink_to("/etc")
        self.assertFalse(safe_command("cd sublink* && cat passwd", self.cwd))

    def test_cd_requires_pure_and_chain(self) -> None:
        # With ";" or "||" a failed or skipped cd leaves later segments in a
        # different directory than modeled; in a pipeline cd runs in its own
        # subshell and applies to nothing.
        for cmd in ("cd sub; ls", "cd sub | ls", "ls | cd sub",
                    "cd sub || ls", "ls; cd sub && cat f",
                    "ls || cd sub && cat ../f"):
            self.assertFalse(safe_command(cmd, self.cwd), cmd)

    def test_cd_boundary_is_original_cwd(self) -> None:
        self.assertFalse(safe_command("cd sub && cat ../../f", self.cwd))
        (self.cwd / "link").symlink_to("/etc")
        self.assertFalse(safe_command("cd link && ls", self.cwd))

    def test_cdpath_rejects_cd_only(self) -> None:
        with patch.dict(os.environ, {"CDPATH": "/tmp"}):
            self.assertFalse(safe_command("cd sub && ls", self.cwd))
            self.assertTrue(safe_command("ls | wc -l", self.cwd))

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

    def test_symlink_loop_confirms_instead_of_raising(self) -> None:
        """resolve() raises RuntimeError on a loop, outside any tool-call error
        boundary: it has to come back as a permission decision, not end the turn."""
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory).resolve()
            loop = cwd / "loop"
            loop.symlink_to(loop)

            self.assertFalse(_inside(loop, cwd))
            self.assertEqual(gate("read_file", {"path": "loop"}, "read", cwd), "confirm")


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
