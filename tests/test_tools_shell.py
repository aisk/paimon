import asyncio
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from paimon.tools.background import BackgroundCommand, _TaskOutput, start_background, tail_text
from paimon.tools.base import MAX_OUTPUT
from paimon.tools.shell import (
    _SHELL_MAX_BYTES,
    _kill_tree,
    _OutputTail,
    _shell,
    _signal_group,
    decode_output,
    find_git_bash,
    git_bash_root,
    shell_executable,
    shell_output_dir,
)
from tests.support.shell import (
    alive,
    filler,
    git_bash,
    overflow_path,
    pid_from,
    posix_shell,
    printer,
    reap,
    spawner,
)


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

    async def test_tail_survives_a_large_output_and_the_head_is_kept_on_disk(self) -> None:
        # 300 * 101 bytes overshoots the byte budget while staying well under
        # the line budget, so this is byte truncation alone.
        command = filler(300, "x" * 100, last="FINAL-ERROR", exit_code=7)
        outcome = await _shell({"command": command}, self.cwd)
        result = outcome.text

        self.assertEqual((outcome.status, outcome.detail), ("failed", "exit 7"))
        self.assertIn("FINAL-ERROR", result, "the tail is what the model needs")
        self.assertIn("(exit code 7)", result)
        self.assertIn("showing last", result)
        self.assertLessEqual(len(result), MAX_OUTPUT, "execute_tool must never re-cut this")

        path = overflow_path(result)
        self.assertEqual(path.parent, shell_output_dir())
        if os.name == "posix":  # Windows has no permission bits to check
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        full = path.read_text()
        self.assertTrue(full.startswith("x" * 100), "the head is recoverable")
        self.assertIn("FINAL-ERROR", full)

    async def test_line_budget_truncates_before_the_byte_budget(self) -> None:
        command = filler(3000, "line")  # ~15KB, far under the byte cap
        outcome = await _shell({"command": command}, self.cwd)
        result = outcome.text

        self.assertEqual((outcome.status, outcome.detail), ("success", ""))
        body = result.split("\n\n[")[0]
        self.assertEqual(len(body.splitlines()), 2_000)
        self.assertIn("of 3,000 lines", result)
        self.assertIn("full output:", result)
        self.assertIn("(exit code 0)", result)

    async def test_multibyte_output_stays_within_the_result_budget(self) -> None:
        command = filler(400, "😀" * 40, last="TAIL-OK")
        result = (await _shell({"command": command}, self.cwd)).text

        self.assertIn("TAIL-OK", result, "the cut must not eat the last line")
        self.assertLessEqual(len(result), MAX_OUTPUT)

    async def test_one_enormous_line_keeps_its_end(self) -> None:
        """No newline to cut on, so the end of the line is the whole answer."""
        result = (await _shell({"command": filler(400, "x" * 100, newline=False)}, self.cwd)).text

        self.assertIn("of line 1 (line is ", result)
        self.assertLessEqual(len(result), MAX_OUTPUT)
        self.assertIn("full output:", result)
        self.assertEqual(len(overflow_path(result).read_bytes()), 40_000)

    async def test_timeout_keeps_what_the_command_already_printed(self) -> None:
        command = printer("partial-output", then_sleep=True, ignore_term=True)
        with (
            patch("paimon.tools.shell._COMMAND_TIMEOUT", 0.3),
            patch("paimon.tools.shell._KILL_GRACE", 0.05),
            patch("paimon.tools.shell._KILL_TIMEOUT", 0.5),
        ):
            outcome = await _shell({"command": command}, self.cwd)
        result = outcome.text

        self.assertEqual((outcome.status, outcome.detail), ("failed", "timed out"))
        self.assertIn("partial-output", result, "a timeout is when output matters most")
        self.assertIn("(timed out after 0.3s)", result)

    async def test_a_backgrounded_descendant_does_not_hold_the_turn(self) -> None:
        """The command is finished even though something it started holds the pipe."""
        with patch("paimon.tools.shell._COMMAND_TIMEOUT", 10.0):
            started = time.monotonic()
            result = (await _shell({"command": spawner(then_sleep=False)}, self.cwd)).text
            elapsed = time.monotonic() - started
        self.addCleanup(reap, pid_from(result))

        self.assertIn("(exit code 0)", result)
        self.assertLess(elapsed, 3.0, "waiting for stdout EOF would have taken the full timeout")

    async def test_timeout_kills_the_descendants_too(self) -> None:
        command = spawner(then_sleep=True, ignore_term=True)
        with (
            # Long enough for the command to start its child and say so: an
            # interpreter takes far longer to come up than a shell builtin.
            patch("paimon.tools.shell._COMMAND_TIMEOUT", 3.0),
            patch("paimon.tools.shell._KILL_GRACE", 0.05),
            patch("paimon.tools.shell._KILL_TIMEOUT", 0.5),
        ):
            result = (await _shell({"command": command}, self.cwd)).text
        pid = pid_from(result)
        self.addCleanup(reap, pid)

        for _ in range(40):
            if not alive(pid):
                return
            await asyncio.sleep(0.05)
        self.fail(f"backgrounded descendant {pid} outlived the timeout")


class BackgroundCommandTest(unittest.IsolatedAsyncioTestCase):
    """A command that outlives the turn: it streams, it ends, it can be stopped."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()

    async def _start(self, command: str):
        running = await start_background(command, self.cwd)
        self.addAsyncCleanup(self._stop, running)
        return running

    @staticmethod
    async def _stop(running) -> None:
        """End the command before its working directory is cleaned up.

        terminate_now only fires the kill on Windows, and a directory that is
        still some process's working directory cannot be removed there, so the
        tree kill is awaited instead.
        """
        if os.name == "nt":
            await _kill_tree(running._proc, running._pgid)
        else:
            running.terminate_now()

    async def _wait(self, predicate, message: str) -> None:
        for _ in range(200):
            if predicate():
                return
            await asyncio.sleep(0.05)
        self.fail(message)

    async def test_output_arrives_before_the_command_ends(self) -> None:
        running = await self._start(printer("first", then_sleep=True))
        await self._wait(lambda: running.output.total_bytes, "nothing was read from the pipe")

        data, cursor, dropped = running.output.since(0)
        self.assertEqual(data, b"first\n")
        self.assertEqual((cursor, dropped), (6, 0))
        self.assertTrue(running.running, "the command has not exited")
        self.assertEqual(running.output.since(cursor)[0], b"",
                         "a second read sees only what is new")

    async def test_it_ends_with_an_exit_code(self) -> None:
        running = await self._start(printer("done", exit_code=3))
        await self._wait(lambda: not running.running, "the command never finished")

        self.assertEqual(running.exit_code, 3)
        self.assertFalse(running.killed)
        self.assertEqual(running.output.since(0)[0], b"done\n")

    async def test_kill_stops_the_whole_group(self) -> None:
        running = await self._start(spawner(then_sleep=True, ignore_term=True))
        await self._wait(lambda: running.output.total_bytes, "no pid was printed")
        pid = pid_from(running.output.since(0)[0].decode())
        self.addCleanup(reap, pid)

        with patch("paimon.tools.shell._KILL_GRACE", 0.05), patch("paimon.tools.shell._KILL_TIMEOUT", 0.5):
            running.kill()
            await self._wait(lambda: not running.running, "the command survived being killed")
            await self._wait(lambda: not alive(pid),
                             f"backgrounded descendant {pid} outlived the kill")
        self.assertTrue(running.killed)

    async def test_a_reader_that_fell_behind_is_told_what_it_missed(self) -> None:
        output = _TaskOutput(limit=16)
        output.append(b"0123456789")
        output.append(b"abcdefghij")

        data, cursor, dropped = output.since(0)
        self.assertEqual(dropped, 4, "the oldest bytes are gone, not silently skipped")
        self.assertEqual(data, b"456789abcdefghij")
        self.assertEqual(cursor, 20)

    async def test_the_model_only_ever_gets_the_tail(self) -> None:
        text = tail_text(b"x" * 100 + b"end", limit=10)
        self.assertTrue(text.endswith("xxxxxxxend"))
        self.assertIn("dropped", text.splitlines()[0])


class OutputDecodingTest(unittest.TestCase):
    """A cut lands inside a character more often than not; neither the cut
    nor one stray byte may turn the rest of the output into garbage."""

    LINE = ("中文输出" * 10 + "\n").encode("utf-8")

    def test_a_tail_cut_inside_a_character_decodes_cleanly(self) -> None:
        lines = self.LINE * (_SHELL_MAX_BYTES // len(self.LINE) + 50)
        # A last line just long enough to put the cut inside a character.
        data = next(lines + b"a" * pad + b"\n" for pad in range(3)
                    if 0x80 <= (lines + b"a" * pad + b"\n")[-_SHELL_MAX_BYTES] <= 0xBF)
        tail = _OutputTail()
        try:
            tail.append(data)
            text = tail.render()
        finally:
            tail.close()
            if tail.path is not None:
                tail.path.unlink()
        body = text.rpartition("\n\n[")[0]
        self.assertNotIn("�", body)
        self.assertEqual(set(body.split("\n")[:-1]), {"中文输出" * 10})

    def test_one_huge_line_cut_inside_a_character_decodes_cleanly(self) -> None:
        line = "中文".encode("utf-8") * _SHELL_MAX_BYTES
        data = next(line + b"a" * pad for pad in range(3)
                    if 0x80 <= (line + b"a" * pad)[-_SHELL_MAX_BYTES] <= 0xBF)
        tail = _OutputTail()
        try:
            tail.append(data)
            text = tail.render()
        finally:
            tail.close()
            if tail.path is not None:
                tail.path.unlink()
        self.assertTrue(text.startswith("中文") or text.startswith("文"), text[:20])
        self.assertNotIn("�", text)

    def test_the_models_tail_of_a_background_command_decodes_cleanly(self) -> None:
        data = b"x" + self.LINE * 1000
        text = tail_text(data, limit=10_000)
        first, _, rest = text.partition("\n")
        self.assertIn("dropped", first)
        self.assertNotIn("�", rest)
        self.assertTrue(rest.endswith("中文输出\n"))
        # Read where an earlier read stopped mid-character.
        self.assertNotIn("�", tail_text(self.LINE[1:], dropped=1))

    @unittest.skipUnless(os.name == "nt", "the OEM code page is Windows'")
    def test_utf8_with_a_stray_or_cut_byte_is_still_utf8(self) -> None:
        body = self.LINE * 20
        self.assertEqual(decode_output(body + "中".encode("utf-8")[:2]), self.LINE.decode() * 20 + "�")
        self.assertEqual(decode_output(body + b"\xff" + body).count("�"), 1)

    @unittest.skipUnless(os.name == "nt", "the OEM code page is Windows'")
    def test_output_in_the_oem_code_page_is_read_as_such(self) -> None:
        """What cmd.exe writes into a pipe, in whatever code page this
        machine's console uses."""
        for char in ("é", "ü", "ж", "中"):
            try:
                if char.encode("oem") != char.encode("utf-8"):
                    break
            except UnicodeEncodeError:
                continue
        else:
            self.skipTest("no test character in this OEM code page")
        proc = subprocess.run(["cmd", "/d", "/c", f"echo {char * 3} done"],
                              capture_output=True, timeout=30)
        # cmd writes in the console's code page, which a machine may have set
        # to something other than the OEM one (a CI runner on UTF-8); the
        # real output is only evidence where the two are the same.
        if proc.stdout == f"{char * 3} done\r\n".encode("oem"):
            self.assertEqual(decode_output(proc.stdout), f"{char * 3} done\r\n")
        self.assertEqual(decode_output(f"ok\n{char}\n".encode("oem") * 50), f"ok\n{char}\n" * 50)


class WindowsCleanupPathTest(unittest.IsolatedAsyncioTestCase):
    """SHELL-2 (Windows): the cleanup path must not touch POSIX-only APIs.

    Runs on POSIX too, by patching os.name: taskkill does not exist there, so
    the tree kill exercises its OSError fallback — proving the nt branch never
    reaches os.killpg/os.getpgid, which do not exist on Windows (hence
    ``create=True`` on the patches that stand in for them).
    """

    async def test_kill_tree_reaps_the_leader_without_posix_group_calls(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)",
            stdout=asyncio.subprocess.DEVNULL)
        try:
            with patch("paimon.tools.shell.os.name", "nt"), \
                    patch("paimon.tools.shell.os.killpg", create=True,
                          side_effect=AssertionError("killpg called on the nt path")):
                await _kill_tree(proc, proc.pid)
            self.assertIsNotNone(proc.returncode)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    def test_signal_group_on_windows_kills_the_child_directly(self) -> None:
        from types import SimpleNamespace
        calls = []
        fake = SimpleNamespace(kill=lambda: calls.append("kill"))
        with patch("paimon.tools.shell.os.name", "nt"), \
                patch("paimon.tools.shell.os.killpg", create=True,
                      side_effect=AssertionError("killpg called on the nt path")):
            _signal_group(1234, signal.SIGTERM, fake)
        self.assertEqual(calls, ["kill"])

    def _exiting_command(self, calls: list) -> BackgroundCommand:
        """A running command as terminate_now sees it, without a process."""
        from types import SimpleNamespace
        running = BackgroundCommand.__new__(BackgroundCommand)
        running.killed = False
        running._pgid = 1234
        running._proc = SimpleNamespace(pid=1234, returncode=None,
                                        kill=lambda: calls.append("kill"))
        return running

    def test_terminate_now_on_windows_leaves_the_tree_to_taskkill(self) -> None:
        from types import SimpleNamespace
        calls: list = []
        running = self._exiting_command(calls)
        # Windows' signal module has no SIGKILL at all, so the stand-in has
        # none either: reaching for it is the failure this guards against.
        with patch("paimon.tools.background.os.name", "nt"), \
                patch("paimon.tools.background.signal", SimpleNamespace(SIGTERM=signal.SIGTERM)), \
                patch("paimon.tools.background.subprocess.Popen",
                      side_effect=lambda argv, **_: calls.append(argv)):
            running.terminate_now()

        self.assertTrue(running.killed)
        self.assertEqual(calls, [["taskkill", "/PID", "1234", "/T", "/F"]],
                         "killing the leader first would orphan the tree taskkill walks")

    def test_terminate_now_on_windows_kills_the_leader_when_taskkill_is_missing(self) -> None:
        from types import SimpleNamespace
        calls: list = []
        running = self._exiting_command(calls)
        with patch("paimon.tools.background.os.name", "nt"), \
                patch("paimon.tools.background.signal", SimpleNamespace(SIGTERM=signal.SIGTERM)), \
                patch("paimon.tools.background.subprocess.Popen", side_effect=OSError("no taskkill")):
            running.terminate_now()

        self.assertEqual(calls, ["kill"])


class ShellExecutableTest(unittest.IsolatedAsyncioTestCase):
    """SHELL-3: the executing shell is explicit, and the system prompt reports
    that shell rather than the user's login shell."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()

    def test_prompt_reports_the_shell_the_tool_uses(self) -> None:
        from paimon.prompt import _shell_description
        with patch.dict("os.environ", {"SHELL": "/bin/some-login-shell"}):
            # No shell found on Windows (no Git Bash): the tool runs ComSpec
            # there, and that is what the prompt has to say.
            expected = shell_executable() or os.environ.get("ComSpec") or "cmd.exe"
            self.assertTrue(_shell_description().startswith(expected), _shell_description())

    @posix_shell
    async def test_commands_run_in_the_reported_shell(self) -> None:
        result = (await _shell({"command": "echo shell=$0"}, self.cwd)).text
        self.assertIn(f"shell={shell_executable()}", result)

    @unittest.skipUnless("bash" in (shell_executable() or ""), "bash not installed")
    async def test_bashisms_work_when_bash_is_installed(self) -> None:
        self.assertIn("bash", shell_executable())
        result = (await _shell({"command": "[[ 1 == 1 ]] && echo yes-bash"}, self.cwd)).text
        self.assertIn("yes-bash", result)
        self.assertIn("(exit code 0)", result)


class GitBashDiscoveryTest(unittest.TestCase):
    """Which bash.exe Windows commands run in. Fake installations on disk, so
    this runs on every platform; PATH is written with Windows' ";"."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()

    def _install(self, name: str, git_in: str = "cmd") -> Path:
        """A Git for Windows layout: both bash.exe files, git.exe in ``git_in``."""
        root = self.base / name
        for relative in ("bin/bash.exe", "usr/bin/bash.exe", f"{git_in}/git.exe"):
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_bytes(b"MZ")
        return root

    def _wsl_launcher(self) -> Path:
        """What System32 holds: a bash.exe, and no Git around it."""
        system32 = self.base / "System32"
        system32.mkdir()
        (system32 / "bash.exe").write_bytes(b"MZ")
        return system32

    def test_derived_from_git_on_path_wherever_the_installer_put_it(self) -> None:
        for git_in in ("cmd", "bin", "mingw64/bin"):
            root = self._install(f"Git-{git_in.replace('/', '-')}", git_in)
            path = f"{self._wsl_launcher() if git_in == 'cmd' else ''};{root / git_in}"
            found = find_git_bash({"PATH": path})
            self.assertEqual(found, str(root / "bin" / "bash.exe"), git_in)

    def test_the_wsl_launcher_is_never_chosen(self) -> None:
        system32 = self._wsl_launcher()
        self.assertIsNone(find_git_bash({"PATH": str(system32), "ProgramFiles": str(self.base)}))

    def test_a_git_without_the_msys_layout_is_not_git_for_windows(self) -> None:
        root = self._install("NotGit")
        (root / "usr" / "bin" / "bash.exe").unlink()
        self.assertIsNone(find_git_bash({"PATH": str(root / "cmd")}))

    def test_standard_locations_when_git_is_not_on_path(self) -> None:
        programs = self.base / "Program Files"
        programs.mkdir()
        root = self._install("Program Files/Git")
        self.assertEqual(find_git_bash({"PATH": "", "ProgramFiles": str(programs)}),
                         str(root / "bin" / "bash.exe"))
        local = self.base / "Local"
        root = self._install("Local/Programs/Git")
        self.assertEqual(find_git_bash({"LOCALAPPDATA": str(local)}),
                         str(root / "bin" / "bash.exe"))

    def test_nothing_found_falls_back_to_cmd(self) -> None:
        self.assertIsNone(find_git_bash({"PATH": str(self.base)}))
        with patch("paimon.tools.shell.os.name", "nt"),                 patch("paimon.tools.shell.find_git_bash", return_value=None):
            self.assertIsNone(shell_executable.__wrapped__())

    # Not with os.name patched: that is the one os module, and on POSIX
    # pathlib then refuses to build the Windows paths the lookup works on.
    @unittest.skipUnless(os.name == "nt", "Windows paths")
    def test_the_root_is_known_only_for_the_launcher(self) -> None:
        root = self._install("Git")
        with patch("paimon.tools.shell.os.name", "nt"):
            self.assertEqual(git_bash_root(str(root / "bin" / "bash.exe")), root)
            self.assertIsNone(git_bash_root(str(root / "usr" / "bin" / "bash.exe")))
            self.assertIsNone(git_bash_root("/bin/sh"))
            self.assertIsNone(git_bash_root(None))


@git_bash
class GitBashShellTest(unittest.IsolatedAsyncioTestCase):
    """The shell tool for real under Git Bash on this machine."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()

    async def test_posix_commands_run_with_gits_tools(self) -> None:
        (self.cwd / "中文 文件.txt").write_bytes("你好\n".encode("utf-8"))
        command = "type -P ls find && ls | grep -c txt && cat '中文 文件.txt' && printf '[%s]\\n' 'a  b'"
        outcome = await _shell({"command": command}, self.cwd)

        self.assertEqual(outcome.status, "success", outcome.text)
        lines = outcome.text.splitlines()
        self.assertEqual(lines[:2], ["/usr/bin/ls", "/usr/bin/find"],
                         "GNU tools from Git, not System32's find.exe")
        self.assertEqual(lines[2:5], ["1", "你好", "[a  b]"])

    async def test_a_timeout_kills_what_a_forked_bash_started(self) -> None:
        """A "&" job's process is started by a forked bash that is gone by
        then, so walking parent pids from the shell never finds it."""
        command = "sleep 57 & echo pid $(cat /proc/$!/winpid); sleep 58 | cat"
        with (
            patch("paimon.tools.shell._COMMAND_TIMEOUT", 3.0),
            patch("paimon.tools.shell._KILL_TIMEOUT", 0.5),
        ):
            result = (await _shell({"command": command}, self.cwd)).text
        pid = pid_from(result)
        self.addCleanup(reap, pid)

        self.assertIn("(timed out after 3s)", result)
        for _ in range(40):
            if not alive(pid):
                return
            await asyncio.sleep(0.05)
        self.fail(f"sleep {pid} outlived the timeout")
