import asyncio
import os
import shutil
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from paimon.tools import (
    MAX_OUTPUT,
    BackgroundCommand,
    _kill_tree,
    _shell,
    _signal_group,
    _TaskOutput,
    shell_executable,
    shell_output_dir,
    start_background,
    tail_text,
)
from tests.support.shell import (
    alive,
    filler,
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
        result = await _shell({"command": command}, self.cwd)

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
        result = await _shell({"command": command}, self.cwd)

        body = result.split("\n\n[")[0]
        self.assertEqual(len(body.splitlines()), 2_000)
        self.assertIn("of 3,000 lines", result)
        self.assertIn("full output:", result)
        self.assertIn("(exit code 0)", result)

    async def test_multibyte_output_stays_within_the_result_budget(self) -> None:
        command = filler(400, "😀" * 40, last="TAIL-OK")
        result = await _shell({"command": command}, self.cwd)

        self.assertIn("TAIL-OK", result, "the cut must not eat the last line")
        self.assertLessEqual(len(result), MAX_OUTPUT)

    async def test_one_enormous_line_keeps_its_end(self) -> None:
        """No newline to cut on, so the end of the line is the whole answer."""
        result = await _shell({"command": filler(400, "x" * 100, newline=False)}, self.cwd)

        self.assertIn("of line 1 (line is ", result)
        self.assertLessEqual(len(result), MAX_OUTPUT)
        self.assertIn("full output:", result)
        self.assertEqual(len(overflow_path(result).read_bytes()), 40_000)

    async def test_timeout_keeps_what_the_command_already_printed(self) -> None:
        command = printer("partial-output", then_sleep=True, ignore_term=True)
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
            result = await _shell({"command": spawner(then_sleep=False)}, self.cwd)
            elapsed = time.monotonic() - started
        self.addCleanup(reap, pid_from(result))

        self.assertIn("(exit code 0)", result)
        self.assertLess(elapsed, 3.0, "waiting for stdout EOF would have taken the full timeout")

    async def test_timeout_kills_the_descendants_too(self) -> None:
        command = spawner(then_sleep=True, ignore_term=True)
        with (
            # Long enough for the command to start its child and say so: an
            # interpreter takes far longer to come up than a shell builtin.
            patch("paimon.tools._COMMAND_TIMEOUT", 3.0),
            patch("paimon.tools._KILL_GRACE", 0.05),
            patch("paimon.tools._KILL_TIMEOUT", 0.5),
        ):
            result = await _shell({"command": command}, self.cwd)
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
        self.addCleanup(running.terminate_now)
        return running

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

        with patch("paimon.tools._KILL_GRACE", 0.05), patch("paimon.tools._KILL_TIMEOUT", 0.5):
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
            with patch("paimon.tools.os.name", "nt"), \
                    patch("paimon.tools.os.killpg", create=True,
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
        with patch("paimon.tools.os.name", "nt"), \
                patch("paimon.tools.os.killpg", create=True,
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
        with patch("paimon.tools.os.name", "nt"), \
                patch("paimon.tools.signal", SimpleNamespace(SIGTERM=signal.SIGTERM)), \
                patch("paimon.tools.subprocess.Popen",
                      side_effect=lambda argv, **_: calls.append(argv)):
            running.terminate_now()

        self.assertTrue(running.killed)
        self.assertEqual(calls, [["taskkill", "/PID", "1234", "/T", "/F"]],
                         "killing the leader first would orphan the tree taskkill walks")

    def test_terminate_now_on_windows_kills_the_leader_when_taskkill_is_missing(self) -> None:
        from types import SimpleNamespace
        calls: list = []
        running = self._exiting_command(calls)
        with patch("paimon.tools.os.name", "nt"), \
                patch("paimon.tools.signal", SimpleNamespace(SIGTERM=signal.SIGTERM)), \
                patch("paimon.tools.subprocess.Popen", side_effect=OSError("no taskkill")):
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
            # No explicit shell on Windows: the tool runs ComSpec there, and
            # that is what the prompt has to say.
            expected = shell_executable() or os.environ.get("ComSpec") or "cmd.exe"
            self.assertEqual(_shell_description(), expected)

    @posix_shell
    async def test_commands_run_in_the_reported_shell(self) -> None:
        result = await _shell({"command": "echo shell=$0"}, self.cwd)
        self.assertIn(f"shell={shell_executable()}", result)

    @posix_shell
    @unittest.skipUnless(shutil.which("bash"), "bash not installed")
    async def test_bashisms_work_when_bash_is_installed(self) -> None:
        self.assertIn("bash", shell_executable())
        result = await _shell({"command": "[[ 1 == 1 ]] && echo yes-bash"}, self.cwd)
        self.assertIn("yes-bash", result)
        self.assertIn("(exit code 0)", result)
