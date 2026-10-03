"""Shell output generators and readers shared by tool tests.

The commands here are Python one-liners rather than shell loops: the shell tool
runs /bin/sh or bash on POSIX and cmd.exe on Windows, and the only syntax the
two share is a quoted program with a quoted argument. What the tests exercise
is what paimon does with a command's output and its process tree, which does
not depend on who wrote the bytes.
"""

import os
import re
import signal
import subprocess
import sys
import unittest
from pathlib import Path

# For the few tests that are about the POSIX shell itself ($0, bashisms).
posix_shell = unittest.skipUnless(os.name == "posix", "needs a POSIX shell")

# Stops SIGTERM from ending the process, so a test reaches the forced kill.
# A no-op on Windows, where nothing delivers SIGTERM to begin with.
IGNORE_TERM = "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); "

_SLEEPER = "import time; time.sleep(30)"


def python_command(code: str) -> str:
    """``code`` as a command line both sh and cmd.exe run the same way.

    The code goes inside double quotes, so it must hold none itself, and no
    ``$``, backtick or ``%`` either; a backslash escape such as ``\\n`` passes
    through both shells untouched and is Python's to interpret.
    """
    assert not set(code) & set('"$`%'), code
    return f'"{sys.executable}" -c "{code}"'


def filler(lines: int, payload: str, newline: bool = True, *,
           last: str = "", exit_code: int = 0) -> str:
    """A command printing ``payload`` ``lines`` times, then ``last`` on a line
    of its own, and exiting with ``exit_code``.

    Written as bytes: text mode would turn every newline into CRLF on Windows
    and encode with the console's code page rather than UTF-8.
    """
    chunk = ascii(payload + ("\n" if newline else ""))
    code = f"import sys; w = sys.stdout.buffer.write; w({chunk}.encode('utf-8') * {lines}); "
    if last:
        code += f"w({ascii(last + chr(10))}.encode('utf-8')); "
    return python_command(code + f"sys.exit({exit_code})")


def printer(text: str, *, then_sleep: bool = False, exit_code: int = 0,
            ignore_term: bool = False) -> str:
    """A command that prints ``text`` on a line, then sleeps or exits."""
    code = IGNORE_TERM if ignore_term else ""
    code += (f"import sys; sys.stdout.buffer.write({ascii(text + chr(10))}.encode('utf-8')); "
             "sys.stdout.flush(); ")
    code += _SLEEPER if then_sleep else f"sys.exit({exit_code})"
    return python_command(code)


def sleeper(*, ignore_term: bool = False) -> str:
    """A command that just outstays the test."""
    return python_command((IGNORE_TERM if ignore_term else "") + _SLEEPER)


def spawner(*, then_sleep: bool, ignore_term: bool = False, own_pid: bool = False) -> str:
    """A command that starts a long-lived child and prints ``pid <n>``.

    The pid is the child's — the descendant a tree kill has to reach — or the
    command's own with ``own_pid``. With ``then_sleep`` the command stays
    alive beside its child; without, it exits and leaves the child behind.
    """
    code = IGNORE_TERM if ignore_term else ""
    code += ("import os, subprocess, sys, time; "
             f"child = subprocess.Popen([sys.executable, '-c', '{_SLEEPER}']); "
             f"print('pid', {'os.getpid()' if own_pid else 'child.pid'}, flush=True); ")
    if then_sleep:
        code += "time.sleep(30)"
    return python_command(code)


def pid_from(output: str) -> int:
    match = re.search(r"pid (\d+)", output)
    assert match, f"no pid in output: {output!r}"
    return int(match.group(1))


def alive(pid: int) -> bool:
    """Whether a process with this pid exists.

    Never ``os.kill(pid, 0)`` on Windows: there any signal number that is not
    a console event is passed to TerminateProcess as the exit code, so the
    probe would kill the process it was asking about.
    """
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            kernel32.GetExitCodeProcess(wintypes.HANDLE(handle), ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(wintypes.HANDLE(handle))
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def reap(pid: int) -> None:
    """Kill a process a test started, whatever its assertions did."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def overflow_path(result: str) -> Path:
    match = re.search(r"full output: (\S+?)\]", result)
    assert match, f"no overflow path in result: {result[-300:]!r}"
    return Path(match.group(1))
