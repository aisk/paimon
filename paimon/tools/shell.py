"""The shell tool: spawning a command, collecting its output, killing its tree."""

import asyncio
import os
import shutil
import signal
import subprocess
import time
import weakref
from functools import cache
from pathlib import Path

from ..session import data_dir
from .base import MAX_OUTPUT, ToolContext
from .files import _real
from .result import ToolResult


_COMMAND_TIMEOUT = 120.0
_KILL_GRACE = 2.0  # seconds to wait after SIGTERM before forcing SIGKILL
_KILL_TIMEOUT = 2.0  # maximum wait to reap the process after SIGKILL
_READ_CHUNK = 64 * 1024
# Once the command itself has exited, a descendant may still be writing to the
# pipe it inherited. Keep reading while bytes arrive, but never for long.
_DRAIN_GRACE = 0.1
_DRAIN_TOTAL = 2.0

_SHELL_MAX_LINES = 2_000
# The footer and the status line share the result budget with the output, so
# the byte cap is derived rather than written down. That keeps one invariant:
# a shell result never exceeds MAX_OUTPUT characters, so execute_tool's generic
# truncation — which cuts from the head, and would take back exactly the tail
# this collector went to the trouble of keeping — never fires for shell.
_SHELL_FOOTER_BUDGET = 2_000
_SHELL_MAX_BYTES = MAX_OUTPUT - _SHELL_FOOTER_BUDGET
_OVERFLOW_TTL = 7 * 24 * 60 * 60  # seconds an overflow file is kept around


def shell_output_dir() -> Path:
    """Where a command's full output goes when more of it than fits is produced."""
    return data_dir() / "shell-output"


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def decode_output(data: bytes) -> str:
    """Command output as text.

    Git Bash and the programs Git for Windows ships write UTF-8, but Windows
    console programs (cmd.exe built-ins above all, and native programs run
    from either shell) write in the OEM code page, so output that is not
    valid UTF-8 is read as that instead of being turned into replacement
    marks.
    """
    if os.name == "nt":
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("oem", errors="replace")
    return data.decode("utf-8", errors="replace")


def _prune_overflow(directory: Path) -> None:
    """Drop overflow files old enough that no live session can still cite them."""
    cutoff = time.time() - _OVERFLOW_TTL
    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink()
        except OSError:
            pass


class _OutputTail:
    """Bounded collector for one command's streamed output.

    Keeps the last ``_SHELL_MAX_BYTES`` in memory and, once the command writes
    more than fits, mirrors every byte into an overflow file. The tail is what
    the model needs (the error and the exit code live at the end) and the file
    is how it can still get back to the first error of a long build log.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._newlines = 0
        self._open_line = False
        self._line_bytes = 0
        self._fd: int | None = None
        self.path: Path | None = None
        self.total_bytes = 0

    @property
    def total_lines(self) -> int:
        return self._newlines + (1 if self._open_line else 0)

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total_bytes += len(chunk)
        self._newlines += chunk.count(b"\n")
        last_newline = chunk.rfind(b"\n")
        if last_newline == -1:
            self._line_bytes += len(chunk)
        else:
            self._line_bytes = len(chunk) - last_newline - 1
        self._open_line = not chunk.endswith(b"\n")

        if self._fd is None and (self.total_bytes > _SHELL_MAX_BYTES
                                 or self.total_lines > _SHELL_MAX_LINES):
            self._start_overflow()
        self._write(chunk)
        self._buffer += chunk
        # Only bounds memory; the budget itself is applied in render(). Copying
        # on every chunk would make a chatty command quadratic in chunk count.
        if len(self._buffer) > 2 * _SHELL_MAX_BYTES:
            del self._buffer[:len(self._buffer) - _SHELL_MAX_BYTES]

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def render(self) -> str:
        """The retained tail, with a footer whenever anything was dropped."""
        data = bytes(self._buffer)
        trimmed = len(data) < self.total_bytes
        if len(data) > _SHELL_MAX_BYTES:
            data = data[len(data) - _SHELL_MAX_BYTES:]
            trimmed = True
        text = decode_output(data)
        partial_line = False
        if trimmed:
            newline = text.find("\n")
            if newline == -1:
                # A single line longer than the whole budget: its end is all we
                # can keep, and the end is where the message usually is.
                partial_line = True
            else:
                text = text[newline + 1:]  # drop the fragment the cut left behind
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if len(lines) > _SHELL_MAX_LINES:
            lines = lines[-_SHELL_MAX_LINES:]
        text = "\n".join(lines)

        if not trimmed and len(lines) >= self.total_lines:
            return text
        where = f"; full output: {self.path}" if self.path is not None else ""
        shown = _format_size(len(text.encode("utf-8", errors="replace")))
        if partial_line:
            note = (f"showing last {shown} of line {self.total_lines:,} "
                    f"(line is {_format_size(self._line_bytes)})")
        else:
            note = (f"showing last {len(lines):,} of {self.total_lines:,} lines "
                    f"({shown} of {_format_size(self.total_bytes)})")
        return f"{text}\n\n[{note}{where}]"

    def _start_overflow(self) -> None:
        directory = shell_output_dir()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = directory / f"{stamp}-{os.getpid()}-{os.urandom(3).hex()}.log"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            _prune_overflow(directory)
            # The output can hold anything the command printed, so keep it
            # private the way config.py keeps the api key file private.
            self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError:
            return  # losing the head is bad; failing the command over it is worse
        self.path = path
        self._write(bytes(self._buffer))

    def _write(self, chunk: bytes) -> None:
        if self._fd is None or not chunk:
            return
        try:
            view = memoryview(chunk)
            while view:
                # os.write is free to accept only part of a large buffer.
                view = view[os.write(self._fd, view):]
        except OSError:
            # A half-written file must not be advertised as the full output.
            self.close()
            if self.path is not None:
                try:
                    self.path.unlink()
                except OSError:
                    pass
                self.path = None


def _signal_group(pgid: int, sig: int, proc: asyncio.subprocess.Process) -> None:
    """Send a signal to the whole process group, falling back to the child alone.

    Windows has no process groups to signal (os.killpg does not exist there);
    the tree is handled by the job or taskkill path in _kill_tree, and this falls
    back to killing the child itself.
    """
    if os.name == "nt":
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
        return
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


@cache
def _win32():
    """kernel32 and ntdll with the signatures the job code below relies on.

    Declared rather than left to ctypes' defaults, which pass and return C
    ints and would truncate a 64-bit HANDLE.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    handle, dword = wintypes.HANDLE, wintypes.DWORD
    for name, restype, argtypes in (
        ("CreateJobObjectW", handle, (wintypes.LPVOID, wintypes.LPCWSTR)),
        ("SetInformationJobObject", wintypes.BOOL, (handle, ctypes.c_int, wintypes.LPVOID, dword)),
        ("AssignProcessToJobObject", wintypes.BOOL, (handle, handle)),
        ("TerminateJobObject", wintypes.BOOL, (handle, wintypes.UINT)),
        ("TerminateProcess", wintypes.BOOL, (handle, wintypes.UINT)),
        ("CloseHandle", wintypes.BOOL, (handle,)),
    ):
        function = getattr(kernel32, name)
        function.restype, function.argtypes = restype, argtypes
    ntdll.NtResumeProcess.restype = ctypes.c_long
    ntdll.NtResumeProcess.argtypes = (handle,)
    return kernel32, ntdll


def _contain(process: int) -> int | None:
    """Put the suspended process behind the handle ``process`` in a job
    object of its own, then resume it. Returns the job handle, or None when
    there is no job (the process is resumed either way, and killed if even
    that fails).

    Takes the handle the process was created with rather than its pid: that
    one has every right needed, and cannot fail to open and leave the command
    suspended for good.

    The job is Windows' process group: every process the command starts
    joins it, whoever its parent was, so terminating the job reaches all of
    them. The parent-pid tree taskkill /T walks does not: under Git Bash a
    pipeline's or a "&" job's processes are created by a forked bash that has
    already exited, and they outlived the kill. Started suspended so nothing
    can be spawned before the job is in place. Breakaway is not allowed:
    Cygwin, which Git Bash runs on, asks for it whenever it may, so allowing
    it lets exactly those processes escape.
    """
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):  # JOBOBJECT_BASIC_LIMIT_INFORMATION
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    kernel32, ntdll = _win32()
    job = None
    try:
        job = kernel32.CreateJobObjectW(None, None)
        limits = BasicLimits(LimitFlags=0)
        if job and not (kernel32.SetInformationJobObject(job, 2, ctypes.byref(limits),
                                                         ctypes.sizeof(limits))
                        and kernel32.AssignProcessToJobObject(job, process)):
            kernel32.CloseHandle(job)
            job = None
    finally:
        if ntdll.NtResumeProcess(process) != 0:
            kernel32.TerminateProcess(process, 1)  # a command that never runs would hang the turn
    return job or None


def _terminate_job(proc: asyncio.subprocess.Process) -> bool:
    """Kill every process in the command's job; False when it has none."""
    job = getattr(proc, "_paimon_job", None)
    return job is not None and bool(_win32()[0].TerminateJobObject(job, 1))


async def _taskkill_tree(proc: asyncio.subprocess.Process) -> None:
    """Windows tree kill: the command's job when it has one (see _contain),
    else taskkill /T /F, which walks the parent-pid tree. That is best
    effort — a child whose parent already exited cannot be found by it —
    with a direct kill of the leader as the fallback.
    """
    if _terminate_job(proc):
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=_KILL_TIMEOUT)
        except asyncio.TimeoutError:
            pass
        return
    try:
        killer = await asyncio.create_subprocess_exec(
            "taskkill", "/PID", str(proc.pid), "/T", "/F",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await asyncio.wait_for(killer.wait(), timeout=_KILL_GRACE + _KILL_TIMEOUT)
    except (OSError, asyncio.TimeoutError):
        pass
    if proc.returncode is None:
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=_KILL_TIMEOUT)
        except asyncio.TimeoutError:
            pass


async def _kill_tree(proc: asyncio.subprocess.Process, pgid: int) -> None:
    """Terminate and reap a command's process group without letting cleanup hang.

    The group is signalled even when the shell itself has already exited: the
    processes it backgrounded are precisely the ones that outlive it, and on a
    timeout or an interrupt the whole tree has to go.
    """
    if os.name == "nt":
        await _taskkill_tree(proc)
        return
    _signal_group(pgid, signal.SIGTERM, proc)
    if proc.returncode is None:
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=_KILL_GRACE)
        except asyncio.TimeoutError:
            pass
    # Unconditional: SIGTERM may have been trapped, and a leader that exited
    # says nothing about the rest of its group.
    _signal_group(pgid, signal.SIGKILL, proc)
    if proc.returncode is None:
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=_KILL_TIMEOUT)
        except asyncio.TimeoutError:
            # The OS should eventually reap it; most importantly, cleanup cannot
            # block the agent indefinitely.
            pass


async def _pump(stream: asyncio.StreamReader, tail: _OutputTail) -> None:
    while True:
        chunk = await stream.read(_READ_CHUNK)
        if not chunk:
            return
        tail.append(chunk)


def _git_root(directory: Path) -> Path | None:
    """The Git for Windows installation ``directory`` belongs to, if any.

    Recognized by the two bash.exe files every installation has, so a stray
    bin\\bash.exe elsewhere (System32 holds the WSL one) does not qualify.
    """
    if (directory / "bin" / "bash.exe").is_file() and (directory / "usr" / "bin" / "bash.exe").is_file():
        return directory
    return None


def find_git_bash(environ: dict | None = None) -> str | None:
    """Git for Windows' bash.exe, or None when Git for Windows is not found.

    Never shutil.which("bash"): on Windows that is usually System32's or
    WindowsApps' bash.exe, the WSL launcher, which runs the command inside a
    Linux VM with a filesystem of its own.

    The installation is found from git.exe on PATH, which sits in <Git>\\cmd,
    <Git>\\bin or <Git>\\mingw64\\bin depending on the installer's PATH
    option (a portable copy is found the same way), then in the standard
    install locations. Only absolute PATH entries are searched, so a git.exe
    in the working directory cannot nominate a bash.exe next to it.

    The bash.exe returned is <Git>\\bin\\bash.exe, not <Git>\\usr\\bin\\bash.exe.
    The former is Git's launcher: it sets MSYSTEM and HOME and puts
    <Git>\\mingw64\\bin and <Git>\\usr\\bin first on PATH before starting the
    latter. Started directly, usr\\bin\\bash.exe searches the Windows PATH
    alone, where "ls" or "find" is whatever happens to be installed there
    (System32's find.exe, another coreutils port) or nothing at all.
    """
    environ = os.environ if environ is None else environ
    candidates: list[Path] = []
    for entry in (environ.get("PATH") or "").split(";"):
        entry = entry.strip().strip('"')
        if not entry or not os.path.isabs(entry):
            continue
        git = Path(entry) / "git.exe"
        if git.is_file():
            try:
                git = git.resolve()  # a WinGet link points into the real install
            except OSError:
                pass
            candidates += [git.parent.parent, git.parent.parent.parent]
    for variable in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
        if environ.get(variable):
            candidates.append(Path(environ[variable]) / "Git")
    if environ.get("LOCALAPPDATA"):  # the installer's per-user option
        candidates.append(Path(environ["LOCALAPPDATA"]) / "Programs" / "Git")
    for candidate in candidates:
        root = _git_root(candidate)
        if root is not None:
            return str(root / "bin" / "bash.exe")
    return None


@cache
def shell_executable() -> str | None:
    """The shell the shell tool actually runs commands with.

    Bash is preferred when installed, so the bashisms models habitually write
    ([[ ]], process substitution) behave; /bin/sh is the fallback. On Windows
    that bash is Git for Windows' (see find_git_bash), and None — cmd.exe via
    ComSpec — is left for a machine without it. The system prompt reports
    this same value: the model must never be told a shell the tool does not
    use (previously it reported $SHELL while running /bin/sh).

    Cached because the prompt, every command and every permission check ask,
    and the answer only changes when software is installed or removed.
    """
    if os.name == "nt":
        return find_git_bash()
    for candidate in ("/bin/bash", "/usr/bin/bash"):
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("bash") or "/bin/sh"


def git_bash_root(shell: str | None) -> Path | None:
    """The Git for Windows installation ``shell`` is the launcher of, else None."""
    if os.name != "nt" or shell is None:
        return None
    path = Path(shell)
    if path.name.lower() != "bash.exe" or path.parent.name.lower() != "bin":
        return None
    return _git_root(path.parent.parent)


def _output_pipe():
    """A pipe for a command's output: the read end for the loop, the write fd
    for the child.

    On Windows the proactor loop can only read a handle opened for overlapped
    I/O, which os.pipe() does not produce (registering it with the completion
    port fails and the output is silently lost), so the pipe is built the way
    asyncio builds its own subprocess pipes.
    """
    if os.name == "nt":
        import msvcrt
        from asyncio import windows_utils

        read_handle, write_handle = windows_utils.pipe(overlapped=(True, False))
        return windows_utils.PipeHandle(read_handle), msvcrt.open_osfhandle(write_handle, 0)
    read_fd, write_fd = os.pipe()
    return os.fdopen(read_fd, "rb", 0), write_fd


async def _spawn_shell(
    command: str, cwd: Path, env: dict | None = None
) -> tuple[asyncio.subprocess.Process, asyncio.StreamReader, asyncio.ReadTransport]:
    """Start ``command`` with its output on a pipe asyncio does not manage.

    With stdout=PIPE, Process.wait() on Python 3.11+ also waits for the pipe
    to reach EOF (gh-88050): a backgrounded descendant that inherited the pipe
    then holds the turn open long after the command itself exited. Handing the
    shell the write end of a plain os.pipe() keeps wait() about the shell
    alone; the read end comes back as a stream plus the transport to close
    when a descendant will not let go.
    """
    pipe, write_fd = _output_pipe()
    try:
        options = dict(
            cwd=str(cwd),
            # Without this a command that reads stdin (a bare "cat") would
            # block until the timeout, eating the user's keystrokes in the TUI.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=write_fd,
            stderr=write_fd,
            env=env,
        )
        if os.name == "nt":
            # Suspended until _contain has put it in a job of its own, which
            # is what cleanup kills as one unit; start_new_session is
            # POSIX-only.
            options["creationflags"] = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                                        | getattr(subprocess, "CREATE_SUSPENDED", 0x4))
        else:
            # start_new_session puts the child in its own process group so the
            # whole tree (the shell plus anything it spawns) can be killed on
            # timeout/interrupt.
            options["start_new_session"] = True
        shell = shell_executable()
        if shell is None:
            proc = await asyncio.create_subprocess_shell(command, **options)
        else:
            # Exec form rather than create_subprocess_shell's executable
            # override: the override would run bash with argv[0] "sh", which
            # flips it into POSIX mode.
            proc = await asyncio.create_subprocess_exec(shell, "-c", command, **options)
        if os.name == "nt":
            popen = proc._transport.get_extra_info("subprocess")
            job = _contain(int(popen._handle))
            if job is not None:
                proc._paimon_job = job
                weakref.finalize(proc, _win32()[0].CloseHandle, job)
    except BaseException:
        pipe.close()
        raise
    finally:
        os.close(write_fd)
    stream = asyncio.StreamReader()
    try:
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(stream), pipe
        )
    except BaseException:
        pipe.close()
        raise
    return proc, stream, transport


def _stop_reading(transport: asyncio.ReadTransport) -> None:
    """Close the read end of a pipe a descendant is still holding open.

    Without this the event loop keeps draining that pipe into the stream
    buffer for as long as the descendant lives, unbounded and unread.
    """
    try:
        transport.close()
    except OSError:
        pass


async def _drain(reader: asyncio.Task, tail) -> None:
    """Take what the pipe still holds now that the command itself has exited.

    Re-armed while bytes keep arriving, so a descendant in the middle of
    writing is not cut off mid-sentence, and never for long.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _DRAIN_TOTAL
    while not reader.done():
        seen = tail.total_bytes
        try:
            await asyncio.wait_for(asyncio.shield(reader), timeout=_DRAIN_GRACE)
        except asyncio.TimeoutError:
            if tail.total_bytes == seen or loop.time() >= deadline:
                return


async def _stop_reader(reader: asyncio.Task, transport: asyncio.ReadTransport) -> None:
    """End the pump task, closing the pipe if a descendant still holds it open."""
    if not reader.done():
        reader.cancel()
    # Unconditional: the proactor transport stays open after EOF, unlike the
    # POSIX one, and closing a closed transport does nothing.
    _stop_reading(transport)
    try:
        await reader
    except (asyncio.CancelledError, Exception):  # noqa: BLE001 — cleanup only
        pass


async def _collect(
    proc: asyncio.subprocess.Process,
    stream: asyncio.StreamReader,
    transport: asyncio.ReadTransport,
    tail: _OutputTail,
) -> None:
    """Read output until the command exits, then drain what is still in flight.

    Deliberately never waits for stdout EOF. A backgrounded descendant inherits
    the pipe and can hold it open for as long as it runs ("svc & echo ready"),
    which would block the turn until the timeout with the command itself long
    finished.
    """
    reader = asyncio.create_task(_pump(stream, tail))
    try:
        await asyncio.wait_for(proc.wait(), timeout=_COMMAND_TIMEOUT)
        await _drain(reader, tail)
    finally:
        await _stop_reader(reader, transport)


async def _shell(args: dict, cwd: Path, ctx: ToolContext | None = None) -> ToolResult:
    """Run a command; a non-zero exit or a timeout is a failed result.

    The exit status is written into the text as well, since that is the only
    place the model reads it from; the result's own status is what everything
    else goes by.
    """
    proc, stream, transport = await _spawn_shell(args["command"], cwd)
    # The child leads its own group, so the group id is its pid. Recorded here
    # because os.getpgid() stops working the moment that leader is reaped.
    pgid = proc.pid
    tail = _OutputTail()
    try:
        try:
            await _collect(proc, stream, transport, tail)
            status = f"(exit code {proc.returncode})"
            problem = f"exit {proc.returncode}" if proc.returncode else ""
        except asyncio.TimeoutError:
            await _kill_tree(proc, pgid)
            status = f"(timed out after {_COMMAND_TIMEOUT:g}s)"
            problem = "timed out"
        except asyncio.CancelledError:
            await _kill_tree(proc, pgid)
            raise
    finally:
        tail.close()
        # Recorded so this agent — and only this agent — can read the file back
        # without another confirmation; see gate().
        if ctx is not None and tail.path is not None:
            real = _real(tail.path)
            if real is not None:
                ctx.shell_outputs.add(real)
    out = tail.render()
    # The status follows the output instead of replacing it: a timeout or a
    # non-zero exit is exactly when what the command managed to print matters.
    text = f"{out}\n{status}" if out.strip() else status
    return ToolResult(text, "failed", problem) if problem else ToolResult(text)


async def user_shell(command: str, cwd: Path, ctx: ToolContext | None = None) -> str:
    """Run a command the user typed themselves, with no permission gate.

    The TUI's "!" prefix. Same execution as the shell tool — same timeout, the
    same process group killed on interrupt, the same tail kept with the rest
    spilled to a file — so a command the user runs and one the model runs
    behave alike and read alike in the log.
    """
    return (await _shell({"command": command}, cwd, ctx)).text
