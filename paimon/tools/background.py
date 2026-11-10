"""Background commands, started by run_background and shown in their own tab."""

import asyncio
import os
import shlex
import shutil
import signal
import subprocess
from pathlib import Path

from .shell import (
    _SHELL_MAX_BYTES,
    _drain,
    _format_size,
    _kill_tree,
    _pump,
    _signal_group,
    _spawn_shell,
    _stop_reader,
    _terminate_job,
    decode_output,
    shell_executable,
)


# ---- background commands ---------------------------------------------------
#
# A command that outlives the turn that started it: no timeout, no result, just
# a buffer its own tab renders and its agent reads. Deliberately not a PTY —
# there is no keyboard and no terminal emulation, which is also why output can
# arrive in blocks (see _line_buffered).

# What a background command keeps in memory. Larger than the shell tool's
# budget because this is a live view the user scrolls, while only the tail of
# it is ever handed to the model.
_TASK_BUFFER_BYTES = 256 * 1024


class _TaskOutput:
    """A background command's output, read incrementally by several readers.

    Bounded: the oldest bytes go once the buffer is full. Cursors are global
    byte offsets rather than buffer indexes, so a reader that fell behind the
    trimming is told how much it missed instead of being handed the wrong
    bytes.
    """

    def __init__(self, limit: int = _TASK_BUFFER_BYTES) -> None:
        self._buffer = bytearray()
        self._limit = limit
        self.total_bytes = 0

    @property
    def start(self) -> int:
        """Offset of the oldest byte still retained."""
        return self.total_bytes - len(self._buffer)

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total_bytes += len(chunk)
        self._buffer += chunk
        if len(self._buffer) > self._limit:
            del self._buffer[: len(self._buffer) - self._limit]

    def since(self, offset: int) -> tuple[bytes, int, int]:
        """Output after ``offset``, as (bytes, new cursor, bytes dropped).

        Bytes, not text: a chunk boundary falls wherever the pipe said it did,
        and decoding one that splits a character would put a replacement mark
        in the middle of the output every time. Callers hold the fragment back
        instead.
        """
        offset = max(0, min(offset, self.total_bytes))
        dropped = max(0, self.start - offset)
        return bytes(self._buffer[offset + dropped - self.start:]), self.total_bytes, dropped


def _line_buffered(command: str) -> str:
    """``command`` wrapped so its libc stdio flushes per line, where it can be.

    A background command's stdout is a pipe, so libc buffers it in blocks and a
    tab can sit empty for minutes under a chatty build before everything
    arrives at once. stdbuf works through LD_PRELOAD, which descendants
    inherit; it does nothing for Go or Rust programs, or on musl. This is a
    mitigation, not a fix — output arriving late is what a pipe instead of a
    terminal costs.

    Left alone on Windows, whichever shell runs the command: the native
    programs whose buffering matters there (python, node, a compiler) load no
    LD_PRELOAD library, and a stdbuf found on PATH might be any port of it,
    running the command in some other shell than the one the system prompt
    names.
    """
    shell = shell_executable()
    if shell is None or os.name == "nt" or not shutil.which("stdbuf"):
        return command
    return f"stdbuf -oL -eL {shlex.quote(shell)} -c {shlex.quote(command)}"


class BackgroundCommand:
    """One command running past the end of the turn that started it."""

    def __init__(
        self,
        command: str,
        proc: asyncio.subprocess.Process,
        stream: asyncio.StreamReader,
        transport: asyncio.ReadTransport,
        pgid: int,
    ) -> None:
        self.command = command
        self.output = _TaskOutput()
        self.exit_code: int | None = None
        self.killed = False
        self._proc = proc
        self._stream = stream
        self._transport = transport
        self._pgid = pgid
        self._reading = asyncio.ensure_future(self._read())
        self._killing: asyncio.Future | None = None

    @property
    def running(self) -> bool:
        return self.exit_code is None

    async def _read(self) -> None:
        reader = asyncio.create_task(_pump(self._stream, self.output))
        try:
            await self._proc.wait()
            await _drain(reader, self.output)
        finally:
            await _stop_reader(reader, self._transport)
            self.exit_code = self._proc.returncode

    async def wait(self) -> int | None:
        """Block until the command has exited and its output has been drained.

        Shielded, because whoever waits is liable to be cancelled — a job being
        killed, the app going down — and cancelling the read would abandon
        whatever the pipe still held.
        """
        await asyncio.shield(self._reading)
        return self.exit_code

    def kill(self) -> None:
        """Stop the whole process group. Returns at once; reaping runs on.

        Callers are all in places that cannot wait — a pane closing, a session
        being swapped out — and the escalation from SIGTERM to SIGKILL takes
        seconds of waiting the UI does not have.
        """
        if self.killed or not self.running:
            return
        self.killed = True
        # Held so the escalation is not collected halfway: the loop keeps only
        # a weak reference to a task.
        self._killing = asyncio.ensure_future(_kill_tree(self._proc, self._pgid))

    def terminate_now(self) -> None:
        """Signal the group without awaiting anything, on the way out.

        The app is exiting: nothing here will be reaped, and a group left
        unsignalled is reparented to init and keeps running. SIGKILL follows
        SIGTERM immediately rather than after a grace period, because there is
        no loop left to come back and finish the job.
        """
        self.killed = True
        if self._proc.returncode is not None:
            return
        if os.name == "nt":
            # Terminating the command's job is synchronous, so nothing has to
            # be awaited for the whole tree to go.
            if _terminate_job(self._proc):
                return
            # Without a job there is nothing left to await taskkill on; fire
            # it detached so the tree still goes down after this process is
            # gone. The leader is left
            # to taskkill as well: it finds the tree by walking parent pids
            # from the leader, so killing that first would orphan the rest.
            # There is no SIGKILL to escalate to here (the name does not even
            # exist on Windows), and /F is already the forced kill.
            try:
                subprocess.Popen(["taskkill", "/PID", str(self._proc.pid), "/T", "/F"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                _signal_group(self._pgid, signal.SIGTERM, self._proc)
            return
        _signal_group(self._pgid, signal.SIGTERM, self._proc)
        _signal_group(self._pgid, signal.SIGKILL, self._proc)


async def start_background(command: str, cwd: Path) -> BackgroundCommand:
    """Start ``command`` detached from the turn, in its own process group."""
    proc, stream, transport = await _spawn_shell(
        _line_buffered(command),
        cwd,
        # Python is the one interpreter that can be told to stop block
        # buffering from the outside, and the one this agent starts most.
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return BackgroundCommand(command, proc, stream, transport, proc.pid)


def tail_text(data: bytes, dropped: int = 0, limit: int = _SHELL_MAX_BYTES) -> str:
    """Output for the model: the end of it, with a note about what was cut."""
    trimmed = len(data) - limit
    if trimmed > 0:
        data = data[-limit:]
        dropped += trimmed
    text = decode_output(data)
    if dropped > 0:
        text = f"[{_format_size(dropped)} of earlier output dropped]\n{text}"
    return text
