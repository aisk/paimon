"""Cross-process advisory file locks.

POSIX uses flock, which is tied to the locking file description: other fds
the process opens and closes on the same file (e.g. for appending) do not
release it, unlike POSIX record locks. Windows has no advisory whole-file
lock, so one byte far past any real offset is range-locked instead; the
(mandatory) lock then never collides with reads or appends. The locked handle
is also opened sharing delete access, so holding a lock does not pin the file
in place any more than flock does. Both variants die with the process, so a
crash never leaves a stale lock behind.

Locks are refcounted per process: re-acquiring a path this process already
holds succeeds, while other processes are kept out until the last release.
"""

import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes
else:
    import fcntl

_LOCK_OFFSET = 1 << 62  # Windows: byte position no real file offset reaches
_POLL_SECONDS = 0.05

# Held locks: resolved path -> [fd, refcount].
_held: dict[str, list] = {}


def _open(path: Path) -> int:
    """A read-write fd on path, created if missing, to hold the lock on.

    os.open on Windows never grants FILE_SHARE_DELETE, so the file could be
    neither removed nor renamed for as long as the lock was held — a session
    log could not be deleted while its pane was open, which flock allows on
    POSIX. CreateFileW with every share mode gives the same latitude here.
    """
    if sys.platform != "win32":
        return os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    handle = kernel32.CreateFileW(
        str(path),
        0x80000000 | 0x40000000,  # GENERIC_READ | GENERIC_WRITE
        0x1 | 0x2 | 0x4,  # FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
        None,
        4,  # OPEN_ALWAYS
        0x80,  # FILE_ATTRIBUTE_NORMAL
        None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(handle, os.O_RDWR)
    except OSError:
        kernel32.CloseHandle(wintypes.HANDLE(handle))
        raise


def _try_lock(fd: int) -> None:
    """Non-blocking exclusive lock on fd; raises OSError if already taken."""
    if sys.platform == "win32":
        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def acquire(path: Path, timeout: float = 0.0) -> bool:
    """Claim an exclusive lock on path, giving whoever holds it up to
    ``timeout`` seconds to let go. False if it never becomes free."""
    key = str(path.resolve())
    entry = _held.get(key)
    if entry:
        entry[1] += 1
        return True
    fd = _open(path)
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        try:
            _try_lock(fd)
        except OSError:
            if time.monotonic() < deadline:
                time.sleep(_POLL_SECONDS)
                continue
            os.close(fd)
            return False
        _held[key] = [fd, 1]
        return True


def release(path: Path) -> None:
    """Drop one claim; the OS lock is released with the last one."""
    key = str(path.resolve())
    entry = _held.get(key)
    if not entry:
        return
    entry[1] -= 1
    if entry[1] == 0:
        os.close(entry[0])  # closing the locked fd releases the OS lock
        del _held[key]


def held(path: Path) -> bool:
    """Whether this process currently holds a claim on path."""
    return str(path.resolve()) in _held
