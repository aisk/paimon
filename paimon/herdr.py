"""Tell Herdr what this process is doing, when it runs inside a Herdr pane.

Herdr is a terminal workspace manager for coding agents. A process in one of
its panes inherits a few environment variables, and an agent that reports
through them gets its state in Herdr's sidebar, notifications when it finishes
or needs an answer, and its session back after a Herdr restart.

Reports go through the Herdr binary rather than its socket: that is the
portable path, and the one Herdr documents for agents that own their
integration. Nothing here may slow the agent down or fail it, so reports are
sent from a thread, only the latest one is kept while another is in flight,
and every failure is swallowed.
"""

import os
import shutil
import subprocess
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Mapping, Optional

# Both the integration's identity and the name users see. Herdr reserves the
# "herdr:" prefix for its own integrations.
NAME = "paimon"

IDLE = "idle"
WORKING = "working"
BLOCKED = "blocked"

_TIMEOUT = 2.0
# The release is waited for while the app exits, so it gets less.
_RELEASE_TIMEOUT = 1.0
# Herdr's limits on a resume command. A command over them is rejected along
# with the state report carrying it, so it is dropped here instead.
_MAX_ARGS = 64
_MAX_BYTES = 8 * 1024


@dataclass(frozen=True)
class Report:
    state: str
    session_id: Optional[str] = None
    # The command that reopens the session, or empty when there is nothing to
    # reopen yet.
    resume: tuple[str, ...] = ()


def resumable(argv: tuple[str, ...]) -> bool:
    """Whether Herdr would accept ``argv`` as a resume command, and run it.

    Herdr looks the command up on PATH, which is not where paimon is when it
    was started through ``uv run`` or ``python -m``.
    """
    if not argv or len(argv) > _MAX_ARGS:
        return False
    if sum(len(arg.encode("utf-8")) + 1 for arg in argv) > _MAX_BYTES:
        return False
    if any(ch == "'" or unicodedata.category(ch) == "Cc" for arg in argv for ch in arg):
        return False
    return shutil.which(argv[0]) is not None


class Reporter:
    """Sends this pane's agent state to the Herdr that owns it."""

    def __init__(self, binary: str, pane_id: str) -> None:
        self._binary = binary
        self._pane_id = pane_id
        self._cond = threading.Condition()
        self._latest: Optional[Report] = None
        self._pending: Optional[Report] = None
        self._closed = False
        self._thread: Optional[threading.Thread] = None
        self._seq = 0

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> Optional["Reporter"]:
        """A reporter for the surrounding Herdr pane, or None outside Herdr."""
        env = os.environ if env is None else env
        binary = env.get("HERDR_BIN_PATH")
        pane_id = env.get("HERDR_PANE_ID")
        if env.get("HERDR_ENV") != "1" or not binary or not pane_id or not env.get("HERDR_SOCKET_PATH"):
            return None
        return cls(binary, pane_id)

    def report(self, report: Report) -> None:
        """Queue ``report`` unless it repeats the last one. Never blocks."""
        with self._cond:
            if self._closed or report == self._latest:
                return
            self._latest = self._pending = report
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="herdr-report", daemon=True)
                self._thread.start()
            self._cond.notify()

    def release(self) -> None:
        """Give the pane back. Called once, when the user quits.

        Synchronous, unlike the reports: the process is about to exit, and a
        release still queued behind it would never be sent.
        """
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._pending = None
            self._cond.notify()
        # A report still in flight is not waited for: it carries a lower
        # number than the release, so Herdr drops it if it lands second.
        self._call("release-agent", [], timeout=_RELEASE_TIMEOUT)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                report, self._pending = self._pending, None
            options = ["--state", report.state]
            if report.session_id:
                options += ["--agent-session-id", report.session_id]
            self._call("report-agent", options, report.resume if resumable(report.resume) else ())

    def _next_seq(self) -> int:
        # Herdr drops a report whose number is not above the last one it took
        # from this source, across restarts of the agent too, hence the clock.
        with self._cond:
            self._seq = max(time.time_ns() // 1_000_000, self._seq + 1)
            return self._seq

    def _call(self, command: str, options: list[str], resume: tuple[str, ...] = (), *,
              timeout: float = _TIMEOUT) -> None:
        argv = [self._binary, "pane", command, self._pane_id, "--source", NAME, "--agent", NAME,
                "--seq", str(self._next_seq()), *options]
        if resume:
            argv += ["--", *resume]
        try:
            subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=timeout, check=False)
        except (OSError, subprocess.SubprocessError):
            pass
