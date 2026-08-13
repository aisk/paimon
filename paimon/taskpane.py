"""One background command's pane: its output, and nothing to type into.

The downgrade this deliberately is: no PTY, no keyboard, no terminal
emulation. A tab here is a window onto a pipe — which is why a chatty program
can go quiet for a while (see tools._line_buffered) and why a command that
wants input never gets any.
"""

from rich.text import Text
from textual.app import ComposeResult
from textual.content import Content
from textual.widgets import RichLog, Static

from .pane import Pane

# How often the pane takes what has arrived. Polling rather than a callback
# from the reader: bursts coalesce into one repaint, and nothing in the output
# path ever reaches into a widget.
_POLL_INTERVAL = 0.1

# Lines kept on screen. The command's own buffer is what read_task reads, so
# this bound is about the terminal, not about what the agent can still see.
_MAX_LINES = 5_000


class TaskPane(Pane):
    """A running command, streamed into a tab of its own."""

    def __init__(self, task_id: str, command, description: str, *, cwd, mode: str,
                 id: str | None = None) -> None:
        super().__init__(id=id)
        self.task_id = task_id
        self.command = command
        self.description = description
        # Where the command runs and the mode it was started under. Neither
        # means anything to the pane itself; they are what the app inherits
        # from if this is the last pane left when it closes.
        self.cwd = cwd
        self.mode = mode
        self._cursor = 0
        # Bytes of a line the pipe has not finished handing over. Held back
        # rather than written: a chunk boundary lands wherever it lands, and
        # decoding one that splits a character would print a replacement mark
        # into the middle of the output.
        self._pending = b""
        self._timer = None
        self._exited = False
        # close() runs once: the widget goes away a message loop after the
        # command is killed, and the second call would read a log that is no
        # longer mounted.
        self._pane_closing = False

    @property
    def is_running(self) -> bool:
        return self.command.running and not self.command.killed

    @property
    def tab_title(self) -> str:
        label = " ".join((self.description or self.command.command).split()) or "task"
        return f"{self.task_id} {label}"

    @property
    def status_text(self) -> str:
        """How this task is doing, for the tab and the status bar."""
        if self.command.killed:
            return "stopped"
        if self.command.running:
            return "running"
        return f"exited (code {self.command.exit_code})"

    def compose(self) -> ComposeResult:
        yield RichLog(id="log", wrap=True, markup=False, max_lines=_MAX_LINES,
                      auto_scroll=True)
        yield Static(id="task-status")

    def on_mount(self) -> None:
        self.query_one("#log", RichLog).write(
            Text(f"$ {self.command.command}", style="bold"))
        self._refresh_status()
        self._timer = self.set_interval(_POLL_INTERVAL, self._collect)
        self._focus_input()

    def _focus_input(self) -> None:
        # The log is the only thing to focus, and focusing it is what makes
        # the arrow keys scroll it. Never from a pane the user is not looking
        # at: focusability ignores display, so it would take the keyboard away
        # from whoever is typing.
        if self.is_current:
            self.query_one("#log", RichLog).focus()

    def notice(self, renderable) -> None:
        self.query_one("#log", RichLog).write(renderable)

    def close(self) -> None:
        """Closing the tab is what stopping the command means.

        A process with no window onto it is one the user cannot see, cannot
        stop and will not remember; the supervisor keeps its output readable
        for the agent that started it.
        """
        if self._pane_closing:
            return
        self._pane_closing = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._collect()
        self.command.kill()

    def shutdown(self) -> None:
        # No loop will run after this, so the group is signalled outright.
        self.command.terminate_now()

    def on_show(self) -> None:
        """Catch up on everything that arrived while this tab was hidden."""
        self._collect()
        self._focus_input()

    def _collect(self) -> None:
        """One tick: move new output into the log, and notice the end."""
        if self.display:
            self._drain()
            self._refresh_status()
        if self.is_running or self._exited:
            return
        # Over. Stop polling a buffer nothing writes to any more; anything
        # still undrained is caught by on_show, which is the only way this
        # pane can be looked at again.
        self._exited = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._refresh_status()
        self._notify_state()

    def _drain(self) -> None:
        """Write what has arrived, a whole line at a time.

        Only ever called while the tab is on screen: RichLog defers every
        write until it has a size, and a chatty command in a tab nobody opened
        would pile up renders without bound. The command's own buffer is the
        backlog, and it is the one with a limit.
        """
        data, self._cursor, dropped = self.command.output.since(self._cursor)
        log = self.query_one("#log", RichLog)
        if dropped:
            log.write(Text(f"[{dropped:,} earlier bytes dropped]", style="dim"))
        self._pending += data
        head, newline, self._pending = self._pending.rpartition(b"\n")
        if newline:
            for line in head.decode("utf-8", errors="replace").split("\n"):
                # A carriage return means the program redrew the line in place
                # (a progress bar); the last redraw is what a terminal would
                # be showing.
                log.write(Text.from_ansi(line.rpartition("\r")[2]))
        if self._pending and not self.is_running:
            # Nothing will terminate this line now.
            log.write(Text.from_ansi(self._pending.decode("utf-8", errors="replace")))
            self._pending = b""

    def _refresh_status(self) -> None:
        self.query_one("#task-status", Static).update(
            Content(f"task {self.task_id}  ·  {self.status_text}"))
        if self.is_current:
            self.app.refresh_statusbar()
