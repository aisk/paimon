"""One conversation pane: an agent, its rendered log, and the prompt driving it.

A pane owns everything that belongs to a single session — the agent, the
driver that runs its turns, the queued prompts and the confirmation panel.
``PaimonApp`` owns what is global: the config, the theme, the command palette
and the status bar.
"""

import asyncio
import random
import time
from datetime import datetime

from textual import events, on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, TextArea
from textual.worker import get_current_worker

from . import lockfile
from .agent import Agent
from .events import (
    JobNotice,
    CompactionNotice,
    ContextCompactionFailed,
    ContextCompacted,
    ModelRetry,
    ReasoningDelta,
    RequestStats,
    TodosUpdate,
    ToolEnd,
    ToolStart,
    TurnEnd,
    UserInput,
    replay_events,
)
from .jobs import Job
from .tools.permissions import MODES, confirm_origin
from .turns import CompactionOver, CompactionStarted, Outcome, Result, TurnDriver, TurnOver
from .login import PickerScreen
from .session import Session, SessionError
from .sparkle import Shimmer, Twinkle
from .transcript import EventRenderer, Transcript
from .ui import (
    BlockingPanel,
    ConfirmPanel,
    QuestionPanel,
    PromptInput,
    RecapMessage,
)

# One is picked whenever the spinner enters a new state, Genshin style.
# The spinner only covers stretches with no visible stream: "thinking" while
# reasoning is hidden, "tool" while a tool runs, "waiting" in between.
_STATUS_PHRASES = {
    "waiting": [
        "Counting mora…",
        "Ehe…",
        "Asking the Traveler…",
        "Paimon is NOT emergency food…",
    ],
    "thinking": [
        "Paimon is thinking…",
        "Hmm, let Paimon think…",
        "Paimon will figure this out…",
    ],
    "tool": [
        "Exploring the area ahead…",
        "Wow, treasure…!",
        "Let's go take a look!",
        "Snacking on Sweet Madame…",
    ],
}


def _session_label(session: Session) -> str:
    """'07-24 15:30 · 3f2a8b1c · first user message…' (local time)."""
    when = ""
    created = session.created_at()
    if created:
        try:
            when = datetime.fromisoformat(created).astimezone().strftime("%m-%d %H:%M")
        except ValueError:
            pass
    preview = " ".join((session.first_user_text() or "").split())
    if len(preview) > 40:
        preview = preview[:40] + "…"
    return f"{when} · {session.id[:8]} · {preview}"


class Pane(Vertical):
    """What the app and the tab strip may assume about any pane.

    Three kinds share the strip: a conversation (``SessionPane``), an agent a
    conversation started (``AgentPane``) and a background command
    (``CommandPane``). Everything app-wide — switching,
    closing, the tab label, the status bar — goes through this class, and only
    the session-specific actions look at the concrete type.

    A container, deliberately not a Screen: screens are mutually exclusive and
    ``app.query_one`` only ever sees the active one.
    """

    class StateChanged(Message):
        """This pane started or finished something the strip or the bar shows.

        A conversation's turns run in its TurnDriver and a subagent's in its
        job, neither of which is a widget or posts anything the app would
        hear, so anything app-wide that tracks panes only hears about them
        through this.
        """

        def __init__(self, pane: "Pane") -> None:
            self.pane = pane
            super().__init__()

    # The defaults are the quiet ones, so a kind of pane only has to state
    # what is true of it. Subclasses carry two attributes besides: ``cwd`` and
    # ``mode``, which is what a pane opened in place of this one inherits.
    needs_confirm = False
    # The conversation whose work this pane shows: itself for a conversation,
    # the one that started it for an agent or a command. What the app's
    # global actions act on while this pane is on screen.
    owner: "SessionPane | None" = None
    # The panel that is waiting, while ``needs_confirm`` is true.
    asking: BlockingPanel | None = None
    is_busy = False
    is_running = False
    _last_focus: Widget | None = None

    @property
    def is_current(self) -> bool:
        """Whether this is the pane the user is looking at."""
        return self.app.pane is self

    @property
    def tab_title(self) -> str:
        """Short label for the tab strip."""
        raise NotImplementedError

    def _notify_state(self) -> None:
        self.post_message(self.StateChanged(self))

    def notice(self, renderable) -> None:
        """Show a line of app news in this pane's log."""
        raise NotImplementedError

    def _focus_input(self) -> None:
        """Focus whatever this pane is waiting on, if it is the one on screen."""

    def _request_focus(self, widget: Widget) -> None:
        """All automatic pane focus goes through the app's ownership checks."""
        self.app.focus_pane_widget(self, widget)

    def focus_attention(self) -> None:
        """Focus what is waiting on the user here, if anything is."""
        self._focus_input()

    def close(self) -> None:
        """Give up everything this pane owns; the app removes the widget."""

    def shutdown(self) -> None:
        """The app is exiting. Release what the OS would not release for us."""

    def on_key(self, event: events.Key) -> None:
        """Panes that take typing claim keys here; the rest let them pass."""


class SessionPane(Pane):
    """A single conversation: an agent, its rendered log and its prompt."""

    def __init__(self, agent: Agent, *, resumed: bool = False, id: str | None = None) -> None:
        super().__init__(id=id)
        # One confirmation or question on screen at a time: a second panel
        # mounting over the first would leave the first one's answer waiting
        # forever.
        self._panel_lock = asyncio.Lock()
        # Made here rather than in compose: _adopt points a renderer at it.
        self.transcript = Transcript(id="log")
        # Set before _adopt, which cancels whatever recap the pane had armed.
        self._recap_timer = None
        # Whether the turn now running has called a tool. A turn that only
        # answered needs no recap: the answer is right there on the screen.
        self._used_tools = False
        self._adopt(agent)
        self._resumed = resumed
        # Tab label, kept here rather than read back from the session file on
        # every repaint of the strip.
        self._title = agent.session.first_user_text() or ""
        # Last context size measured for this session, so redrawing the status
        # bar for an unrelated reason does not blank the readout.
        self._tokens: int | None = None
        # Set by close(): the driver is cancelled and the widgets go away, so
        # nothing it unwinds through must touch the DOM. Not named _closing:
        # MessagePump already owns that attribute, and setting it strands the
        # widget's message loop on teardown.
        self._pane_closing = False
        # The spinner: what the turn is doing, and since when.
        self._status_state: str | None = None
        self._phrase = ""
        self._turn_started = 0.0
        self._status_timer = None
        self._queue: list[str] = []
        self._shell_worker = None
        self._tps: float | None = None
        # Cumulative over the session: single-request rates swing with each
        # round's tool output size, while the running rate reflects what the
        # cache actually saved and makes server-side invalidation visible as
        # a steady decline instead of one noisy dip.
        self._cache_hit: float | None = None
        self._cache_reads = 0
        self._cache_inputs = 0
        self._cache_reported = False

    @property
    def owner(self) -> "SessionPane":
        return self

    @property
    def mode(self) -> str:
        """The permission mode. The agent's, which is what its tools are
        checked against and what the agents it started follow."""
        return self.agent.mode

    @property
    def config(self):
        """The active config. Process-wide, kept in sync with the app's."""
        return self.agent.config

    @property
    def cwd(self):
        """Where this conversation works. Panes opened from it inherit it."""
        return self.agent.cwd

    @property
    def is_running(self) -> bool:
        """Whether a turn is streaming right now. For display only."""
        return self.driver.is_running

    @property
    def is_busy(self) -> bool:
        """Whether a turn or a compaction is running, or one is already
        queued behind it.

        What every guard asks. Taken from the driver rather than from a worker's
        status: the driver accepts a prompt the instant it is submitted, so
        there is no window in which a second turn can be started by the user
        and a wake-up at the same moment.
        """
        return self.driver.is_busy

    @property
    def needs_confirm(self) -> bool:
        """Whether a confirmation or question here is waiting on the user."""
        return self.driver.blocked > 0

    @property
    def tab_title(self) -> str:
        """Short label for the tab strip."""
        return " ".join(self._title.split()) or "new session"

    def notice(self, renderable) -> None:
        self.transcript.add(renderable)

    def close(self) -> None:
        """Give up everything this pane owns; the app removes the widget.

        Idempotent: the widget goes away one message loop after the agent is
        killed, and unlocking a session twice would drop a claim somebody else
        has since taken.
        """
        if self._pane_closing:
            return
        self._pane_closing = True
        # Nothing more may reach the widgets: they go one message loop from
        # now, and mounting into them after that raises.
        self.driver.sink = None
        self._cancel_recap()
        self._retire_agent()

    def shutdown(self) -> None:
        """Release the session on the way out.

        The lock would go with the process anyway, but only after every pane's
        turn has been cancelled by the shutdown; doing it here keeps "a pane
        holds its session for exactly as long as it is open" true even when the
        app is torn down and rebuilt inside one process.
        """
        if not self._pane_closing:
            self._pane_closing = True
            self.driver.sink = None
            self._cancel_recap()
            self.driver.shutdown()
            self.agent.close()

    def compose(self) -> ComposeResult:
        yield self.transcript
        status = Horizontal(
            Twinkle(classes="status-spinner"),
            Shimmer(classes="status-label"),
            id="response-status",
        )
        status.display = False
        yield status
        queued = Static(id="queued")
        queued.display = False
        yield queued
        prompt = PromptInput(id="prompt", soft_wrap=True, placeholder=PromptInput.PLACEHOLDER)
        prompt.border_subtitle = PromptInput.INPUT_HELP
        yield prompt

    async def on_mount(self) -> None:
        # The first driver is built in __init__, before there is a loop to
        # run it; every later one is started by _swap_agent as it is made.
        self.driver.start()
        self._focus_input()
        self._refresh_mode()
        if self._resumed:
            await self._show_resumed()
        else:
            self.transcript.show_welcome(self.cwd)
        diagnostics = self.agent.skill_diagnostics
        if diagnostics:
            first = diagnostics[0]
            more = f" (+{len(diagnostics) - 1} more)" if len(diagnostics) > 1 else ""
            self.notify(f"{first.path}: {first.message}{more}", title="Skill warnings",
                        severity="warning")

    def on_key(self, event: events.Key) -> None:
        """Route stray typing back into the prompt.

        Clicking the log (say, to expand a folded result) focuses the scroll
        container and keystrokes would silently vanish; any printable key that
        bubbles up unclaimed refocuses the prompt and lands in it. Modal
        screens keep the keyboard to themselves, and so does a panel that has
        taken the prompt's place.
        """
        if not self.is_mounted or not event.is_printable or len(self.app.screen_stack) > 1:
            return
        prompt = self.query_one(PromptInput)
        if not prompt.display or isinstance(self.app.focused, BlockingPanel):
            return
        if self.app.focused is not prompt:
            self._request_focus(prompt)
            if self.app.focused is prompt:
                prompt.insert(event.character)
                event.stop()

    # ---- focus --------------------------------------------------------------

    def _focus_input(self) -> None:
        """Focus what this pane is waiting on, unless another pane is on screen.

        A panel that has taken the prompt's place wins over it: the prompt is
        hidden underneath, and switching to a pane to answer it has to land on
        the panel or the keys go nowhere.

        Widget.focusable only looks at ``visible``, which is unrelated to
        ``display``, so a hidden pane focusing anything really does take the
        keyboard away from the pane the user is typing into.
        """
        if not self.is_current:
            return
        prompt = self.query_one(PromptInput)
        panels = self.query(BlockingPanel)
        self._request_focus(prompt if prompt.display or not panels else panels.last())

    # ---- session switching --------------------------------------------------

    def _adopt(self, agent: Agent) -> None:
        """Take over an agent: this pane confirms and answers for it and renders its turns."""
        self._cancel_recap()
        self.agent = agent
        agent.confirm = self._confirm
        agent.ask = self._ask
        agent.jobs.open_job = self._open_job
        # One renderer per conversation rather than one per turn: a turn now
        # opens with a UserInput event, which is what resets it. A new one per
        # agent, so a swapped-out session cannot leave a live markdown stream
        # pointing at a log that has just been emptied.
        self._renderer = EventRenderer(self.transcript, self.agent)
        # The driver takes the agent's history hooks (queued input, job
        # moves) and decides when the agent wakes; the pane is told after.
        self.driver = TurnDriver(agent)
        self.driver.sink = self._on_event
        self.driver.on_change = self._on_change
        self.driver.take_queued = self._take_queued
        self.driver.on_jobs_changed = self._jobs_changed

    def _swap_agent(self, agent: Agent) -> None:
        """Put a different conversation in this pane, under a driver of its own."""
        self._adopt(agent)
        self.driver.start()

    def _retire_agent(self) -> list[str]:
        """Release the current session and stop the jobs it started.

        Their ids only mean anything inside the conversation being left
        behind, so nothing would ever hear from them again. Returns the ids
        of the ones still running, for the caller to report.
        """
        killed = [job_id for job_id, job in self.agent.jobs.items() if job.running]
        # A "!" command outlives neither the session it was run in nor the
        # pane: it holds a process tree, and nothing left would read its
        # output.
        worker, self._shell_worker = self._shell_worker, None
        if worker is not None:
            worker.cancel()
        self.driver.cancel()
        self.agent.close()
        return killed

    def _report_killed(self, killed: list[str]) -> None:
        if killed:
            self.transcript.add(Content.from_markup(
                "[$text-muted]Stopped $n job(s) started by the previous session: $ids[/]",
                n=str(len(killed)), ids=", ".join(killed)))

    async def _show_resumed(self) -> None:
        renderer = EventRenderer(self.transcript, self.agent)
        events = replay_events(self.agent.history)
        for ev in events:
            await renderer.handle(ev)
        await renderer.close()
        self.transcript.add(Content.from_markup("[$text-muted]Resumed session $id[/]", id=self.agent.session.id[:8]))
        self._sync_statusbar(tokens=True)
        # Coming back to a session is when a recap is wanted most, so this one
        # does not wait for the idle timer. Same bar as after a turn: there
        # has to be work to recap, not just an answer already on the screen.
        if (self.config.recap_enabled and self.is_current
                and any(isinstance(ev, (ToolStart, CompactionNotice)) for ev in events)):
            self._recap()

    def _reset_measurements(self) -> None:
        """A swapped-in session starts with no measurements of its own."""
        self._tokens = None
        self._tps = None
        self._cache_hit = None
        self._cache_reads = 0
        self._cache_inputs = 0
        self._cache_reported = False

    def _open_agent(self, session: Session | None = None) -> Agent:
        """An agent to put in place of this pane's, on ``session`` or a new one.

        Opened the way the app opens every conversation (see
        PaimonApp.open_agent), where this one works and in its mode. The
        hooks are _adopt's to set.
        """
        return self.app.open_agent(self.cwd, self.mode, session)

    def new_session(self) -> None:
        if self.is_busy:
            return
        try:
            agent = self._open_agent()
        except SessionError as exc:
            self.transcript.add(Content.from_markup("[$text-error b]Cannot start a new session:[/] $body",
                                                    body=str(exc)))
            return
        killed = self._retire_agent()
        self._swap_agent(agent)
        self._title = ""
        self._reset_measurements()
        self.transcript.clear()
        self._queue.clear()
        self._refresh_queued()
        self.transcript.show_welcome(self.cwd)
        self.transcript.add(Content.from_markup("[$text-muted]Started new session $id[/]", id=self.agent.session.id[:8]))
        self._report_killed(killed)
        self._sync_statusbar()
        self._notify_state()

    def fork_session(self) -> None:
        if self.is_busy:
            return
        try:
            # Copying the log is file I/O of its own and can fail (disk full,
            # permissions) before any agent exists to open.
            forked = self.agent.session.fork()
            agent = self._open_agent(forked)
        except (SessionError, OSError) as exc:
            self.transcript.add(Content.from_markup("[$text-error b]Cannot fork:[/] $body", body=str(exc)))
            return
        # The conversation on screen is the fork's history verbatim, so the
        # log stays; only the agent underneath changes.
        agent.todos = list(self.agent.todos)
        killed = self._retire_agent()
        self._swap_agent(agent)
        self.transcript.add(Content.from_markup("[$text-muted]Forked session $id[/]", id=agent.session.id[:8]))
        self._report_killed(killed)
        self._sync_statusbar()
        self._notify_state()

    # The picker gets its own group so that no exclusive worker can cancel it
    # while it is open. Turns run in the TurnDriver now, not as workers here,
    # and the exclusive workers left on this node (the recap) have groups of
    # their own, so this is a guard against one ever being started in the
    # default group rather than a fix for one that is.
    @work(group="picker")
    async def resume_session(self) -> None:
        if self.is_busy:
            return
        # A session open in another pane is left out: two agents on one log
        # would interleave their turns into it. The in-process lock cannot
        # refuse them — it refcounts — so the picker has to.
        labels = {_session_label(session): session
                  for session in Session.list(self.agent.cwd)
                  if not lockfile.held(session.path)}
        if not labels:
            self.transcript.add(Content.from_markup("[$text-muted]No sessions to resume in this directory[/]"))
            return
        choice = await self.app.push_screen_wait(PickerScreen("Resume session", list(labels)))
        if choice in labels and self.is_busy:
            # A wake-up (a job reporting) or a compaction started while the
            # picker was up; swapping the session now would cut it off.
            self.transcript.add(Content.from_markup(
                "[$text-muted]Busy — resume a session after this turn[/]"))
        if choice not in labels or self.is_busy:
            self._focus_input()
            return
        try:
            agent = self._open_agent(labels[choice])
        except SessionError as exc:  # already open elsewhere, or no persisted system prompt
            self.transcript.add(Content.from_markup("[$text-error b]Cannot resume:[/] $body", body=str(exc)))
            return
        killed = self._retire_agent()
        self._swap_agent(agent)
        self._title = agent.session.first_user_text() or ""
        self._reset_measurements()
        self.transcript.clear()
        self._queue.clear()
        self._refresh_queued()
        await self._show_resumed()
        self._report_killed(killed)
        self._sync_statusbar()
        self._notify_state()
        self._focus_input()

    def compact(self) -> None:
        """Compact the context, as an item of the driver's like a turn.

        Through the driver so that nothing else writes the history while the
        summary request is out: a prompt typed meanwhile queues behind it, and
        Esc stops it. What it shows comes back through the sink.
        """
        if not self.driver.compact():
            self.transcript.add(Content.from_markup("[$text-muted]Busy — compact the context after this turn[/]"))

    # ---- permission mode ----------------------------------------------------

    def cycle_mode(self) -> None:
        self.agent.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        self._refresh_mode()
        self._sync_statusbar()
        self._notify_state()

    def _refresh_mode(self) -> None:
        prompt = self.query_one(PromptInput)
        prompt.border_title = f" {self.mode} "
        # The title is drawn as a chip in the mode's own colour.
        for mode in MODES:
            prompt.set_class(mode == self.mode, f"-{mode}")

    # ---- status --------------------------------------------------------------

    def _set_status(self, visible: bool, label: str = "") -> None:
        status = self.query_one("#response-status", Horizontal)
        status.display = visible
        if visible and label:
            status.query_one(".status-label", Static).update(label)

    def _sync_statusbar(self, *, tokens: bool = False) -> None:
        """Redraw the app-wide status bar, which only ever shows this pane's
        session while it is the one on screen."""
        if not self.is_current:
            return
        if tokens:
            self.app.update_statusbar_tokens()
        else:
            self.app.refresh_statusbar()

    # ---- confirmation and question hooks (called from the agent loop) ------

    async def _confirm(self, tool_name: str, args: dict) -> bool:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        # A call from a subagent running unseen lands here too, marked with
        # where it came from so the panel can say so.
        panel = ConfirmPanel(tool_name, args, future, cwd=self.agent.cwd,
                             origin=confirm_origin())
        return await self._block_on(panel, future) == "allow"

    async def _ask(self, question: str, options: list[str]) -> str | None:
        future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        return await self._block_on(QuestionPanel(question, options, future), future)

    async def _block_on(self, panel: BlockingPanel, future: asyncio.Future):
        """Show ``panel`` in place of the prompt until ``future`` is answered."""
        async with self._panel_lock:
            # Held across the wait: the session can be swapped while a panel is
            # up, and the count taken from one driver must go back to that one.
            driver = self.driver
            prompt = self.query_one(PromptInput)
            # Removal below is asynchronous, so a panel from the previous
            # confirm (or an interrupted turn) may still be mounted; sweep it
            # first. Under the lock nothing live can be among them. The query
            # is pane-scoped: another pane's pending panel is not ours to
            # remove.
            await self.query(BlockingPanel).remove()
            await self.mount(panel, before=prompt)
            focused = self.app.focused
            prompt.display = False
            # Replace input ownership, not deliberate reading focus. Background
            # panes and modals are additionally protected by the central guard.
            if focused is prompt or focused is None:
                self._request_focus(panel)
            # Counted on the driver rather than here: removing the panel is
            # asynchronous, and the tab badge has to clear the moment the
            # answer is in, not whenever the widget finally goes.
            self.asking = panel
            driver.mark_blocked(True)
            try:
                return await future
            finally:
                self.asking = None
                driver.mark_blocked(False)
                focused = self.app.focused if self.app.screen_stack else None
                restore = focused is panel or (focused is not None and panel in focused.ancestors)
                prompt.display = True
                if restore and not self._pane_closing:
                    # Transfer before removal resets Textual's focus chain;
                    # do not restore after an await where the user can move it.
                    self._request_focus(prompt)
                await panel.remove()

    # ---- input → turn -------------------------------------------------------

    @on(PromptInput.Submitted)
    def handle_submit(self, event: PromptInput.Submitted) -> None:
        text = event.text
        self._cancel_recap()
        self.query_one(PromptInput).clear()
        # "!" runs the rest of the line in a shell instead of sending it, and
        # "!!" does the same without telling the model. Handled before the
        # busy check: a command is the user's own, so it runs while a turn is
        # in flight rather than queueing behind it.
        if text.startswith("!"):
            quiet = text.startswith("!!")
            command = text[2:].strip() if quiet else text[1:].strip()
            if command:
                self.run_user_command(command, record=not quiet)
            return
        if self.is_busy:
            self._queue.append(text)
            self._refresh_queued()
            return
        self.driver.submit(text)

    def run_user_command(self, command: str, *, record: bool = True) -> None:
        """Run a "!" command, log it, and let the model in on it unless quiet."""
        if self._shell_worker is not None:
            self.transcript.add(Content.from_markup(
                "[$text-muted]A command is already running — Esc cancels it[/]"))
            return
        self._shell_worker = self._user_command(command, record)

    @work(group="user-command")
    async def _user_command(self, command: str, record: bool) -> None:
        """One "!" run: the command line goes up first, the output when it ends.

        Not a turn, so it neither waits for one nor blocks one. Where the
        result lands is the driver's call (see TurnDriver.record_shell), since
        it owns everything that writes the history.
        """
        # Logged without a pause before the try: Esc can come at any await,
        # the first one included, and has to be answered in the log.
        step = self.transcript.add_shell_call(command)
        worker = get_current_worker()
        try:
            output = await self.agent.run_shell(command)
        except asyncio.CancelledError:
            # The process tree is already dead (the tool kills it on the way
            # out); what it managed to print went with it, so there is nothing
            # worth telling the model about.
            if not self._pane_closing:
                self.transcript.add_tool_result("(interrupted)", label=command, container=step)
            raise
        finally:
            # Only if it is still this run's: whoever cancelled it let go of
            # it already, and a run started since, while this one's process
            # tree was being killed, is the one the reference now belongs to.
            if self._shell_worker is worker:
                self._shell_worker = None
        if self._pane_closing:
            return
        self.transcript.add_tool_result(output, label=command, container=step)
        if record:
            self.driver.record_shell(command, output)

    def _refresh_queued(self) -> None:
        widget = self.query_one("#queued", Static)
        widget.display = bool(self._queue)
        if not self._queue:
            widget.update("")
            return
        kwargs = {f"q{i}": text for i, text in enumerate(self._queue)}
        lines = "\n".join(f"[$text-muted]⏸ ${key}[/]" for key in kwargs)
        widget.update(Content.from_markup(lines, **kwargs))

    def _take_queued(self) -> list[str]:
        """Hand the queued messages to the running turn, which injects them.

        Called from the agent loop through the driver, on this app's event
        loop, so touching the panel from here is safe.
        """
        if self._pane_closing or not self._queue:
            return []
        texts, self._queue = self._queue, []
        self._refresh_queued()
        return texts

    def _release_queue(self, *, send: bool) -> None:
        """Deal with what was typed while the driver was busy: send it as the
        next turn, or hand it back to the input box."""
        if not self._queue:
            return
        text = "\n\n".join(self._queue)
        self._queue.clear()
        self._refresh_queued()
        if send:
            self.driver.submit(text)
            return
        prompt_input = self.query_one(PromptInput)
        draft = prompt_input.text
        prompt_input.load_text(f"{text}\n{draft}" if draft else text)
        prompt_input.move_cursor(prompt_input.document.end)

    def interrupt(self) -> None:
        # Esc stops whatever this pane is doing on its own, a recap included.
        self._cancel_recap()
        # A "!" command runs beside the turn rather than inside it, so Esc has
        # two things to stop. The command goes first and alone: it is the more
        # immediate one, a second Esc still reaches the turn, and a process
        # tree nobody is waiting on cannot be reached any other way.
        worker, self._shell_worker = self._shell_worker, None
        if worker is not None:
            worker.cancel()
            return
        self.driver.interrupt()

    # ---- idle recap ---------------------------------------------------------

    @on(TextArea.Changed, "#prompt")
    def _typing_defers_recap(self, event: TextArea.Changed) -> None:
        """Somebody is at the keyboard, so they have not gone anywhere yet."""
        # Only while one is armed: at any other time typing means nothing here,
        # and this runs on every keystroke.
        if self._recap_timer is not None:
            self._arm_recap()

    def _arm_recap(self) -> None:
        """Offer a recap if this pane is still the one being looked at when
        the idle wait runs out."""
        self._cancel_recap()
        if not (self.config.recap_enabled and self._used_tools
                and self.is_current and not self.is_busy):
            return
        self._recap_timer = self.set_timer(self.config.recap_idle_seconds, self._recap_due)

    def _recap_due(self) -> None:
        # The timer's own callback, so the reference it clears is the timer
        # that fired: cleared from inside the worker it starts, it would be
        # whichever timer had been armed by the time the worker ran, which
        # nothing could then cancel.
        self._recap_timer = None
        self._recap()

    def _cancel_recap(self) -> None:
        """Drop a pending recap, and any request already out for one."""
        if self._recap_timer is not None:
            self._recap_timer.stop()
            self._recap_timer = None
        # Not before the pane is mounted: _adopt runs from __init__, where
        # there is no app to hold a worker yet.
        if self.is_mounted:
            self.workers.cancel_group(self, "recap")

    @work(exclusive=True, group="recap")
    async def _recap(self) -> None:
        """Ask what the session looks like now, and show it under the log.

        Read-only (see Agent.recap): the recap is never part of the
        conversation, so a resumed session does not replay it.
        """
        # The history is only a complete request between turns. And only for
        # the pane on screen, asked before the request rather than after it:
        # a recap nobody will see is a request nobody should pay for.
        if self.is_busy or not self.is_current:
            return
        try:
            text = await self.agent.recap()
        except Exception:  # noqa: BLE001 — nobody asked for this, so nobody hears about it
            return
        if self._pane_closing or self.is_busy or not self.is_current or not text:
            return
        self.transcript.mount(RecapMessage(text))

    # ---- the agent's job hooks ----------------------------------------------

    def _jobs_changed(self) -> None:
        """An agent or command this conversation started moved. Whether the
        agent wakes for it is the driver's business; this only redraws."""
        if self._pane_closing or not self.is_mounted:
            return
        self._sync_statusbar()

    async def _open_job(self, job_id: str, job: Job) -> None:
        await self.app.open_job(self, job_id, job)

    # ---- the driver's two hooks ---------------------------------------------

    def _on_change(self) -> None:
        """This pane's driver moved. Called from its loop and from cancellation.

        Only a notification: everything that has to happen in order with the
        turn's own output belongs in the sink below, which is the one place
        guaranteed to run after the last event of a turn.
        """
        if self._pane_closing or not self.is_mounted:
            return
        self._notify_state()

    async def _on_event(self, ev) -> None:
        """Render one event of this pane's turns. The driver's sink.

        Awaited by the driver, so the renderer's mounts are what paces a turn
        and nothing has to be buffered on the way here.
        """
        if self._pane_closing or not self.app.is_running:
            return
        if isinstance(ev, TurnOver):
            await self._end_turn(ev.result)
            return
        if isinstance(ev, CompactionStarted):
            self._cancel_recap()
            self._set_status(True, " Compacting context")
            return
        if isinstance(ev, CompactionOver):
            self._end_compaction(ev)
            return
        if isinstance(ev, UserInput):
            self._begin_turn(ev.text)
        elif isinstance(ev, JobNotice):
            # A wake-up turn opens with a job notice instead of a user
            # message; in a normal turn the UserInput already ran this and the
            # timer marker makes it a no-op. Empty text keeps the tab title.
            self._begin_turn("")
        await self._renderer.handle(ev)
        if isinstance(ev, TurnEnd):
            self._set_state(None)
            self._sync_statusbar(tokens=True)
        elif isinstance(ev, ToolStart):
            self._used_tools = True
            self._set_state("tool")
        elif isinstance(ev, (ToolEnd, TodosUpdate, ContextCompacted,
                             ContextCompactionFailed, ModelRetry)):
            # the model is about to react to what just happened
            self._set_state("waiting")
        elif isinstance(ev, ReasoningDelta):
            # folded reasoning only ticks a line count, so the spinner stands in
            self._set_state(None if self.config.show_reasoning else "thinking")
        elif isinstance(ev, RequestStats):
            self._tps = ev.output_tokens / ev.seconds
            # A request that missed the cache entirely still counts against
            # the rate. Only a session with no cache tokens at all, from a
            # provider that reports none, shows no rate instead of 0%.
            self._cache_reads += ev.cache_read_tokens
            self._cache_inputs += ev.input_tokens
            self._cache_reported = self._cache_reported or bool(
                ev.cache_read_tokens or ev.cache_write_tokens)
            if self._cache_reported and self._cache_inputs:
                self._cache_hit = self._cache_reads / self._cache_inputs
            self._sync_statusbar(tokens=True)
        elif not isinstance(ev, UserInput):
            self._set_state(None)

    def _begin_turn(self, text: str) -> None:
        if not self._title:
            self._title = text
        self._cancel_recap()
        self._status_state = None
        self._set_state("waiting")
        if self._status_timer is None:
            # A message injected mid-turn arrives as a UserInput event too, and
            # must not restart the clock of the turn it landed in. The timer is
            # the marker: it only clears when a turn ends.
            self._turn_started = time.monotonic()
            self._used_tools = False
            self._status_timer = self.set_interval(1, self._tick_status)

    async def _end_turn(self, result: Result) -> None:
        """Close out a turn, however it ended, and start whatever waits behind it.

        Queued messages were typed against a turn that then stopped or failed,
        so they go back to the input rather than at a model the user just
        interrupted. That is a decision about a text box, which is why it lives
        here and not in the driver's inbox. Waking the agent for a job that
        reported meanwhile is the driver's, once this returns.
        """
        await self._renderer.close()
        self._set_state(None)
        if self._status_timer is not None:
            self._status_timer.stop()
            self._status_timer = None
        if result.outcome is Outcome.INTERRUPTED:
            self.transcript.add(Content.from_markup("[$text-warning]⏹ Paimon stopped![/]"))
        elif result.outcome is Outcome.FAILED:
            self.transcript.add(Content.from_markup("[$text-error b]Error:[/] $body", body=result.error))
        # Finishing a turn must not steal focus from reading or a modal.
        self._release_queue(send=result.finished)
        # Last, so a turn started just above (a queued message) is
        # already busy and arms nothing. Only for a turn that ran to its own
        # end: recapping work the user just stopped is not what they asked for.
        if result.finished:
            self._arm_recap()

    def _end_compaction(self, over: CompactionOver) -> None:
        """Report how a compaction went, and send what was typed meanwhile.

        Prompts queued behind it were typed against the conversation, not
        against the compaction, so they go out after a failed one too; only
        Esc, which stops everything this pane is doing, hands them back.
        """
        self._set_status(False)
        result, compacted = over.result, over.compacted
        if result.outcome is Outcome.INTERRUPTED:
            self.transcript.add(Content.from_markup("[$text-warning]⏹ Compaction stopped[/]"))
        elif result.outcome is Outcome.FAILED:
            self.transcript.add(Content.from_markup("[$text-error b]Compaction failed:[/] $body",
                                                    body=result.error))
        elif compacted is None:
            self.transcript.add(Content.from_markup(
                "[$text-muted]Nothing to compact yet — the context is still short[/]"))
        else:
            self.transcript.add(Content.from_markup(
                "[$text-muted]Context compacted: $before → ~$after tokens[/]",
                before=f"{compacted.tokens_before:,}",
                after=f"{compacted.tokens_after:,}",
            ))
            self._sync_statusbar(tokens=True)
        self._release_queue(send=result.outcome is not Outcome.INTERRUPTED)

    # ---- the spinner --------------------------------------------------------

    def _set_state(self, new: str | None) -> None:
        """Show (or hide) the status line for what the turn is doing now.

        The phrase is re-rolled only on state changes so it does not flicker
        through the pool while one state lasts.
        """
        if new == self._status_state:
            return
        self._status_state = new
        if new is None:
            self._set_status(False)
        else:
            self._phrase = random.choice(_STATUS_PHRASES[new])
            self._set_status(True, self._status_label())

    def _status_label(self) -> str:
        elapsed = int(time.monotonic() - self._turn_started)
        return f" {self._phrase} {elapsed}s" if elapsed else f" {self._phrase}"

    def _tick_status(self) -> None:
        if self._status_state is not None:
            self._set_status(True, self._status_label())
