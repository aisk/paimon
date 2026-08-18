"""One conversation pane: an agent, its rendered log, and the prompt driving it.

A pane owns everything that belongs to a single session — the agent, the turn
worker, the queued prompts, the permission mode and the confirmation panel.
``PaimonApp`` owns what is global: the config, the theme, the command palette
and the status bar.
"""

import asyncio
import random
import time
from datetime import datetime
from uuid import uuid4

from textual import events, on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.widget import Widget
from textual.widgets import LoadingIndicator, Static, TextArea
from textual.widgets.markdown import MarkdownStream

from . import aside, lockfile, tools
from .agent import (
    Agent,
    AgentsNotice,
    CompactionNotice,
    ContextCompactionFailed,
    ContextCompacted,
    ModelRetry,
    ReasoningDelta,
    RequestStats,
    SessionHandoff,
    TextDelta,
    TodosUpdate,
    ToolEnd,
    ToolStart,
    TurnEnd,
    UserInput,
    replay_events,
)
from .diff import locate_line
from .jobs import AgentJob, Outcome, Result, State, TurnOver
from .login import PickerScreen
from .session import Session, SessionError, resume_hint
from .ui import (
    AssistantMessage,
    ConfirmPanel,
    EditCall,
    FoldedText,
    PromptInput,
    RecapMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)

# All three markers are East Asian Width "narrow", so the labels stay aligned on
# terminals that render ambiguous-width glyphs double-wide. Finished work is
# struck through and dimmed to keep the accent on whatever is in progress.
_TODO_STYLE = {
    "completed": ("✓", "$text-disabled strike"),
    "in_progress": ("▸", "$text-accent b"),
    "pending": ("◦", "$text-muted"),
}

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


class _EventRenderer:
    """Renders agent events into a pane's log.

    The single rendering path: live turns and resumed-history replay both feed
    events through ``handle``, so history always looks like it did live.
    """

    def __init__(self, pane: "SessionPane") -> None:
        self._pane = pane
        self._stream: MarkdownStream | None = None
        self._reasoning: FoldedText | None = None
        self._reasoning_buf = ""
        self._first_text_block = True
        self._call_labels: dict[str, str] = {}

    async def handle(self, ev: object) -> None:
        if isinstance(ev, UserInput):
            await self.close()
            self._first_text_block = True
            self._pane._add_user(ev.text)

        elif isinstance(ev, CompactionNotice):
            await self.close()
            self._first_text_block = True
            self._pane._add(Content.from_markup("[$text-muted]Earlier context was compacted[/]"))

        elif isinstance(ev, AgentsNotice):
            await self.close()
            self._first_text_block = True
            self._pane._add(Content.from_markup("[$text-muted]Agents: $text[/]", text=ev.text))

        elif isinstance(ev, ReasoningDelta):
            self._reasoning_buf += ev.text
            if self._reasoning is None:
                self._reasoning = FoldedText(
                    "",
                    classes="reasoning",
                    expanded=self._pane.config.show_reasoning,
                    label="reasoning",
                )
                await self._pane.query_one("#log", VerticalScroll).mount(self._reasoning)
            self._reasoning.set_text(self._reasoning_buf)

        elif isinstance(ev, TextDelta):
            if self._stream is None:
                widget = AssistantMessage("", heading=self._first_text_block)
                self._first_text_block = False
                # Await the mount so the initial document (the Paimon heading)
                # is rendered before the stream appends to it.
                await self._pane.query_one("#log", VerticalScroll).mount(widget)
                self._stream = AssistantMessage.get_stream(widget)
            await self._stream.write(ev.text)

        elif isinstance(ev, ToolStart):
            # start fresh assistant/reasoning blocks after a tool runs
            await self.close()
            self._call_labels[ev.id] = (
                f"{ev.name} {tools.summarize_call(ev.name, ev.args, limit=40)}"
            )
            self._pane._add_tool_start(ev.name, ev.args)

        elif isinstance(ev, TodosUpdate):
            await self.close()
            self._pane._show_todos(ev.todos)

        elif isinstance(ev, ToolEnd):
            label = self._call_labels.pop(ev.id, ev.name)
            self._pane._add_tool_result(ev.result, label=label, denied=ev.denied)

        elif isinstance(ev, ContextCompacted):
            self._pane._add(
                Content.from_markup(
                    "[$text-muted]Context compacted: $before → ~$after tokens[/]",
                    before=f"{ev.tokens_before:,}",
                    after=f"{ev.tokens_after:,}",
                )
            )

        elif isinstance(ev, ContextCompactionFailed):
            self._pane._add(
                Content.from_markup(
                    "[$text-warning]Context compaction failed; continuing without it: $error[/]",
                    error=ev.error,
                )
            )

        elif isinstance(ev, ModelRetry):
            self._pane._add(
                Content.from_markup(
                    "[$text-warning]$error — retrying in $delay s ($attempt/$total)[/]",
                    error=ev.error,
                    delay=f"{ev.delay:g}",
                    attempt=str(ev.attempt),
                    total=str(ev.max_attempts - 1),
                )
            )

    async def close(self) -> None:
        """End the current text/reasoning blocks so the next output starts fresh."""
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None
        if self._reasoning is not None and self._pane.config.show_reasoning:
            # fold the live stream now that the block is over; blocks the user
            # clicked open themselves are left alone
            self._reasoning.collapse()
        self._reasoning = None
        self._reasoning_buf = ""


class Pane(Vertical):
    """What the app and the tab strip may assume about any pane.

    Two kinds share the strip: a conversation (``SessionPane``) and a
    background command (``TaskPane``). Everything app-wide — switching,
    closing, the tab label, the status bar — goes through this class, and only
    the session-specific actions look at the concrete type.

    A container, deliberately not a Screen: screens are mutually exclusive and
    ``app.query_one`` only ever sees the active one.
    """

    class StateChanged(Message):
        """This pane started or finished something the strip or the bar shows.

        Worker.StateChanged does not bubble and the turn worker lives on the
        pane, so anything app-wide that tracks panes only hears about them
        through this.
        """

        def __init__(self, pane: "Pane") -> None:
            self.pane = pane
            super().__init__()

    # The defaults are the quiet ones, so a kind of pane only has to state
    # what is true of it. Subclasses carry three attributes besides: ``job``,
    # the thing this pane is a window onto, and ``cwd`` and ``mode``, which is
    # what a pane opened in place of this one inherits.
    needs_confirm = False
    is_busy = False
    is_running = False

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

    def close(self) -> None:
        """Give up everything this pane owns; the app removes the widget."""

    def shutdown(self) -> None:
        """The app is exiting. Release what the OS would not release for us."""

    def on_key(self, event: events.Key) -> None:
        """Panes that take typing claim keys here; the rest let them pass."""


class SessionPane(Pane):
    """A single conversation: an agent, its rendered log and its prompt."""

    def __init__(self, agent: Agent, *, job_id: str, parent=None, resumed: bool = False,
                 id: str | None = None, supervisor=None) -> None:
        super().__init__(id=id)
        # The pool this pane's agent can start and talk to other agents through.
        self.supervisor = supervisor
        # Whose subagent this conversation is, or None when the user opened
        # it. It is what decides who may read and stop this pane's work, so it
        # lives on the job; the pane keeps it to build the next one. Not named
        # _parent: MessagePump owns that one, and assigning it takes the whole
        # pane out of the DOM.
        self._owner = parent
        # Set before _adopt, which cancels whatever recap the pane had armed.
        self._recap_timer = None
        # Whether the turn now running has called a tool. A turn that only
        # answered needs no recap: the answer is right there on the screen.
        self._used_tools = False
        self._adopt(agent, job_id)
        self.mode = agent.mode
        self._resumed = resumed
        # Tab label, kept here rather than read back from the session file on
        # every repaint of the strip.
        self._title = agent.session.first_user_text() or ""
        # Last context size measured for this session, so redrawing the status
        # bar for an unrelated reason does not blank the readout.
        self._tokens: int | None = None
        # Set by close(): the job is cancelled and the widgets go away, so
        # nothing it unwinds through must touch the DOM. Not named _closing:
        # MessagePump already owns that attribute, and setting it strands the
        # widget's message loop on teardown.
        self._pane_closing = False
        # The spinner, which used to be closures inside the turn worker.
        self._status_state: str | None = None
        self._phrase = ""
        self._turn_started = 0.0
        self._status_timer = None
        self._todo_panel: Static | None = None
        self._queue: list[str] = []
        self._pending_handoff: str | None = None
        self._tps: float | None = None
        # Cumulative over the session: single-request rates swing with each
        # round's tool output size, while the running rate reflects what the
        # cache actually saved and makes server-side invalidation visible as
        # a steady decline instead of one noisy dip.
        self._cache_hit: float | None = None
        self._cache_reads = 0
        self._cache_inputs = 0

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
        return self.job.state is State.RUNNING

    @property
    def is_busy(self) -> bool:
        """Whether a turn is running, or one is already queued behind it.

        What every guard asks. Taken from the job rather than from a worker's
        status: the driver accepts a prompt the instant it is submitted, so
        there is no window in which a second turn can be started by the user
        and the supervisor at the same moment.
        """
        return self.job.is_busy

    @property
    def needs_confirm(self) -> bool:
        """Whether this pane is blocked on a permission confirmation."""
        return self.job.blocked > 0

    @property
    def tab_title(self) -> str:
        """Short label for the tab strip."""
        title = " ".join(self._title.split())
        title = title or "new session"
        # The id is only worth a tab's width while somebody holds it: it is
        # what the model that started this pane calls it.
        return f"{self.job.job_id} {title}" if self.job.parent is not None else title

    def notice(self, renderable) -> None:
        self._add(renderable)

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
        self.job.sink = None
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
            self.job.sink = None
            self._cancel_recap()
            self.job.shutdown()
            self.agent.session.unlock()

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="log")
        status = Horizontal(
            LoadingIndicator(),
            Static(classes="status-label"),
            id="response-status",
        )
        status.display = False
        yield status
        queued = Static(id="queued")
        queued.display = False
        yield queued
        prompt = PromptInput(id="prompt", soft_wrap=True)
        prompt.border_subtitle = "Enter send · Ctrl+J newline · / commands · Esc interrupt · Shift+Tab mode"
        yield prompt

    async def on_mount(self) -> None:
        # The first job is built in __init__, before there is a loop to drive
        # it; every later one is started by _swap_agent as it is made.
        self.job.start()
        self.query_one("#log", VerticalScroll).anchor()
        self._focus_input()
        self._refresh_mode()
        if self._resumed:
            await self._show_resumed()

    def on_key(self, event: events.Key) -> None:
        """Route stray typing back into the prompt.

        Clicking the log (say, to expand a folded result) focuses the scroll
        container and keystrokes would silently vanish; any printable key that
        bubbles up unclaimed refocuses the prompt and lands in it. Modal
        screens and this pane's confirm panel keep the keyboard to themselves.
        """
        if not event.is_printable or len(self.app.screen_stack) > 1 or self.query(ConfirmPanel):
            return
        prompt = self.query_one(PromptInput)
        if self.app.focused is not prompt:
            prompt.focus()
            prompt.insert(event.character)
            event.stop()

    # ---- focus --------------------------------------------------------------

    def _focus_input(self) -> None:
        """Focus what this pane is waiting on, unless another pane is on screen.

        A pending confirmation wins over the prompt: the prompt is hidden
        underneath it, and switching to a pane to answer it has to land on the
        panel or the keys go nowhere.

        Widget.focusable only looks at ``visible``, which is unrelated to
        ``display``, so a hidden pane focusing anything really does take the
        keyboard away from the pane the user is typing into.
        """
        if not self.is_current:
            return
        panels = self.query(ConfirmPanel)
        (panels.last() if panels else self.query_one(PromptInput)).focus()

    # ---- session switching --------------------------------------------------

    def _adopt(self, agent: Agent, job_id: str) -> None:
        """Take over an agent: this pane confirms for it and renders its job."""
        self._cancel_recap()
        self.agent = agent
        agent.confirm = self._confirm
        agent.supervisor = self.supervisor
        agent.pending = self._take_queued
        # One renderer per conversation rather than one per turn: a turn now
        # opens with a UserInput event, which is what resets it. A new one per
        # agent, so a swapped-out session cannot leave a live markdown stream
        # pointing at a log that has just been emptied.
        self._renderer = _EventRenderer(self)
        self.job = AgentJob(job_id, agent, parent=self._owner)
        self.job.sink = self._on_event
        self.job.on_change = self._on_change
        if self.supervisor is not None:
            self.supervisor.register(self.job)

    def _swap_agent(self, agent: Agent) -> None:
        """Put a different conversation in this pane, under a new job.

        The new one is nobody's subagent even when the old one was: the id the
        parent holds names the conversation being left behind, which stays in
        the table as killed and readable rather than quietly becoming a
        different session the parent never asked for.
        """
        self._owner = None
        self._adopt(agent, self._new_job_id())
        self.job.start()

    def _new_job_id(self) -> str:
        return self.supervisor.new_id() if self.supervisor is not None else uuid4().hex[:4]

    def _retire_agent(self) -> list[str]:
        """Release the current session and stop the agents it started.

        Their ids only mean anything inside the conversation being left behind,
        and their sessions are children of it, so nothing would ever read them
        again. Returns the ids, for the caller to report.
        """
        killed = self.supervisor.kill_children(self.agent) if self.supervisor is not None else []
        self.job.cancel()
        if self.supervisor is not None:
            self.supervisor.released(self.job)
        self.agent.session.unlock()
        return killed

    def _report_killed(self, killed: list[str]) -> None:
        if killed:
            self._add(Content.from_markup(
                "[$text-muted]Stopped $n agent(s) started by the previous session: $ids[/]",
                n=str(len(killed)), ids=", ".join(killed)))

    async def _show_resumed(self) -> None:
        renderer = _EventRenderer(self)
        for ev in replay_events(self.agent.history):
            await renderer.handle(ev)
        await renderer.close()
        self._add(Content.from_markup("[$text-muted]Resumed session $id[/]", id=self.agent.session.id[:8]))
        self._sync_statusbar(tokens=True)

    def _reset_measurements(self) -> None:
        """A swapped-in session starts with no measurements of its own."""
        self._tokens = None
        self._tps = None
        self._cache_hit = None
        self._cache_reads = 0
        self._cache_inputs = 0

    def new_session(self) -> None:
        if self.is_busy:
            return
        # The toolset travels with the pane, not with the session: a pane
        # started as a subagent keeps its narrowed set, without spawning or
        # handing off, even though the new conversation is nobody's subagent.
        agent = Agent.open(cwd=self.agent.cwd, confirm=self._confirm, mode=self.mode,
                           config=self.config, toolset=self.agent.toolset,
                           parent=self.agent.session.parent)
        killed = self._retire_agent()
        self._swap_agent(agent)
        self._title = ""
        self._reset_measurements()
        self.query_one("#log", VerticalScroll).remove_children()
        self._todo_panel = None
        self._queue.clear()
        self._refresh_queued()
        self._add(Content.from_markup("[$text-muted]Started new session $id[/]", id=self.agent.session.id[:8]))
        self._report_killed(killed)
        self._sync_statusbar()
        self._notify_state()

    def fork_session(self) -> None:
        if self.is_busy:
            return
        forked = self.agent.session.fork()
        try:
            agent = Agent.open(cwd=self.agent.cwd, session=forked, confirm=self._confirm,
                               mode=self.mode, config=self.config, toolset=self.agent.toolset)
        except SessionError as exc:
            self._add(Content.from_markup("[$text-error b]Cannot fork:[/] $body", body=str(exc)))
            return
        # The conversation on screen is the fork's history verbatim, so the
        # log stays; only the agent underneath changes.
        agent.todos = list(self.agent.todos)
        killed = self._retire_agent()
        self._swap_agent(agent)
        self._add(Content.from_markup("[$text-muted]Forked session $id[/]", id=agent.session.id[:8]))
        self._report_killed(killed)
        self._sync_statusbar()
        self._notify_state()

    # The picker gets its own group: run_turn is an exclusive worker on this
    # same node, and the default group would let a turn cancel the open picker.
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
            self._add(Content.from_markup("[$text-muted]No sessions to resume in this directory[/]"))
            return
        choice = await self.app.push_screen_wait(PickerScreen("Resume session", list(labels)))
        if choice not in labels or self.is_busy:
            self._focus_input()
            return
        try:
            agent = Agent.open(cwd=self.agent.cwd, session=labels[choice], confirm=self._confirm,
                               mode=self.mode, config=self.config, toolset=self.agent.toolset)
        except SessionError as exc:  # already open elsewhere, or no persisted system prompt
            self._add(Content.from_markup("[$text-error b]Cannot resume:[/] $body", body=str(exc)))
            return
        killed = self._retire_agent()
        self._swap_agent(agent)
        self._title = agent.session.first_user_text() or ""
        self._reset_measurements()
        self.query_one("#log", VerticalScroll).remove_children()
        self._todo_panel = None
        self._queue.clear()
        self._refresh_queued()
        await self._show_resumed()
        self._report_killed(killed)
        self._sync_statusbar()
        self._notify_state()
        self._focus_input()

    @work(exclusive=True, group="compact")
    async def compact(self) -> None:
        if self.is_busy:
            self._add(Content.from_markup("[$text-muted]Busy — compact the context after this turn[/]"))
            return
        self._set_status(True, " Compacting context")
        try:
            result = await self.agent.compact_now()
        except Exception as exc:  # noqa: BLE001 — the session is still usable
            self._add(Content.from_markup("[$text-error b]Compaction failed:[/] $body", body=str(exc)))
            return
        finally:
            self._set_status(False)
            self._focus_input()
        if result is None:
            self._add(Content.from_markup("[$text-muted]Nothing to compact yet — the context is still short[/]"))
            return
        self._add(
            Content.from_markup(
                "[$text-muted]Context compacted: $before → ~$after tokens[/]",
                before=f"{result.tokens_before:,}",
                after=f"{result.tokens_after:,}",
            )
        )
        self._sync_statusbar(tokens=True)

    # ---- permission mode ----------------------------------------------------

    def cycle_mode(self) -> None:
        self.mode = tools.MODES[(tools.MODES.index(self.mode) + 1) % len(tools.MODES)]
        self.agent.mode = self.mode
        self._refresh_mode()
        self._sync_statusbar()

    def _refresh_mode(self) -> None:
        self.query_one(PromptInput).border_title = f" {self.mode} "

    # ---- rendering helpers --------------------------------------------------

    # The #log container is anchored once in on_mount: the compositor keeps an
    # anchored scrollable pinned to the bottom as content grows, releases the
    # anchor while the user scrolls up, and re-engages it when they return to
    # the bottom. Helpers therefore just mount widgets — no manual scrolling.

    def _add(self, renderable, classes: str = "") -> Static:
        log = self.query_one("#log", VerticalScroll)
        widget = Static(renderable, classes=classes)
        log.mount(widget)
        return widget

    def _add_user(self, body: str) -> UserMessage:
        log = self.query_one("#log", VerticalScroll)
        widget = UserMessage(body)
        log.mount(widget)
        return widget

    def _set_status(self, visible: bool, label: str = "") -> None:
        status = self.query_one("#response-status", Horizontal)
        status.display = visible
        if visible and label:
            status.query_one(".status-label", Static).update(label)

    def _add_tool_start(self, name: str, args: dict) -> Widget:
        log = self.query_one("#log", VerticalScroll)
        if name == "edit_file":
            path = str(args.get("path") or "")
            old = str(args.get("old_string") or "")
            new = str(args.get("new_string") or "")
            widget: Widget = EditCall(
                path, old, new,
                start_line=locate_line(path, old, new, cwd=self.cwd),
            )
        else:
            widget = ToolCall(name, tools.summarize_call(name, args))
        log.mount(widget)
        return widget

    def _add_tool_result(
        self, result: str, *, label: str = "", denied: bool = False
    ) -> ToolResult:
        log = self.query_one("#log", VerticalScroll)
        widget = ToolResult(result, label=label, denied=denied)
        log.mount(widget)
        return widget

    def _show_todos(self, todos: list[dict]) -> None:
        """Update the panel in place while it is still the tail of the log, so a
        burst of revisions collapses into one; once anything is logged under it
        the panel stays put as a snapshot of the plan at that point and the next
        revision starts a new one."""
        log = self.query_one("#log", VerticalScroll)
        if not todos:
            if self._todo_panel is not None:
                self._todo_panel.remove()
                self._todo_panel = None
            return
        body = self._render_todos(todos)
        if self._todo_panel is not None and log.children[-1:] == [self._todo_panel]:
            self._todo_panel.update(body)
        else:
            self._todo_panel = self._add(body, classes="todos")

    def _render_todos(self, todos: list[dict]) -> Content:
        done = sum(1 for t in todos if t.get("status") == "completed")
        lines = [f"[$text-muted b]Plan[/][$text-muted]  {done}/{len(todos)}[/]"]
        kwargs = {}
        for i, t in enumerate(todos):
            marker, style = _TODO_STYLE.get(t.get("status"), _TODO_STYLE["pending"])
            kwargs[f"c{i}"] = t.get("content", "")
            lines.append(f"[{style}]{marker} ${f'c{i}'}[/]")
        return Content.from_markup("\n".join(lines), **kwargs)

    def _sync_statusbar(self, *, tokens: bool = False) -> None:
        """Redraw the app-wide status bar, which only ever shows this pane's
        session while it is the one on screen."""
        if not self.is_current:
            return
        if tokens:
            self.app.update_statusbar_tokens()
        else:
            self.app.refresh_statusbar()

    # ---- confirmation hook (called from the agent loop) --------------------

    async def _confirm(self, tool_name: str, args: dict) -> bool:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        panel = ConfirmPanel(tool_name, args, future, cwd=self.agent.cwd)
        prompt = self.query_one(PromptInput)
        # Removal below is asynchronous, so a panel from the previous confirm
        # (or an interrupted turn) may still be mounted; sweep it first. The
        # query is pane-scoped: another pane's pending confirmation is not ours
        # to remove.
        await self.query(ConfirmPanel).remove()
        await self.mount(panel, before=prompt)
        prompt.display = False
        # A panel in a background pane must not grab the keyboard: the user's
        # next keystroke would answer a confirmation they never saw.
        if self.is_current:
            panel.focus()
        # Counted on the job rather than here: removing the panel is
        # asynchronous, and the tab badge has to clear the moment the answer is
        # in, not whenever the widget finally goes.
        self.job.mark_blocked(True)
        try:
            verdict = await future
        finally:
            self.job.mark_blocked(False)
            prompt.display = True
            panel.remove()
        return verdict == "allow"

    # ---- input → turn -------------------------------------------------------

    @on(PromptInput.Submitted)
    def handle_submit(self, event: PromptInput.Submitted) -> None:
        text = event.text
        self._cancel_recap()
        self.query_one(PromptInput).clear()
        if self.is_busy:
            self._queue.append(text)
            self._refresh_queued()
            return
        self.job.submit(text)

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

        Called from the agent loop, which runs on this app's event loop, so
        touching the panel from here is safe.
        """
        if self._pane_closing or not self._queue:
            return []
        texts, self._queue = self._queue, []
        self._refresh_queued()
        return texts

    def interrupt(self) -> None:
        # Esc stops whatever this pane is doing on its own, a recap included.
        self._cancel_recap()
        self.job.interrupt()

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
        self._recap_timer = self.set_timer(self.config.recap_idle_seconds, self._recap)

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

        Read-only (see Agent.ask_aside): the recap is never part of the
        conversation, so a resumed session does not replay it.
        """
        self._recap_timer = None
        try:
            text = "".join([delta async for delta in self.agent.ask_aside(
                aside.RECAP_QUESTION, instructions=aside.RECAP_INSTRUCTIONS)])
        except Exception:  # noqa: BLE001 — nobody asked for this, so nobody hears about it
            return
        # Collected before it is mounted rather than streamed in: cancelling
        # then leaves no half-written block behind, and after a wait this long
        # the extra second is nobody's problem.
        if self._pane_closing or self.is_busy or not self.is_current or not text.strip():
            return
        self.query_one("#log", VerticalScroll).mount(RecapMessage(text.strip()))

    # ---- the job's two hooks ------------------------------------------------

    def _on_change(self, job) -> None:
        """This pane's job moved. Called from its driver and from cancellation.

        Only a notification: everything that has to happen in order with the
        turn's own output belongs in the sink below, which is the one place
        guaranteed to run after the last event of a turn.
        """
        if self._pane_closing or not self.is_mounted:
            return
        self._notify_state()

    async def _on_event(self, ev) -> None:
        """Render one event of this pane's job. The job's sink.

        Awaited by the driver, so the renderer's mounts are what paces a turn
        and nothing has to be buffered on the way here.
        """
        if self._pane_closing or not self.app.is_running:
            return
        if isinstance(ev, TurnOver):
            await self._end_turn(ev.result)
            return
        if isinstance(ev, UserInput):
            self._begin_turn(ev.text)
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
            # Zero on both cache fields means the provider does not report
            # caching, so no rate is shown rather than a misleading 0%.
            if (ev.cache_read_tokens or ev.cache_write_tokens) and ev.input_tokens:
                self._cache_reads += ev.cache_read_tokens
                self._cache_inputs += ev.input_tokens
                self._cache_hit = self._cache_reads / self._cache_inputs
            self._sync_statusbar(tokens=True)
        elif isinstance(ev, SessionHandoff):
            # the switch happens once the turn is over, in _end_turn, since
            # new_session refuses to run while one is in flight
            self._pending_handoff = ev.prompt
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
        here and not in the job's inbox.
        """
        await self._renderer.close()
        self._set_state(None)
        if self._status_timer is not None:
            self._status_timer.stop()
            self._status_timer = None
        if result.outcome is Outcome.INTERRUPTED:
            self._add(Content.from_markup("[$text-warning]⏹ Paimon stopped![/]"))
        elif result.outcome is Outcome.FAILED:
            self._add(Content.from_markup("[$text-error b]Error:[/] $body", body=result.error))
        self._focus_input()
        if self._pending_handoff is not None:
            prompt, self._pending_handoff = self._pending_handoff, None
            if result.finished:
                self._complete_handoff(prompt)
                return
        if self._queue:
            text = "\n\n".join(self._queue)
            self._queue.clear()
            self._refresh_queued()
            if result.finished:
                self.job.submit(text)
            else:
                prompt_input = self.query_one(PromptInput)
                draft = prompt_input.text
                prompt_input.load_text(f"{text}\n{draft}" if draft else text)
                prompt_input.move_cursor(prompt_input.document.end)
        # Last, so a turn started just above (a queued message, a handoff) is
        # already busy and arms nothing. Only for a turn that ran to its own
        # end: recapping work the user just stopped is not what they asked for.
        if result.finished:
            self._arm_recap()

    def _complete_handoff(self, prompt: str) -> None:
        """Switch to a fresh session and submit the approved handoff prompt.

        Queued messages were typed against the old context, so they go back
        to the input instead of being fired at the new session.
        """
        if self._queue:
            text = "\n\n".join(self._queue)
            self._queue.clear()
            self._refresh_queued()
            prompt_input = self.query_one(PromptInput)
            draft = prompt_input.text
            prompt_input.load_text(f"{text}\n{draft}" if draft else text)
            prompt_input.move_cursor(prompt_input.document.end)
        hint = resume_hint(self.agent.session.id)
        self.new_session()
        self._add(Content.from_markup(
            "[$text-muted]Handed off — previous session: $hint[/]", hint=hint))
        self.job.submit(prompt)

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
