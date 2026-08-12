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

from textual import events, on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.widgets import LoadingIndicator, Static
from textual.widgets.markdown import MarkdownStream
from textual.worker import Worker, WorkerState

from . import lockfile, tools
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
from .login import PickerScreen
from .session import Session, SessionError, resume_hint
from .ui import AssistantMessage, ConfirmPanel, FoldedText, PromptInput, ToolResult, UserMessage

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
                    "", classes="reasoning", expanded=self._pane.config.show_reasoning
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
            self._pane._add_tool_start(ev.name, ev.args)

        elif isinstance(ev, TodosUpdate):
            await self.close()
            self._pane._show_todos(ev.todos)

        elif isinstance(ev, ToolEnd):
            self._pane._add_tool_result(ev.result, denied=ev.denied)

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


class SessionPane(Vertical):
    """A single conversation. A container, deliberately not a Screen: screens
    are mutually exclusive and ``app.query_one`` only sees the active one."""

    class StateChanged(Message):
        """This pane started or finished a turn, or changed what it shows.

        Worker.StateChanged does not bubble and the turn worker now lives on
        the pane, so anything app-wide that tracks panes — the tab strip, the
        status bar — only hears about them through this.
        """

        def __init__(self, pane: "SessionPane") -> None:
            self.pane = pane
            super().__init__()

    def __init__(self, agent: Agent, *, resumed: bool = False, id: str | None = None,
                 supervisor=None, agent_id: str | None = None) -> None:
        super().__init__(id=id)
        self.agent = agent
        # The pool this pane's agent can start and talk to other agents through,
        # and — when this pane is one of those agents — the id its parent knows
        # it by. Shown on the tab, since that id is what the model reports.
        self.supervisor = supervisor
        self.agent_id = agent_id
        self._adopt(agent)
        self.mode = agent.mode
        self._resumed = resumed
        self._turn: Worker | None = None
        # Tab label, kept here rather than read back from the session file on
        # every repaint of the strip.
        self._title = agent.session.first_user_text() or ""
        # Last context size measured for this session, so redrawing the status
        # bar for an unrelated reason does not blank the readout.
        self._tokens: int | None = None
        # Depth of pending _confirm calls; see needs_confirm.
        self._confirming = 0
        # Set by close(): the turn is cancelled and the widgets go away, so
        # whatever the worker unwinds through must not touch the DOM. Not
        # named _closing: MessagePump already owns that attribute, and
        # setting it strands the widget's message loop on teardown.
        self._pane_closing = False
        # run_turn reports model errors in the log instead of letting them
        # escape the worker (which would exit the app), so the worker still
        # ends up SUCCESS; this is what tells the two apart afterwards.
        self._turn_failed = False
        self._todo_panel: Static | None = None
        self._queue: list[str] = []
        self._pending_handoff: str | None = None
        self._tps: float | None = None

    @property
    def config(self):
        """The active config. Process-wide, kept in sync with the app's."""
        return self.agent.config

    @property
    def is_current(self) -> bool:
        """Whether this is the pane the user is looking at."""
        return self.app.pane is self

    @property
    def is_turn_running(self) -> bool:
        """Whether a turn is streaming right now. For display only."""
        return self._turn is not None and self._turn.is_running

    @property
    def is_busy(self) -> bool:
        """Whether a turn is running or about to start.

        What every guard asks. A worker sits in PENDING for a tick before it
        runs, and a second turn started in that window cancels the first
        (run_turn is exclusive) — which, now that a turn can be started by the
        supervisor and by the user at the same moment, is reachable.
        """
        return self._turn is not None and not self._turn.is_finished

    @property
    def turn_failed(self) -> bool:
        """Whether the last finished turn ended in an error."""
        return self._turn_failed

    @property
    def needs_confirm(self) -> bool:
        """Whether this pane is blocked on a permission confirmation.

        Counted rather than queried: removing the panel is asynchronous, and
        the tab badge must clear the moment the answer is in.
        """
        return self._confirming > 0

    @property
    def tab_title(self) -> str:
        """Short label for the tab strip."""
        title = " ".join(self._title.split())
        title = title or "new session"
        return f"{self.agent_id} {title}" if self.agent_id else title

    def _notify_state(self) -> None:
        self.post_message(self.StateChanged(self))

    def close(self) -> None:
        """Give up everything this pane owns; the app removes the widget.

        Idempotent: the widget goes away one message loop after the agent is
        killed, and unlocking a session twice would drop a claim somebody else
        has since taken.
        """
        if self._pane_closing:
            return
        self._pane_closing = True
        self.interrupt()
        self._retire_agent()

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

    def _adopt(self, agent: Agent) -> None:
        """Take over an agent: this pane confirms for it and supervises it."""
        self.agent = agent
        agent.confirm = self._confirm
        agent.supervisor = self.supervisor

    def _retire_agent(self) -> list[str]:
        """Release the current session and stop the agents it started.

        Their ids only mean anything inside the conversation being left behind,
        and their sessions are children of it, so nothing would ever read them
        again. Returns the ids, for the caller to report.
        """
        killed = self.supervisor.kill_children(self.agent) if self.supervisor is not None else []
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

    def new_session(self) -> None:
        if self.is_busy:
            return
        # The toolset travels with the pane, not with the session: a pane that
        # is somebody's subagent stays one, without spawning or handing off.
        agent = Agent.open(cwd=self.agent.cwd, confirm=self._confirm, mode=self.mode,
                           config=self.config, toolset=self.agent.toolset,
                           parent=self.agent.session.parent)
        killed = self._retire_agent()
        self._adopt(agent)
        self._title = ""
        self._tokens = None
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
        self._adopt(agent)
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
        self._adopt(agent)
        self._title = agent.session.first_user_text() or ""
        self._tokens = None
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

    def _add_tool_start(self, name: str, args: dict) -> Static:
        detail = tools.summarize_call(name, args)
        return self._add(
            Content.from_markup(
                "[$text-accent b]$name[/]  [$text-muted]$detail[/]",
                name=name,
                detail=detail,
            )
        )

    def _add_tool_result(self, result: str, *, denied: bool = False) -> ToolResult:
        log = self.query_one("#log", VerticalScroll)
        widget = ToolResult(result, denied=denied)
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
        panel = ConfirmPanel(tool_name, args, future)
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
        self._confirming += 1
        self._notify_state()
        try:
            verdict = await future
        finally:
            self._confirming -= 1
            prompt.display = True
            panel.remove()
            self._notify_state()
        return verdict == "allow"

    # ---- input → turn -------------------------------------------------------

    @on(PromptInput.Submitted)
    def handle_submit(self, event: PromptInput.Submitted) -> None:
        text = event.text
        self.query_one(PromptInput).clear()
        if self.is_busy:
            self._queue.append(text)
            self._refresh_queued()
            return
        self.start_turn(text)

    def start_turn(self, text: str) -> None:
        """Begin a turn on this pane's agent. Only valid while it is idle."""
        if not self._title:
            self._title = text
        self._add_user(text)
        self._turn = self.run_turn(text)
        self._notify_state()

    def _refresh_queued(self) -> None:
        widget = self.query_one("#queued", Static)
        widget.display = bool(self._queue)
        if not self._queue:
            widget.update("")
            return
        kwargs = {f"q{i}": text for i, text in enumerate(self._queue)}
        lines = "\n".join(f"[$text-muted]⏸ ${key}[/]" for key in kwargs)
        widget.update(Content.from_markup(lines, **kwargs))

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Flush queued prompts once the running turn is over.

        A finished turn submits them as the next turn; a turn that was
        stopped or that failed hands them back to the input, so they aren't
        fired at a model the user just stopped or that just errored.
        """
        done = (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED)
        if event.worker is not self._turn or event.state not in done:
            return
        self._notify_state()
        finished = event.state == WorkerState.SUCCESS and not self._turn_failed
        if self._pending_handoff is not None:
            prompt, self._pending_handoff = self._pending_handoff, None
            if finished:
                self._complete_handoff(prompt)
                return
        if not self._queue:
            return
        text = "\n\n".join(self._queue)
        self._queue.clear()
        self._refresh_queued()
        if finished:
            self.start_turn(text)
        else:
            prompt = self.query_one(PromptInput)
            draft = prompt.text
            prompt.load_text(f"{text}\n{draft}" if draft else text)
            prompt.move_cursor(prompt.document.end)

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
        self.start_turn(prompt)

    def interrupt(self) -> None:
        if self.is_busy:
            self._turn.cancel()

    @work(exclusive=True)
    async def run_turn(self, text: str) -> None:
        renderer = _EventRenderer(self)
        self._turn_failed = False
        turn_started = time.monotonic()
        state: str | None = None
        phrase = ""

        def status_label() -> str:
            elapsed = int(time.monotonic() - turn_started)
            return f" {phrase} {elapsed}s" if elapsed else f" {phrase}"

        def set_state(new: str | None) -> None:
            # The phrase is re-rolled only on state changes so it doesn't
            # flicker through the pool while a state lasts.
            nonlocal state, phrase
            if new == state:
                return
            state = new
            if new is None:
                self._set_status(False)
            else:
                phrase = random.choice(_STATUS_PHRASES[new])
                self._set_status(True, status_label())

        def tick() -> None:
            if state is not None:
                self._set_status(True, status_label())

        set_state("waiting")
        timer = self.set_interval(1, tick)

        try:
            async for ev in self.agent.run(text):
                await renderer.handle(ev)
                if isinstance(ev, TurnEnd):
                    set_state(None)
                    self._sync_statusbar(tokens=True)
                elif isinstance(ev, ToolStart):
                    set_state("tool")
                elif isinstance(ev, (ToolEnd, TodosUpdate, ContextCompacted,
                                     ContextCompactionFailed, ModelRetry)):
                    # the model is about to react to what just happened
                    set_state("waiting")
                elif isinstance(ev, ReasoningDelta):
                    # folded reasoning only ticks a line count, so the spinner stands in
                    set_state(None if self.config.show_reasoning else "thinking")
                elif isinstance(ev, RequestStats):
                    self._tps = ev.output_tokens / ev.seconds
                    self._sync_statusbar(tokens=True)
                elif isinstance(ev, SessionHandoff):
                    # the switch happens after this worker finishes, in
                    # on_worker_state_changed, since new_session
                    # refuses to run while a turn is in flight
                    self._pending_handoff = ev.prompt
                else:
                    set_state(None)
        except asyncio.CancelledError:
            # Quitting the app, and closing this pane, cancel this worker
            # while its widgets are going away — mounting anything into them
            # then raises MountError.
            if self.app.is_running and not self._pane_closing:
                self._add(Content.from_markup("[$text-warning]⏹ Paimon stopped![/]"))
            raise
        except Exception as exc:  # noqa: BLE001 — show errors instead of crashing the UI
            self._turn_failed = True
            self._add(Content.from_markup("[$text-error b]Error:[/] $body", body=str(exc)))
        finally:
            timer.stop()
            if self.app.is_running and not self._pane_closing:
                await renderer.close()
                set_state(None)
                self._focus_input()
