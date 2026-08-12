"""Textual TUI for the Paimon agent."""

import asyncio
import random
import time
from datetime import datetime
from pathlib import Path

from textual import events, on, work
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.widgets import LoadingIndicator, Static
from textual.widgets.markdown import MarkdownStream
from textual.worker import Worker, WorkerState

from .agent import (
    Agent,
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
from . import compaction, tools
from .config import DEFAULT_PROFILE, Config, list_profiles
from .login import LoginScreen, PickerScreen, PromptScreen
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
    """Renders agent events into the log.

    The single rendering path: live turns and resumed-history replay both feed
    events through ``handle``, so history always looks like it did live.
    """

    def __init__(self, app: "PaimonApp") -> None:
        self._app = app
        self._stream: MarkdownStream | None = None
        self._reasoning: FoldedText | None = None
        self._reasoning_buf = ""
        self._first_text_block = True

    async def handle(self, ev: object) -> None:
        if isinstance(ev, UserInput):
            await self.close()
            self._first_text_block = True
            self._app._add_user(ev.text)

        elif isinstance(ev, CompactionNotice):
            await self.close()
            self._first_text_block = True
            self._app._add(Content.from_markup("[$text-muted]Earlier context was compacted[/]"))

        elif isinstance(ev, ReasoningDelta):
            self._reasoning_buf += ev.text
            if self._reasoning is None:
                self._reasoning = FoldedText(
                    "", classes="reasoning", expanded=self._app.config.show_reasoning
                )
                await self._app.query_one("#log", VerticalScroll).mount(self._reasoning)
            self._reasoning.set_text(self._reasoning_buf)

        elif isinstance(ev, TextDelta):
            if self._stream is None:
                widget = AssistantMessage("", heading=self._first_text_block)
                self._first_text_block = False
                # Await the mount so the initial document (the Paimon heading)
                # is rendered before the stream appends to it.
                await self._app.query_one("#log", VerticalScroll).mount(widget)
                self._stream = AssistantMessage.get_stream(widget)
            await self._stream.write(ev.text)

        elif isinstance(ev, ToolStart):
            # start fresh assistant/reasoning blocks after a tool runs
            await self.close()
            self._app._add_tool_start(ev.name, ev.args)

        elif isinstance(ev, TodosUpdate):
            await self.close()
            self._app._show_todos(ev.todos)

        elif isinstance(ev, ToolEnd):
            self._app._add_tool_result(ev.result, denied=ev.denied)

        elif isinstance(ev, ContextCompacted):
            self._app._add(
                Content.from_markup(
                    "[$text-muted]Context compacted: $before → ~$after tokens[/]",
                    before=f"{ev.tokens_before:,}",
                    after=f"{ev.tokens_after:,}",
                )
            )

        elif isinstance(ev, ContextCompactionFailed):
            self._app._add(
                Content.from_markup(
                    "[$text-warning]Context compaction failed; continuing without it: $error[/]",
                    error=ev.error,
                )
            )

        elif isinstance(ev, ModelRetry):
            self._app._add(
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
        if self._reasoning is not None and self._app.config.show_reasoning:
            # fold the live stream now that the block is over; blocks the user
            # clicked open themselves are left alone
            self._reasoning.collapse()
        self._reasoning = None
        self._reasoning_buf = ""


class PaimonApp(App):
    CSS_PATH = "app.tcss"

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("escape", "interrupt", "Interrupt"),
        # priority so the focused PromptInput (a TextArea) can't swallow the key
        Binding("shift+tab", "cycle_mode", "Cycle permission mode", priority=True),
    ]

    def get_system_commands(self, screen) -> list[SystemCommand]:
        return [
            *super().get_system_commands(screen),
            SystemCommand(
                "Login / switch provider",
                "Reconfigure model, API base and API key",
                self.action_login,
            ),
            SystemCommand(
                "Switch profile",
                "Use another profile's account and model, or create a new one",
                self.action_switch_profile,
            ),
            SystemCommand(
                "Toggle thinking display",
                "Show or hide the model's reasoning stream (it is generated either way)",
                self.action_toggle_reasoning,
            ),
            SystemCommand(
                "Compact context",
                "Summarize the earlier conversation now instead of waiting for the context to fill",
                self.action_compact,
            ),
            SystemCommand("New session", "Start a new empty session", self.action_new_session),
            SystemCommand("Fork session", "Copy this conversation into a new session and continue there",
                          self.action_fork_session),
            SystemCommand("Resume session", "Pick an earlier session in this directory to resume",
                          self.action_resume_session),
        ]

    def __init__(self, agent: Agent, *, resumed: bool = False, pick_session: bool = False) -> None:
        self._persist_theme_changes = False
        super().__init__()
        self.agent = agent
        self.agent.confirm = self._confirm
        self.mode = agent.mode
        self.config = agent.config
        self._resumed = resumed
        self._pick_session = pick_session
        self._turn: Worker | None = None
        # run_turn reports model errors in the log instead of letting them
        # escape the worker (which would exit the app), so the worker still
        # ends up SUCCESS; this is what tells the two apart afterwards.
        self._turn_failed = False
        self._todo_panel: Static | None = None
        self._queue: list[str] = []
        self._pending_handoff: str | None = None
        self._tps: float | None = None
        if self.config.theme in self.available_themes:
            self.theme = self.config.theme
        self._persist_theme_changes = True

    def _watch_theme(self, theme_name: str) -> None:
        super()._watch_theme(theme_name)
        if self._persist_theme_changes:
            self.config.save(theme=theme_name)

    def compose(self) -> ComposeResult:
        with Vertical(id="workspace"):
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
            yield Static(id="statusbar")

    async def on_mount(self) -> None:
        self.query_one("#log", VerticalScroll).anchor()
        self.query_one(PromptInput).focus()
        self._refresh_mode()
        self._refresh_statusbar()
        if self._resumed:
            await self._show_resumed()
        if not self.config.model:
            self.action_login()
        elif self._pick_session:
            self.action_resume_session()

    def on_key(self, event: events.Key) -> None:
        """Route stray typing back into the prompt.

        Clicking the log (say, to expand a folded result) focuses the scroll
        container and keystrokes would silently vanish; any printable key that
        bubbles up unclaimed refocuses the prompt and lands in it. Modal
        screens and the confirm panel keep the keyboard to themselves.
        """
        if not event.is_printable or len(self.screen_stack) > 1 or self.query(ConfirmPanel):
            return
        prompt = self.query_one(PromptInput)
        if self.focused is not prompt:
            prompt.focus()
            prompt.insert(event.character)
            event.stop()

    async def _show_resumed(self) -> None:
        renderer = _EventRenderer(self)
        for ev in replay_events(self.agent.history):
            await renderer.handle(ev)
        await renderer.close()
        self._add(Content.from_markup("[$text-muted]Resumed session $id[/]", id=self.agent.session.id[:8]))
        self._update_statusbar_tokens()

    def action_new_session(self) -> None:
        if self._turn is not None and self._turn.is_running:
            return
        agent = Agent.open(cwd=self.agent.cwd, confirm=self._confirm, mode=self.mode,
                           config=self.config)
        self.agent.session.unlock()
        self.agent = agent
        self.query_one("#log", VerticalScroll).remove_children()
        self._todo_panel = None
        self._queue.clear()
        self._refresh_queued()
        self._add(Content.from_markup("[$text-muted]Started new session $id[/]", id=self.agent.session.id[:8]))
        self._refresh_statusbar()

    def action_fork_session(self) -> None:
        if self._turn is not None and self._turn.is_running:
            return
        forked = self.agent.session.fork()
        try:
            agent = Agent.open(cwd=self.agent.cwd, session=forked, confirm=self._confirm,
                               mode=self.mode, config=self.config)
        except SessionError as exc:
            self._add(Content.from_markup("[$text-error b]Cannot fork:[/] $body", body=str(exc)))
            return
        # The conversation on screen is the fork's history verbatim, so the
        # log stays; only the agent underneath changes.
        agent.todos = list(self.agent.todos)
        self.agent.session.unlock()
        self.agent = agent
        self._add(Content.from_markup("[$text-muted]Forked session $id[/]", id=agent.session.id[:8]))
        self._refresh_statusbar()

    @work
    async def action_resume_session(self) -> None:
        if self._turn is not None and self._turn.is_running:
            return
        labels = {_session_label(session): session for session in Session.list(Path.cwd())}
        if not labels:
            self._add(Content.from_markup("[$text-muted]No sessions to resume in this directory[/]"))
            return
        choice = await self.push_screen_wait(PickerScreen("Resume session", list(labels)))
        if choice not in labels or (self._turn is not None and self._turn.is_running):
            self.query_one(PromptInput).focus()
            return
        try:
            agent = Agent.open(cwd=self.agent.cwd, session=labels[choice], confirm=self._confirm,
                               mode=self.mode, config=self.config)
        except SessionError as exc:  # busy in another process, or no persisted system prompt
            self._add(Content.from_markup("[$text-error b]Cannot resume:[/] $body", body=str(exc)))
            return
        self.agent.session.unlock()
        self.agent = agent
        self.query_one("#log", VerticalScroll).remove_children()
        self._todo_panel = None
        self._queue.clear()
        self._refresh_queued()
        await self._show_resumed()
        self._refresh_statusbar()
        self.query_one(PromptInput).focus()

    @work(exclusive=True, group="compact")
    async def action_compact(self) -> None:
        if self._turn is not None and self._turn.is_running:
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
            self.query_one(PromptInput).focus()
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
        self._update_statusbar_tokens()

    # ---- permission mode ----------------------------------------------------

    def action_cycle_mode(self) -> None:
        self.mode = tools.MODES[(tools.MODES.index(self.mode) + 1) % len(tools.MODES)]
        self.agent.mode = self.mode
        self._refresh_mode()
        self._refresh_statusbar()

    def _refresh_mode(self) -> None:
        self.query_one(PromptInput).border_title = f" {self.mode} "

    def action_toggle_reasoning(self) -> None:
        self.config.save(show_reasoning=not self.config.show_reasoning)
        state = "streamed live" if self.config.show_reasoning else "folded"
        self._add(Content.from_markup(f"[$text-muted]Thinking: {state}[/]"))

    # ---- login --------------------------------------------------------------

    def _config_is_busy(self) -> bool:
        """Whether a running turn makes it unsafe to rewrite the config.

        Config is process-wide, and a turn re-reads the model at the top of
        every step (Agent._model), so a login or profile switch landing
        mid-turn silently swaps providers between two tool calls. Both refuse
        while a turn is in flight. This is the one place that widens to "any
        pane is running" once panes exist; today there is only one.
        """
        return self._turn is not None and self._turn.is_running

    def action_login(self) -> None:
        if self._config_is_busy():
            self._add(Content.from_markup("[$text-muted]Busy — log in after this turn[/]"))
            return

        def _done(completed: bool | None) -> None:
            if completed:
                self._add(
                    Content.from_markup(
                        "[$text-success b]Logged in.[/]  [$text-muted]$model[/]",
                        model=self.config.model or "",
                    )
                )
            elif not self.config.model:
                self._add(Content.from_markup("[$text-warning]Login cancelled — no model configured.[/]"))
                self.exit()
            self._refresh_statusbar()
            self.query_one(PromptInput).focus()

        self.push_screen(LoginScreen(), _done)

    # ---- profiles -----------------------------------------------------------

    _NEW_PROFILE = "New profile…"

    @work
    async def action_switch_profile(self) -> None:
        if self._config_is_busy():
            self._add(Content.from_markup("[$text-muted]Busy — switch profiles after this turn[/]"))
            return
        current = self.config.profile
        labels = {f"{name} (current)" if name == current else name: name
                  for name in list_profiles()}
        choice = await self.push_screen_wait(
            PickerScreen("Switch profile", [*labels, self._NEW_PROFILE]))
        if choice == self._NEW_PROFILE:
            choice = await self.push_screen_wait(PromptScreen("New profile name"))
        # An unlisted typed name is accepted too: switching to a profile that
        # does not exist yet is how one gets created.
        name = labels.get(choice, choice) if choice else None
        if name is None or name == current:
            self.query_one(PromptInput).focus()
            return
        try:
            switched = Config.load(name)
        except ValueError as exc:
            self._add(Content.from_markup("[$text-error b]Cannot switch:[/] $body", body=str(exc)))
            self.query_one(PromptInput).focus()
            return
        previous_config = self.config
        self.config = self.agent.config = switched
        if not self.config.model:
            completed = await self.push_screen_wait(LoginScreen())
            if not completed:
                self.config = self.agent.config = previous_config
                self._add(Content.from_markup("[$text-muted]Profile switch cancelled[/]"))
                self.query_one(PromptInput).focus()
                return
        if self.config.theme in self.available_themes:
            self.theme = self.config.theme
        self._add(Content.from_markup(
            "[$text-success b]Profile:[/] $name  [$text-muted]$model[/]",
            name=name, model=self.config.model or ""))
        self._refresh_statusbar()
        self.query_one(PromptInput).focus()

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

    # ---- status bar ---------------------------------------------------------

    def _refresh_statusbar(self, tokens: int | None = None) -> None:
        # The agent's model, not the config's: a pane may override it.
        parts = [f"{self.mode} mode", self.agent.model_name or "no model",
                 f"session {self.agent.session.id[:8]}"]
        if self.config.profile != DEFAULT_PROFILE:
            parts.insert(1, f"profile {self.config.profile}")
        if tokens is not None:
            window = compaction.context_window(self.agent.model_name,
                                               self.config.compaction_context_window)
            if window:
                parts.append(f"context {tokens / 1000:.1f}k/{window / 1000:.0f}k ({tokens / window:.0%})")
            else:
                # Unknown window: auto-compaction cannot trigger, so say so
                # instead of looking like it is merely waiting to.
                parts.append(f"context ~{tokens / 1000:.1f}k tokens "
                             "(auto-compaction off: unknown context window)")
        if self._tps is not None:
            parts.append(f"{self._tps:.0f} tps")
        self.query_one("#statusbar", Static).update(Content("  ·  ".join(parts)))

    @work(exclusive=True, group="statusbar")
    async def _update_statusbar_tokens(self) -> None:
        self._refresh_statusbar(await self.agent.count_context_tokens())

    # ---- confirmation hook (called from the agent loop) --------------------

    async def _confirm(self, tool_name: str, args: dict) -> bool:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        panel = ConfirmPanel(tool_name, args, future)
        prompt = self.query_one(PromptInput)
        # Removal below is asynchronous, so a panel from the previous confirm
        # (or an interrupted turn) may still be mounted; sweep it first or the
        # fixed widget ID collides.
        await self.query(ConfirmPanel).remove()
        await self.query_one("#workspace", Vertical).mount(panel, before=prompt)
        prompt.display = False
        try:
            verdict = await future
        finally:
            prompt.display = True
            panel.remove()
        return verdict == "allow"

    # ---- input → turn -------------------------------------------------------

    @on(PromptInput.Submitted)
    def handle_submit(self, event: PromptInput.Submitted) -> None:
        text = event.text
        self.query_one(PromptInput).clear()
        if self._turn is not None and self._turn.is_running:
            self._queue.append(text)
            self._refresh_queued()
            return
        self._start_turn(text)

    def _start_turn(self, text: str) -> None:
        self._add_user(text)
        self._turn = self.run_turn(text)

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
            self._start_turn(text)
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
        self.action_new_session()
        self._add(Content.from_markup(
            "[$text-muted]Handed off — previous session: $hint[/]", hint=hint))
        self._start_turn(prompt)

    def action_interrupt(self) -> None:
        if self._turn is not None and self._turn.is_running:
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
                    self._update_statusbar_tokens()
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
                    self._update_statusbar_tokens()
                elif isinstance(ev, SessionHandoff):
                    # the switch happens after this worker finishes, in
                    # on_worker_state_changed, since action_new_session
                    # refuses to run while a turn is in flight
                    self._pending_handoff = ev.prompt
                else:
                    set_state(None)
        except asyncio.CancelledError:
            # Quitting the app also cancels this worker, but only after the
            # DOM is torn down — mounting anything then raises MountError.
            if self.is_running:
                self._add(Content.from_markup("[$text-warning]⏹ Paimon stopped![/]"))
            raise
        except Exception as exc:  # noqa: BLE001 — show errors instead of crashing the UI
            self._turn_failed = True
            self._add(Content.from_markup("[$text-error b]Error:[/] $body", body=str(exc)))
        finally:
            timer.stop()
            if self.is_running:
                await renderer.close()
                set_state(None)
                self.query_one(PromptInput).focus()


