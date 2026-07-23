"""Textual TUI for the Paimon agent."""

import asyncio
import json
import random
import time
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult, SystemCommand
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.widgets import LoadingIndicator, Static
from textual.widgets.markdown import MarkdownStream
from textual.worker import Worker, WorkerState

from .agent import (
    Agent,
    ContextCompactionFailed,
    ContextCompacted,
    ReasoningDelta,
    TextDelta,
    TodosUpdate,
    ToolEnd,
    ToolStart,
    TurnEnd,
)
from . import compaction, config, tools
from .compaction import SUMMARY_NAME
from .login import LoginScreen
from .session import Session
from .ui import AssistantMessage, ConfirmPanel, PromptInput, ToolResult, UserMessage

_TODO_STYLE = {
    "completed": ("✔", "$text-success"),
    "in_progress": ("▶", "$text-accent b"),
    "pending": ("○", "$text-muted"),
}

# One is picked per turn, Genshin style.
_STATUS_PHRASES = [
    "Paimon is thinking…",
    "Paimon is NOT emergency food…",
    "Counting mora…",
    "Ehe…",
    "Exploring the area ahead…",
    "Snacking on Sweet Madame…",
    "Asking the Traveler…",
    "Wow, treasure…!",
]


class PaimonApp(App):
    CSS = """
    Screen { background: $background; }
    #workspace { height: 1fr; margin: 1 2; }
    #log {
        height: 1fr;
        padding: 0 1 1 1;
        scrollbar-gutter: stable;
        scrollbar-size-vertical: 1;
        scrollbar-background: transparent;
        scrollbar-background-hover: transparent;
        scrollbar-color: $primary 55%;
        scrollbar-color-hover: $primary;
        scrollbar-color-active: $primary;
    }
    #log > * { margin-bottom: 1; }
    .assistant { padding: 0 1; }
    #response-status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    #response-status LoadingIndicator { width: 3; height: 1; color: $primary; }
    #response-status .status-label { width: auto; }
    #queued { height: auto; padding: 0 1; color: $text-muted; }
    .user-message {
        width: 100%;
        height: auto;
        padding: 1 2;
        background: $primary 12%;
        border-left: solid $primary;
    }
    #prompt {
        height: auto;
        min-height: 4;
        max-height: 12;
        padding: 0 1;
        border: round $surface-lighten-2;
        background: transparent;
    }
    #prompt:focus { border: round $primary; background: transparent; }
    #statusbar { height: 1; padding: 0 1; color: $text-muted; }
    .reasoning { color: $text-disabled; text-style: italic; text-opacity: 60%; }
    .tool-result { color: $text-muted; }
    .tool-result.denied { color: $text; background: $error 20%; padding: 0 1; }
    #confirm-panel {
        height: auto;
        margin-top: 1;
        padding: 1 2;
        border: round $warning;
        background: transparent;
    }
    #confirm-panel:focus { border: round $warning; }
    #confirm-detail { height: auto; max-height: 12; margin-top: 1; color: $text-muted; }
    #confirm-options { height: auto; margin-top: 1; }

    LoginScreen, PickerScreen, PromptScreen { align: center middle; }
    #picker-box, #prompt-screen-box, #login-status {
        width: 70%; height: auto; max-height: 70%;
        padding: 1 2; border: round $accent; background: $surface;
    }
    #picker-title, #prompt-screen-title { margin-bottom: 1; }
    #picker-filter { margin-bottom: 1; }
    #picker-list { height: auto; max-height: 20; }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("escape", "interrupt", "Interrupt"),
    ]

    def get_system_commands(self, screen) -> list[SystemCommand]:
        return [
            *super().get_system_commands(screen),
            SystemCommand(
                "Login / switch provider",
                "Reconfigure model, API base and API key",
                self.action_login,
            ),
            SystemCommand("New session", "Start a new empty session", self.action_new_session),
        ]

    def __init__(self, continue_session: bool = False, yolo: bool = False) -> None:
        self._persist_theme_changes = False
        super().__init__()
        self.yolo = yolo
        cwd = Path.cwd()
        session = Session.latest(cwd) if continue_session else None
        self.agent = Agent(cwd=cwd, confirm=None if yolo else self._confirm, session=session)
        self._resumed = session is not None
        self._turn: Worker | None = None
        self._todo_panel: Static | None = None
        self._session_allowed: set[str] = set()
        self._queue: list[str] = []
        if config.THEME in self.available_themes:
            self.theme = config.THEME
        self._persist_theme_changes = True

    def _watch_theme(self, theme_name: str) -> None:
        super()._watch_theme(theme_name)
        if self._persist_theme_changes:
            config.save(theme=theme_name)

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
            prompt.border_subtitle = "Enter send · Ctrl+J newline · Esc interrupt"
            yield prompt
            yield Static(id="statusbar")

    def on_mount(self) -> None:
        self.query_one("#log", VerticalScroll).anchor()
        self.query_one(PromptInput).focus()
        self._refresh_statusbar()
        if self._resumed:
            self._render_history()
            self._add(Content.from_markup("[$text-muted]Continued session $id[/]", id=self.agent.session.id[:8]))
            self._update_statusbar_tokens()
        if not config.MODEL:
            self.action_login()

    def _render_history(self) -> None:
        show_heading = True
        pending_tools: list[str] = []
        for message in self.agent.messages[1:]:
            role, body = message.get("role"), message.get("content")
            if message.get("name") == SUMMARY_NAME:
                self._add(Content.from_markup("[$text-muted]Earlier context was compacted[/]"))
                show_heading = True
                continue
            if role == "user" and body:
                self._add_user(body)
                show_heading = True
            elif role == "assistant":
                if body:
                    self._add_markdown(body, heading=show_heading)
                    show_heading = False
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    name = function.get("name") or "tool"
                    try:
                        args = json.loads(function.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    pending_tools.append(name)
                    if name == "write_todos":
                        self._show_todos(args.get("todos") or [])
                    else:
                        self._add_tool_start(name, args)
            elif role == "tool":
                name = pending_tools.pop(0) if pending_tools else ""
                # the todos panel already shows this result
                if name != "write_todos":
                    self._add_tool_result(str(body or "(no output)"))

    def action_new_session(self) -> None:
        if self._turn is not None and self._turn.is_running:
            return
        self.agent = Agent(cwd=Path.cwd(), confirm=None if self.yolo else self._confirm)
        self.query_one("#log", VerticalScroll).remove_children()
        self._todo_panel = None
        self._session_allowed.clear()
        self._queue.clear()
        self._refresh_queued()
        self._add(Content.from_markup("[$text-muted]Started new session $id[/]", id=self.agent.session.id[:8]))
        self._refresh_statusbar()

    # ---- login --------------------------------------------------------------

    def action_login(self) -> None:
        def _done(completed: bool | None) -> None:
            if completed:
                self._add(
                    Content.from_markup(
                        "[$text-success b]Logged in.[/]  [$text-muted]$model[/]",
                        model=config.MODEL or "",
                    )
                )
            elif not config.MODEL:
                self._add(Content.from_markup("[$text-warning]Login cancelled — no model configured.[/]"))
                self.exit()
            self._refresh_statusbar()
            self.query_one(PromptInput).focus()

        self.push_screen(LoginScreen(), _done)

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

    def _add_markdown(self, body: str, *, heading: bool = True) -> AssistantMessage:
        log = self.query_one("#log", VerticalScroll)
        widget = AssistantMessage(body, heading=heading)
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
        detail = args.get("command") or args.get("path") or json.dumps(args)
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
        """Keep a single todos panel, moving it to the end of the log on updates."""
        if self._todo_panel is not None:
            self._todo_panel.remove()
        self._todo_panel = self._add(self._render_todos(todos))

    def _render_todos(self, todos: list[dict]) -> Content:
        if not todos:
            return Content.from_markup("[$text-muted]Todos cleared[/]")
        lines, kwargs = [], {}
        for i, t in enumerate(todos):
            marker, style = _TODO_STYLE.get(t.get("status"), _TODO_STYLE["pending"])
            kwargs[f"c{i}"] = t.get("content", "")
            lines.append(f"[{style}]{marker} ${f'c{i}'}[/]")
        return Content.from_markup("\n".join(lines), **kwargs)

    # ---- status bar ---------------------------------------------------------

    def _refresh_statusbar(self, tokens: int | None = None) -> None:
        parts = [config.MODEL or "no model", f"session {self.agent.session.id[:8]}"]
        if tokens is not None:
            window = compaction.context_window(config.MODEL, config.COMPACTION_CONTEXT_WINDOW)
            if window:
                parts.append(f"context {tokens / 1000:.1f}k/{window / 1000:.0f}k ({tokens / window:.0%})")
            else:
                parts.append(f"context ~{tokens / 1000:.1f}k tokens")
        self.query_one("#statusbar", Static).update(Content("  ·  ".join(parts)))

    @work(exclusive=True, group="statusbar")
    async def _update_statusbar_tokens(self) -> None:
        # Token counting walks the whole history; keep it off the UI loop.
        tokens = await asyncio.to_thread(
            compaction.count_tokens, config.MODEL, list(self.agent.messages), tools.TOOLS
        )
        self._refresh_statusbar(tokens)

    # ---- confirmation hook (called from the agent loop) --------------------

    async def _confirm(self, tool_name: str, args: dict) -> bool:
        if tool_name in self._session_allowed:
            return True
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        panel = ConfirmPanel(tool_name, args, future)
        prompt = self.query_one(PromptInput)
        await self.query_one("#workspace", Vertical).mount(panel, before=prompt)
        prompt.display = False
        try:
            verdict = await future
        finally:
            prompt.display = True
            panel.remove()
        if verdict == "always":
            self._session_allowed.add(tool_name)
        return verdict in ("allow", "always")

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

        A finished turn submits them as the next turn; an interrupted turn
        hands them back to the input so they aren't fired at a model the user
        just stopped.
        """
        done = (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED)
        if event.worker is not self._turn or event.state not in done or not self._queue:
            return
        text = "\n\n".join(self._queue)
        self._queue.clear()
        self._refresh_queued()
        if event.state == WorkerState.CANCELLED:
            prompt = self.query_one(PromptInput)
            draft = prompt.text
            prompt.load_text(f"{text}\n{draft}" if draft else text)
            prompt.move_cursor(prompt.document.end)
        else:
            self._start_turn(text)

    def action_interrupt(self) -> None:
        if self._turn is not None and self._turn.is_running:
            self._turn.cancel()

    @work(exclusive=True)
    async def run_turn(self, text: str) -> None:
        stream: MarkdownStream | None = None
        reasoning: Static | None = None
        reasoning_buf = ""
        first_text_block = True
        status_visible = True
        phrase = random.choice(_STATUS_PHRASES)
        turn_started = time.monotonic()

        def status_label() -> str:
            elapsed = int(time.monotonic() - turn_started)
            return f" {phrase} {elapsed}s" if elapsed else f" {phrase}"

        self._set_status(True, status_label())

        def show_status() -> None:
            nonlocal status_visible
            status_visible = True
            self._set_status(True, status_label())

        def clear_status() -> None:
            nonlocal status_visible
            status_visible = False
            self._set_status(False)

        def tick() -> None:
            if status_visible:
                self._set_status(True, status_label())

        timer = self.set_interval(1, tick)

        async def close_stream() -> None:
            nonlocal stream
            if stream is not None:
                await stream.stop()
                stream = None

        try:
            async for ev in self.agent.run(text):
                if isinstance(ev, ReasoningDelta):
                    clear_status()
                    reasoning_buf += ev.text
                    body = Content(reasoning_buf)
                    if reasoning is None:
                        reasoning = self._add(body, classes="reasoning")
                    else:
                        reasoning.update(body)

                elif isinstance(ev, TextDelta):
                    clear_status()
                    if stream is None:
                        widget = AssistantMessage("", heading=first_text_block)
                        first_text_block = False
                        # Await the mount so the initial document (the Paimon
                        # heading) is rendered before the stream appends to it.
                        await self.query_one("#log", VerticalScroll).mount(widget)
                        stream = AssistantMessage.get_stream(widget)
                    await stream.write(ev.text)

                elif isinstance(ev, ToolStart):
                    clear_status()
                    # start fresh assistant/reasoning blocks after a tool runs
                    await close_stream()
                    reasoning, reasoning_buf = None, ""
                    # write_todos renders its own panel via TodosUpdate
                    if ev.name == "write_todos":
                        continue
                    self._add_tool_start(ev.name, ev.args)

                elif isinstance(ev, TodosUpdate):
                    clear_status()
                    self._show_todos(ev.todos)

                elif isinstance(ev, ToolEnd):
                    if ev.name == "write_todos":
                        show_status()
                        continue
                    self._add_tool_result(ev.result, denied=ev.denied)
                    show_status()

                elif isinstance(ev, ContextCompacted):
                    clear_status()
                    self._add(
                        Content.from_markup(
                            "[$text-muted]Context compacted: $before → ~$after tokens[/]",
                            before=f"{ev.tokens_before:,}",
                            after=f"{ev.tokens_after:,}",
                        )
                    )
                    show_status()

                elif isinstance(ev, ContextCompactionFailed):
                    clear_status()
                    self._add(
                        Content.from_markup(
                            "[$text-warning]Context compaction failed; continuing without it: $error[/]",
                            error=ev.error,
                        )
                    )
                    show_status()

                elif isinstance(ev, TurnEnd):
                    clear_status()
                    self._update_statusbar_tokens()
        except asyncio.CancelledError:
            self._add(Content.from_markup("[$text-warning]⏹ Paimon stopped![/]"))
            raise
        except Exception as exc:  # noqa: BLE001 — show errors instead of crashing the UI
            self._add(Content.from_markup("[$text-error b]Error:[/] $body", body=str(exc)))
        finally:
            timer.stop()
            await close_stream()
            clear_status()
            self.query_one(PromptInput).focus()


