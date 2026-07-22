"""Textual TUI for the Paimon agent."""

import argparse
import asyncio
import json
from pathlib import Path

from textual import events, on, work
from textual.app import App, ComposeResult, SystemCommand
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, LoadingIndicator, Static, TextArea
from textual.widgets.markdown import MarkdownStream
from textual.worker import Worker

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
from . import config
from .compaction import SUMMARY_NAME
from .login import LoginScreen
from .session import Session
from .ui import AssistantMessage, UserMessage

_TODO_STYLE = {
    "completed": ("✔", "$text-success"),
    "in_progress": ("▶", "$text-accent b"),
    "pending": ("○", "$text-muted"),
}


class PromptInput(TextArea):
    """Multi-line prompt editor. Enter submits; Shift+Enter / Ctrl+J insert a newline."""

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
            return
        if event.key in ("ctrl+j", "shift+enter"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        await super()._on_key(event)


class ConfirmScreen(ModalScreen[bool]):
    """Yes/No confirmation for a dangerous tool call."""

    BINDINGS = [("y", "allow", "Allow"), ("n", "deny", "Deny"), ("escape", "deny", "Deny")]

    def __init__(self, tool_name: str, args: dict) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.args = args

    def compose(self) -> ComposeResult:
        detail = self.args.get("command") or self.args.get("path") or ""
        body = Content.from_markup(
            "[b]Allow this action?[/]\n\n[$text-warning b]$tool[/]  [$text-muted]$detail[/]",
            tool=self.tool_name,
            detail=detail,
        )
        with Vertical(id="confirm-box"):
            yield Static(body)
            with Horizontal(id="confirm-buttons"):
                yield Button("Allow (y)", variant="success", id="allow")
                yield Button("Deny (n)", variant="error", id="deny")

    @on(Button.Pressed, "#allow")
    def action_allow(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#deny")
    def action_deny(self) -> None:
        self.dismiss(False)


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
    .response-status {
        width: auto;
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    .response-status LoadingIndicator { width: 3; height: 1; color: $primary; }
    .response-status .status-label { width: auto; }
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
        margin-top: 1;
        padding: 0 1;
        border: round $surface-lighten-2;
        background: transparent;
    }
    #prompt:focus { border: round $primary; background: transparent; }
    #prompt:disabled { border: round $surface-lighten-1; opacity: 60%; }
    .reasoning { color: $text-disabled; text-style: italic; text-opacity: 60%; }
    .tool-result { color: $text-muted; }
    .tool-result.denied { color: $text; background: $error 20%; padding: 0 1; }
    ConfirmScreen { align: center middle; }
    #confirm-box { width: 70%; height: auto; padding: 1 2; border: round $warning; background: $surface; }
    #confirm-buttons { height: auto; margin-top: 1; }
    #confirm-buttons Button { width: 1fr; }
    #confirm-buttons #allow { margin-right: 1; }

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
            prompt = PromptInput(id="prompt", soft_wrap=True)
            prompt.border_subtitle = "Enter send · Ctrl+J newline · Esc interrupt"
            yield prompt

    def on_mount(self) -> None:
        self.query_one("#log", VerticalScroll).anchor()
        self.query_one(PromptInput).focus()
        if self._resumed:
            self._render_history()
            self._add(Content.from_markup("[$text-muted]Continued session $id[/]", id=self.agent.session.id[:8]))
        if not config.MODEL:
            self.action_login()

    def _render_history(self) -> None:
        for message in self.agent.messages[1:]:
            role, body = message.get("role"), message.get("content")
            if message.get("name") == SUMMARY_NAME:
                self._add(Content.from_markup("[$text-muted]Earlier context was compacted[/]"))
                continue
            if role == "user" and body:
                self._add_user(body)
            elif role == "assistant":
                if body:
                    self._add_markdown(body)
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    name = function.get("name") or "tool"
                    try:
                        args = json.loads(function.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    self._add_tool_start(name, args)
            elif role == "tool":
                self._add_tool_result(str(body or "(no output)"))

    def action_new_session(self) -> None:
        if self._turn is not None and self._turn.is_running:
            return
        self.agent = Agent(cwd=Path.cwd(), confirm=None if self.yolo else self._confirm)
        self.query_one("#log", VerticalScroll).remove_children()
        self._add(Content.from_markup("[$text-muted]Started new session $id[/]", id=self.agent.session.id[:8]))

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

    def _add_markdown(self, body: str) -> AssistantMessage:
        log = self.query_one("#log", VerticalScroll)
        widget = AssistantMessage(body)
        log.mount(widget)
        return widget

    def _add_user(self, body: str) -> UserMessage:
        log = self.query_one("#log", VerticalScroll)
        widget = UserMessage(body)
        log.mount(widget)
        return widget

    def _add_response_status(self) -> Horizontal:
        log = self.query_one("#log", VerticalScroll)
        widget = Horizontal(
            LoadingIndicator(),
            Static(" Thinking…", classes="status-label"),
            classes="response-status",
        )
        log.mount(widget)
        return widget

    def _add_tool_start(self, name: str, args: dict) -> Static:
        detail = args.get("command") or args.get("path") or json.dumps(args)
        return self._add(
            Content.from_markup(
                "[$text-accent b]$name[/]  [$text-muted]$detail[/]",
                name=name,
                detail=detail,
            )
        )

    def _add_tool_result(self, result: str, *, denied: bool = False) -> Static:
        lines = result.splitlines()
        preview = "\n".join(lines[:15])
        if len(lines) > 15:
            preview += "\n…"
        classes = "tool-result denied" if denied else "tool-result"
        return self._add(Content(preview or "(no output)"), classes=classes)

    def _render_todos(self, todos: list[dict]) -> Content:
        if not todos:
            return Content.from_markup("[$text-muted]Todos cleared[/]")
        lines, kwargs = [], {}
        for i, t in enumerate(todos):
            marker, style = _TODO_STYLE.get(t.get("status"), _TODO_STYLE["pending"])
            kwargs[f"c{i}"] = t.get("content", "")
            lines.append(f"[{style}]{marker} ${f'c{i}'}[/]")
        return Content.from_markup("\n".join(lines), **kwargs)

    # ---- confirmation hook (called from the agent loop) --------------------

    async def _confirm(self, tool_name: str, args: dict) -> bool:
        return await self.push_screen_wait(ConfirmScreen(tool_name, args))

    # ---- input → turn -------------------------------------------------------

    @on(PromptInput.Submitted)
    def handle_submit(self, event: PromptInput.Submitted) -> None:
        text = event.text
        self.query_one(PromptInput).clear()
        self._add_user(text)
        self._turn = self.run_turn(text)

    def action_interrupt(self) -> None:
        if self._turn is not None and self._turn.is_running:
            self._turn.cancel()

    @work(exclusive=True)
    async def run_turn(self, text: str) -> None:
        inp = self.query_one(PromptInput)
        inp.disabled = True

        stream: MarkdownStream | None = None
        reasoning: Static | None = None
        reasoning_buf = ""
        status: Horizontal | None = self._add_response_status()

        def clear_status() -> None:
            nonlocal status
            if status is not None:
                status.remove()
                status = None

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
                        widget = AssistantMessage("")
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
                    self._add(self._render_todos(ev.todos))

                elif isinstance(ev, ToolEnd):
                    if ev.name == "write_todos":
                        status = self._add_response_status()
                        continue
                    self._add_tool_result(ev.result, denied=ev.denied)
                    status = self._add_response_status()

                elif isinstance(ev, ContextCompacted):
                    clear_status()
                    self._add(
                        Content.from_markup(
                            "[$text-muted]Context compacted: $before → ~$after tokens[/]",
                            before=f"{ev.tokens_before:,}",
                            after=f"{ev.tokens_after:,}",
                        )
                    )
                    status = self._add_response_status()

                elif isinstance(ev, ContextCompactionFailed):
                    clear_status()
                    self._add(
                        Content.from_markup(
                            "[$text-warning]Context compaction failed; continuing without it: $error[/]",
                            error=ev.error,
                        )
                    )
                    status = self._add_response_status()

                elif isinstance(ev, TurnEnd):
                    clear_status()
        except asyncio.CancelledError:
            self._add(Content.from_markup("[$text-warning]⏹ Interrupted[/]"))
            raise
        except Exception as exc:  # noqa: BLE001 — show errors instead of crashing the UI
            self._add(Content.from_markup("[$text-error b]Error:[/] $body", body=str(exc)))
        finally:
            await close_stream()
            clear_status()
            inp.disabled = False
            inp.focus()


def main() -> None:
    parser = argparse.ArgumentParser(description="Paimon terminal code agent")
    parser.add_argument("-c", "--continue", dest="continue_session", action="store_true",
                        help="continue the most recent session for this directory")
    parser.add_argument("--yolo", action="store_true",
                        help="allow dangerous tool calls without confirmation")
    args = parser.parse_args()
    PaimonApp(continue_session=args.continue_session, yolo=args.yolo).run()


if __name__ == "__main__":
    main()
