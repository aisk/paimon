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
from textual.widgets import Button, Static, TextArea
from textual.worker import Worker

from .agent import (
    Agent,
    ReasoningDelta,
    TextDelta,
    TodosUpdate,
    ToolEnd,
    ToolStart,
    TurnEnd,
)
from . import config
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
        self.query_one(PromptInput).focus()
        if self._resumed:
            self._render_history()
            self._add(Content.from_markup("[$text-muted]Continued session $id[/]", id=self.agent.session.id[:8]))
        if not config.MODEL:
            self.action_login()

    def _render_history(self) -> None:
        for message in self.agent.messages[1:]:
            role, body = message.get("role"), message.get("content")
            if role == "user" and body:
                self._add_user(body)
            elif role == "assistant" and body:
                self._add_markdown(body)

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

    def _add(self, renderable, classes: str = "") -> Static:
        log = self.query_one("#log", VerticalScroll)
        widget = Static(renderable, classes=classes)
        log.mount(widget)
        self._scroll()
        return widget

    def _add_markdown(self, body: str) -> AssistantMessage:
        log = self.query_one("#log", VerticalScroll)
        widget = AssistantMessage(body)
        log.mount(widget)
        self._scroll()
        return widget

    def _add_user(self, body: str) -> UserMessage:
        log = self.query_one("#log", VerticalScroll)
        widget = UserMessage(body)
        log.mount(widget)
        self._scroll()
        return widget

    def _scroll(self) -> None:
        log = self.query_one("#log", VerticalScroll)
        # Markdown is laid out asynchronously after a streaming update. Scrolling
        # immediately uses its previous height and can leave the newest lines
        # below the viewport, which looks like a truncated response.
        def scroll_after_layout() -> None:
            log.scroll_end(animate=False)
            # Markdown may request one more layout pass after rendering its blocks.
            self.call_after_refresh(log.scroll_end, animate=False)

        self.call_after_refresh(scroll_after_layout)

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

        assistant: AssistantMessage | None = None
        buffer = ""
        reasoning: Static | None = None
        reasoning_buf = ""

        try:
            async for ev in self.agent.run(text):
                if isinstance(ev, ReasoningDelta):
                    reasoning_buf += ev.text
                    body = Content(reasoning_buf)
                    if reasoning is None:
                        reasoning = self._add(body, classes="reasoning")
                    else:
                        reasoning.update(body)
                    self._scroll()

                elif isinstance(ev, TextDelta):
                    buffer += ev.text
                    if assistant is None:
                        assistant = self._add_markdown(buffer)
                    else:
                        assistant.update_body(buffer)
                    self._scroll()

                elif isinstance(ev, ToolStart):
                    # start fresh assistant/reasoning blocks after a tool runs
                    assistant, buffer = None, ""
                    reasoning, reasoning_buf = None, ""
                    # write_todos renders its own panel via TodosUpdate
                    if ev.name == "write_todos":
                        continue
                    detail = ev.args.get("command") or ev.args.get("path") or json.dumps(ev.args)
                    self._add(
                        Content.from_markup(
                            "[$text-accent b]$name[/]  [$text-muted]$detail[/]",
                            name=ev.name,
                            detail=detail,
                        )
                    )

                elif isinstance(ev, TodosUpdate):
                    self._add(self._render_todos(ev.todos))

                elif isinstance(ev, ToolEnd):
                    if ev.name == "write_todos":
                        continue
                    preview = "\n".join(ev.result.splitlines()[:15])
                    if len(ev.result.splitlines()) > 15:
                        preview += "\n…"
                    classes = "tool-result denied" if ev.denied else "tool-result"
                    self._add(Content(preview or "(no output)"), classes=classes)

                elif isinstance(ev, TurnEnd):
                    self._scroll()
        except asyncio.CancelledError:
            self._add(Content.from_markup("[$text-warning]⏹ Interrupted[/]"))
            raise
        except Exception as exc:  # noqa: BLE001 — show errors instead of crashing the UI
            self._add(Content.from_markup("[$text-error b]Error:[/] $body", body=str(exc)))
        finally:
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
