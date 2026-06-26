"""Textual TUI for the Paimon agent."""

import asyncio
import json
from pathlib import Path

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
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
    ToolEnd,
    ToolStart,
    TurnEnd,
)


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
    #log { height: 1fr; padding: 0 1; }
    #log > Static { margin-bottom: 1; }
    #prompt { height: auto; max-height: 12; border: round $surface; padding: 0 1; }
    #prompt:focus { border: round $accent; }
    .reasoning { color: $text-disabled; text-style: italic; text-opacity: 60%; }
    .tool-result { color: $text-muted; }
    .tool-result.denied { color: $text; background: $error 20%; padding: 0 1; }
    ConfirmScreen { align: center middle; }
    #confirm-box { width: 70%; height: auto; padding: 1 2; border: round $warning; background: $surface; }
    #confirm-buttons { height: auto; margin-top: 1; }
    #confirm-buttons Button { width: 1fr; }
    #confirm-buttons #allow { margin-right: 1; }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        Binding("escape", "interrupt", "Interrupt", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.agent = Agent(cwd=Path.cwd(), confirm=self._confirm)
        self._turn: Worker | None = None

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="log")
        prompt = PromptInput(id="prompt", soft_wrap=True)
        prompt.border_subtitle = "Enter to send · Ctrl+J for newline · Ctrl+C to quit"
        yield prompt

    def on_mount(self) -> None:
        self.query_one(PromptInput).focus()

    # ---- rendering helpers --------------------------------------------------

    def _add(self, renderable, classes: str = "") -> Static:
        log = self.query_one("#log", VerticalScroll)
        widget = Static(renderable, classes=classes)
        log.mount(widget)
        log.scroll_end(animate=False)
        return widget

    def _scroll(self) -> None:
        self.query_one("#log", VerticalScroll).scroll_end(animate=False)

    # ---- confirmation hook (called from the agent loop) --------------------

    async def _confirm(self, tool_name: str, args: dict) -> bool:
        return await self.push_screen_wait(ConfirmScreen(tool_name, args))

    # ---- input → turn -------------------------------------------------------

    @on(PromptInput.Submitted)
    def handle_submit(self, event: PromptInput.Submitted) -> None:
        text = event.text
        self.query_one(PromptInput).clear()
        self._add(Content.from_markup("[$text-primary b]Traveler[/]\n$body", body=text))
        self._turn = self.run_turn(text)

    def action_interrupt(self) -> None:
        if self._turn is not None and self._turn.is_running:
            self._turn.cancel()

    @work(exclusive=True)
    async def run_turn(self, text: str) -> None:
        inp = self.query_one(PromptInput)
        inp.disabled = True

        assistant: Static | None = None
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
                    body = Content.from_markup("[$text-success b]Paimon[/]\n$body", body=buffer)
                    if assistant is None:
                        assistant = self._add(body)
                    else:
                        assistant.update(body)
                    self._scroll()

                elif isinstance(ev, ToolStart):
                    detail = ev.args.get("command") or ev.args.get("path") or json.dumps(ev.args)
                    self._add(
                        Content.from_markup(
                            "[$text-accent b]$name[/]  [$text-muted]$detail[/]",
                            name=ev.name,
                            detail=detail,
                        )
                    )
                    # start fresh assistant/reasoning blocks after a tool runs
                    assistant, buffer = None, ""
                    reasoning, reasoning_buf = None, ""

                elif isinstance(ev, ToolEnd):
                    preview = "\n".join(ev.result.splitlines()[:15])
                    if len(ev.result.splitlines()) > 15:
                        preview += "\n…"
                    classes = "tool-result denied" if ev.denied else "tool-result"
                    self._add(Content(preview or "(no output)"), classes=classes)

                elif isinstance(ev, TurnEnd):
                    pass
        except asyncio.CancelledError:
            self._add(Content.from_markup("[$text-warning]⏹ Interrupted[/]"))
            raise
        except Exception as exc:  # noqa: BLE001 — show errors instead of crashing the UI
            self._add(Content.from_markup("[$text-error b]Error:[/] $body", body=str(exc)))
        finally:
            inp.disabled = False
            inp.focus()


def main() -> None:
    PaimonApp().run()


if __name__ == "__main__":
    main()
