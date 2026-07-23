"""Reusable UI components for the Paimon TUI."""

import asyncio
import json
from pathlib import Path

from rich.console import Group, RenderableType
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.widgets import Markdown, Static, TextArea

from .diff import render_diff


class UserMessage(Static):
    """Visually distinct user prompt."""

    def __init__(self, body: str) -> None:
        super().__init__(Content(body), classes="user-message")


class AssistantMessage(Markdown):
    """Markdown-rendered assistant response.

    Only the first text block of a turn carries the Paimon heading; follow-up
    blocks (after tool calls) continue without repeating it.
    """

    def __init__(self, body: str, *, heading: bool = True) -> None:
        super().__init__(self._format_body(body, heading), classes="assistant")

    @staticmethod
    def _format_body(body: str, heading: bool) -> str:
        return f"**Paimon**\n\n{body}" if heading else body


class ToolResult(Static):
    """Tool output shown as a preview; click toggles the full text when truncated."""

    PREVIEW_LINES = 15

    def __init__(self, result: str, *, denied: bool = False) -> None:
        self._full = result or "(no output)"
        self._expanded = False
        classes = "tool-result denied" if denied else "tool-result"
        super().__init__(self._body(), classes=classes)

    @property
    def _hidden_lines(self) -> int:
        return max(0, len(self._full.splitlines()) - self.PREVIEW_LINES)

    def _body(self) -> Content:
        if self._expanded:
            return Content.from_markup(
                "$body\n[$text-muted i]click to collapse[/]", body=self._full
            )
        if not self._hidden_lines:
            return Content(self._full)
        preview = "\n".join(self._full.splitlines()[: self.PREVIEW_LINES])
        return Content.from_markup(
            "$body\n[$text-muted i]… +$more lines — click to expand[/]",
            body=preview,
            more=str(self._hidden_lines),
        )

    def on_click(self) -> None:
        if not self._hidden_lines:
            return
        self._expanded = not self._expanded
        self.update(self._body())


class PromptInput(TextArea):
    """Multi-line prompt editor. Enter submits; Shift+Enter / Ctrl+J insert a newline.

    Up on the first line / Down on the last line walk previously submitted
    prompts, bash-style; walking past the newest entry restores the draft.
    """

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_index: int | None = None
        self._draft = ""

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self._remember(text)
                self.post_message(self.Submitted(text))
            return
        if event.key in ("ctrl+j", "shift+enter"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        if event.key == "up" and self._history and self.cursor_location[0] == 0:
            event.prevent_default()
            event.stop()
            self._history_prev()
            return
        if (
            event.key == "down"
            and self._history_index is not None
            and self.cursor_location[0] == self.document.line_count - 1
        ):
            event.prevent_default()
            event.stop()
            self._history_next()
            return
        await super()._on_key(event)

    def _remember(self, text: str) -> None:
        if not self._history or self._history[-1] != text:
            self._history.append(text)
        self._history_index = None
        self._draft = ""

    def _recall(self, text: str) -> None:
        self.load_text(text)
        self.move_cursor(self.document.end)

    def _history_prev(self) -> None:
        if self._history_index is None:
            self._draft = self.text
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        else:
            return
        self._recall(self._history[self._history_index])

    def _history_next(self) -> None:
        if self._history_index is None:
            return
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self._recall(self._history[self._history_index])
        else:
            self._history_index = None
            self._recall(self._draft)


class ConfirmPanel(Vertical, can_focus=True):
    """Inline confirmation for a dangerous tool call, shown in place of the prompt.

    Resolves its future with "allow", "always" (allow this tool for the rest of
    the session) or "deny". Shows what would actually run/change, not just a path.
    Navigate with Up/Down or 1-3, Enter to confirm, Esc to deny.
    """

    _CLIP = 1_500
    _OPTIONS = [
        ("allow", "Yes"),
        ("always", "Yes, and don't ask again for this tool this session"),
        ("deny", "No (esc)"),
    ]

    def __init__(self, tool_name: str, args: dict, future: "asyncio.Future[str]") -> None:
        super().__init__(id="confirm-panel")
        self.tool_name = tool_name
        self.args = args
        self._future = future
        self._selected = 0

    def compose(self) -> ComposeResult:
        yield Static(
            Content.from_markup(
                "[b]Paimon needs permission![/]  [$text-warning b]$tool[/]", tool=self.tool_name
            )
        )
        with VerticalScroll(id="confirm-detail"):
            yield Static(self._detail())
        yield Static(id="confirm-options")

    def on_mount(self) -> None:
        self._render_options()
        self.focus()

    def _render_options(self) -> None:
        lines = []
        for i, (_, label) in enumerate(self._OPTIONS):
            if i == self._selected:
                lines.append(f"[$text-accent b]❯ {i + 1}. {label}[/]")
            else:
                lines.append(f"[$text-muted]  {i + 1}. {label}[/]")
        self.query_one("#confirm-options", Static).update(Content.from_markup("\n".join(lines)))

    def on_key(self, event: events.Key) -> None:
        key = event.key
        if key in ("up", "k"):
            self._selected = (self._selected - 1) % len(self._OPTIONS)
            self._render_options()
        elif key in ("down", "j", "tab"):
            self._selected = (self._selected + 1) % len(self._OPTIONS)
            self._render_options()
        elif key == "enter":
            self._resolve(self._OPTIONS[self._selected][0])
        elif key in ("1", "2", "3"):
            self._resolve(self._OPTIONS[int(key) - 1][0])
        elif key == "y":
            self._resolve("allow")
        elif key == "a":
            self._resolve("always")
        elif key in ("n", "escape"):
            self._resolve("deny")
        else:
            return
        event.prevent_default()
        event.stop()

    def _resolve(self, verdict: str) -> None:
        if not self._future.done():
            self._future.set_result(verdict)

    @staticmethod
    def _clip(text: str, limit: int = _CLIP) -> str:
        return text if len(text) <= limit else text[:limit] + " …"

    def _diff_width(self) -> int:
        # workspace margins + panel padding + border eat ~10 cells
        return max(60, self.app.size.width - 10)

    def _detail(self) -> RenderableType:
        args = self.args
        if self.tool_name == "bash":
            return Content(self._clip(str(args.get("command") or "")))
        if self.tool_name == "write_file":
            path = str(args.get("path") or "")
            content = self._clip(str(args.get("content") or ""))
            try:
                existing = Path(path).read_text() if path else ""
            except OSError:
                existing = ""
            if existing:
                diff = render_diff(self._clip(existing), content, width=self._diff_width())
                return Group(Text(path), Text(), diff)
            return Content.from_markup(
                "$path\n\n[$text-muted]$content[/]", path=path, content=content
            )
        if self.tool_name == "edit_file":
            diff = render_diff(
                self._clip(str(args.get("old_string") or "")),
                self._clip(str(args.get("new_string") or "")),
                width=self._diff_width(),
            )
            return Group(Text(str(args.get("path") or "")), Text(), diff)
        if self.tool_name == "read_file":
            return Content.from_markup(
                "$path\n[$text-muted]outside the working directory[/]",
                path=str(args.get("path") or ""),
            )
        if self.tool_name == "glob":
            return Content.from_markup(
                "$pattern in $path\n[$text-muted]outside the working directory[/]",
                pattern=str(args.get("pattern") or ""),
                path=str(args.get("path") or ""),
            )
        return Content(self._clip(json.dumps(args, ensure_ascii=False)))
