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

from .diff import locate_line, render_diff


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


class RecapMessage(Markdown):
    """A recap Paimon offered by itself, after the user went quiet.

    Nobody asked for it, so it is styled to be read past: a muted body behind
    an accent rule, distinct from the user's own block and from the plan panel,
    which both carry a $primary rule. Rendered as markdown like an answer, so
    whatever emphasis the model reached for comes out right; the app CSS keeps
    every block inside it muted and quiet.
    """

    HEADER = "⟲ While you were away"

    def __init__(self, body: str) -> None:
        super().__init__(f"**{self.HEADER}**\n\n{body}", classes="recap")


class FoldedText(Static):
    """Long text folded behind a line-count stub; click toggles the full body.

    Bodies of at most one line render as-is with no toggle.
    """

    def __init__(
        self, body: str, *, classes: str = "", expanded: bool = False, label: str = ""
    ) -> None:
        self._full = body
        self._expanded = expanded
        self._label = label
        super().__init__(self._body(), classes=classes)

    @property
    def _foldable(self) -> bool:
        return len(self._full.splitlines()) > 1

    def _body(self) -> Content:
        if not self._foldable:
            return Content(self._full)
        # No explicit color: the stub inherits the host widget's (dim) color.
        if self._expanded:
            return Content.from_markup(
                "$body\n[i]click to collapse[/]", body=self._full
            )
        lines = str(len(self._full.splitlines()))
        if self._label:
            return Content.from_markup(
                "[i]… $label · $lines lines — click to expand[/]",
                label=self._label,
                lines=lines,
            )
        return Content.from_markup(
            "[i]… $lines lines — click to expand[/]", lines=lines
        )

    def set_text(self, body: str) -> None:
        self._full = body
        self.update(self._body())

    def collapse(self) -> None:
        if self._expanded:
            self._expanded = False
            self.update(self._body())

    def on_click(self) -> None:
        if not self._foldable:
            return
        self._expanded = not self._expanded
        self.update(self._body())


class ToolResult(FoldedText):
    """Tool output folded to a line-count stub; click expands the full text."""

    def __init__(self, result: str, *, label: str = "", denied: bool = False) -> None:
        super().__init__(
            result or "(no output)",
            classes="tool-result denied" if denied else "tool-result",
            label=label,
        )


class ToolCall(FoldedText):
    """A tool invocation line; multi-line detail folds down to its first line."""

    def __init__(self, name: str, detail: str) -> None:
        self._name = name
        super().__init__(detail, classes="tool-call")

    def _body(self) -> Content:
        if not self._foldable:
            return Content.from_markup(
                "[$text-accent b]$name[/]  [$text-muted]$detail[/]",
                name=self._name,
                detail=self._full,
            )
        if self._expanded:
            return Content.from_markup(
                "[$text-accent b]$name[/]  [$text-muted]$detail[/]\n[i]click to collapse[/]",
                name=self._name,
                detail=self._full,
            )
        lines = self._full.splitlines()
        return Content.from_markup(
            "[$text-accent b]$name[/]  [$text-muted]$first[/] [i]… +$more lines — click to expand[/]",
            name=self._name,
            first=lines[0],
            more=str(len(lines) - 1),
        )


class EditCall(Vertical):
    """An edit_file invocation with its diff shown inline.

    Unlike other tool calls the change itself is the interesting part, so the
    diff starts expanded; clicking anywhere on the widget folds it behind the
    header line and back.
    """

    _CLIP = 1_500

    def __init__(self, path: str, old: str, new: str, *,
                 start_line: int | None = None) -> None:
        self._path = path
        self._old = old
        self._new = new
        self._start_line = start_line
        self._expanded = True
        super().__init__(classes="tool-call edit-call")

    @staticmethod
    def _clip(text: str, limit: int = _CLIP) -> str:
        return text if len(text) <= limit else text[:limit] + " …"

    def compose(self) -> ComposeResult:
        # built here rather than in __init__: the diff colors follow the app
        # theme, and self.app only exists once the widget is mounted
        diff = render_diff(
            self._clip(self._old), self._clip(self._new), path=self._path,
            start_line=self._start_line,
            theme=self.app.theme or "",
            dark=self.app.current_theme.dark,
        )
        yield Static(self._header(), classes="edit-call-header")
        yield Static(diff, classes="edit-call-diff")

    def _header(self) -> Content:
        if self._expanded:
            return Content.from_markup(
                "[$text-accent b]edit_file[/]  [$text-muted]$path[/] [i]click to collapse[/]",
                path=self._path,
            )
        return Content.from_markup(
            "[$text-accent b]edit_file[/]  [$text-muted]$path[/] [i]… diff — click to expand[/]",
            path=self._path,
        )

    def on_click(self) -> None:
        self._expanded = not self._expanded
        self.query_one(".edit-call-diff", Static).display = self._expanded
        self.query_one(".edit-call-header", Static).update(self._header())


class PromptInput(TextArea):
    """Multi-line prompt editor. Enter submits; Shift+Enter / Ctrl+J insert a newline.

    Up on the first line / Down on the last line walk previously submitted
    prompts, bash-style; walking past the newest entry restores the draft.
    "/" in an empty editor opens the command palette.
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
        if event.character == "/" and not self.text:
            event.prevent_default()
            event.stop()
            self.app.action_command_palette()
            return
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

    Resolves its future with "allow" or "deny". Shows what would actually
    run/change, not just a path.
    Navigate with Up/Down or 1-2, Enter to confirm, Esc to deny.
    """

    _CLIP = 1_500
    _OPTIONS = [
        ("allow", "Yes"),
        ("deny", "No (esc)"),
    ]

    def __init__(self, tool_name: str, args: dict, future: "asyncio.Future[str]") -> None:
        # No ID: several panes can have a panel up at once, and a shared ID
        # would make an app-wide query resolve to whichever one is first.
        super().__init__(classes="confirm-panel")
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
        # Focusing is the caller's job: a panel in a background pane must not
        # take the keyboard, and focusable() only looks at visibility.
        self._render_options()

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
        elif key in ("1", "2"):
            self._resolve(self._OPTIONS[int(key) - 1][0])
        elif key == "y":
            self._resolve("allow")
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

    def _detail(self) -> RenderableType:
        args = self.args
        if self.tool_name == "shell":
            return Content(self._clip(str(args.get("command") or "")))
        if self.tool_name == "run_background":
            # What makes this one different from shell is that saying yes
            # leaves something running, so the panel leads with that.
            return Content.from_markup(
                "[$text-muted]Runs in its own tab until it exits or you close it:[/]\n\n"
                "$command\n\n[$text-muted]$description[/]",
                command=self._clip(str(args.get("command") or "")),
                description=self._clip(str(args.get("description") or ""), 200))
        if self.tool_name == "write_file":
            path = str(args.get("path") or "")
            content = self._clip(str(args.get("content") or ""))
            try:
                existing = Path(path).read_text() if path else ""
            except OSError:
                existing = ""
            if existing:
                diff = render_diff(
                    self._clip(existing), content, path=path, start_line=1,
                    theme=self.app.theme or "",
                    dark=self.app.current_theme.dark,
                )
                return Group(Text(path), Text(), diff)
            return Content.from_markup(
                "$path\n\n[$text-muted]$content[/]", path=path, content=content
            )
        if self.tool_name == "edit_file":
            path = str(args.get("path") or "")
            old = str(args.get("old_string") or "")
            new = str(args.get("new_string") or "")
            diff = render_diff(
                self._clip(old),
                self._clip(new),
                path=path,
                start_line=locate_line(path, old, new),
                theme=self.app.theme or "",
                dark=self.app.current_theme.dark,
            )
            return Group(Text(path), Text(), diff)
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
        if self.tool_name == "start_new_session":
            # reviewing the full handoff prompt is the point of this confirmation,
            # so clip far later than usual; the detail container scrolls
            return Content.from_markup(
                "[$text-muted]Ends this session and starts a fresh one with this first message:[/]\n\n$prompt",
                prompt=self._clip(str(args.get("prompt") or ""), 5_000),
            )
        return Content(self._clip(json.dumps(args, ensure_ascii=False)))
