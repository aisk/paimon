"""Reusable UI components for the Paimon TUI."""

import asyncio
import difflib
import json
from collections import Counter
from pathlib import Path

from rich.console import Group, RenderableType
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, Markdown, Static, TextArea

from .diff import locate_line, render_diff
from .tools import failure, resolve_path


def abbreviate(text: str, limit: int) -> str:
    """Head and tail of an oversized text, with the elision spelled out.

    Never a bare prefix: in a confirmation the dangerous part of a long
    command or write is as likely to sit at its end (`… && rm -rf /`), so the
    tail must be part of what the user approves, and the marker names how much
    was left out.
    """
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit - len(head)):]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n[… {omitted:,} of {len(text):,} characters not shown …]\n{tail}"


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

    def __init__(self, result: str, *, label: str = "", denied: bool = False,
                 expanded: bool = False) -> None:
        super().__init__(
            result or "(no output)",
            classes="tool-result denied" if denied else "tool-result",
            label=label,
            expanded=expanded,
        )


class ToolCall(FoldedText):
    """A tool invocation line; multi-line detail folds down to its first line."""

    def __init__(self, name: str, detail: str, *, expanded: bool = False) -> None:
        # DOMNode owns ``_name`` internally, so keep the tool name separate or
        # a later update after mounting would redraw it as None.
        self._tool_name = name
        super().__init__(detail, classes="tool-call", expanded=expanded)

    def _body(self) -> Content:
        if not self._foldable:
            return Content.from_markup(
                "[$text-accent b]$name[/]  [$text-muted]$detail[/]",
                name=self._tool_name,
                detail=self._full,
            )
        if self._expanded:
            return Content.from_markup(
                "[$text-accent b]$name[/]  [$text-muted]$detail[/]\n[i]click to collapse[/]",
                name=self._tool_name,
                detail=self._full,
            )
        lines = self._full.splitlines()
        return Content.from_markup(
            "[$text-accent b]$name[/]  [$text-muted]$first[/] [i]… +$more lines — click to expand[/]",
            name=self._tool_name,
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
        return abbreviate(text, limit)

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


_TOOL_LABELS = {
    "read_file": "read",
    "write_file": "write",
    "edit_file": "edit",
    "web_search": "search",
    "run_background": "background",
    "textual_inspect": "inspect",
    "textual_screenshot": "screenshot",
    "textual_apply_css": "css",
    "textual_eval": "eval",
    "textual_exec": "exec",
}


def _one_line(value: object, limit: int = 90) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def tool_call_summary(name: str, args: dict) -> str:
    """A compact human summary, never a raw fallback JSON object."""
    path = _one_line(args.get("path"))
    if name == "read_file":
        parts = [path or "(path missing)"]
        if args.get("offset") is not None:
            parts.append(f"from line {args['offset']}")
        if args.get("limit") is not None:
            parts.append(f"up to {args['limit']} lines")
        return " · ".join(parts)
    if name == "write_file":
        lines = len(str(args.get("content") or "").splitlines())
        return f"{path or '(path missing)'} · {lines} lines"
    if name == "edit_file":
        old = str(args.get("old_string") or "").splitlines()
        new = str(args.get("new_string") or "").splitlines()
        added = removed = 0
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new).get_opcodes():
            if tag in ("replace", "delete"):
                removed += i2 - i1
            if tag in ("replace", "insert"):
                added += j2 - j1
        return f"{path or '(path missing)'} · +{added} −{removed}"
    if name in ("shell", "run_background"):
        return _one_line(args.get("command")) or "(empty command)"
    if name == "grep":
        pattern = _one_line(args.get("pattern"))
        return f"{pattern} in {path}" if path and path != "." else pattern
    if name == "glob":
        pattern = _one_line(args.get("pattern"))
        return f"{pattern} in {path}" if path and path != "." else pattern
    if name == "web_search":
        return _one_line(args.get("query"))
    if name == "textual_inspect":
        return _one_line(args.get("selector")) or "active screen"
    for key in ("question", "prompt", "expression", "code", "job_id"):
        if args.get(key):
            return _one_line(args[key])
    if path:
        return path
    count = len(args)
    return f"{count} argument{'s' if count != 1 else ''}"


def _result_summary(result: str, denied: bool) -> tuple[str, str]:
    """Return (status word, CSS class) for a finished entry."""
    if denied:
        return "denied", "-denied"
    problem = failure(result)
    if problem is not None:
        return problem, "-failed"
    if result.rstrip().endswith("(exit code 0)"):
        return "✓", "-done"
    lines = len(result.splitlines())
    if lines > 1:
        return f"{lines} lines", "-done"
    return "✓", "-done"


def _seconds(call: dict) -> float:
    """How long a nested call took; 0 for a log that does not say."""
    seconds = call.get("seconds")
    return seconds if isinstance(seconds, (int, float)) else 0.0


def nested_calls(calls: list[dict]) -> Content:
    """What a run_code script called, one line per call."""
    lines, values = [], {}
    for index, call in enumerate(calls):
        status = str(call.get("status"))
        style = "$text-muted" if status == "ok" else "$text-warning"
        values[f"name{index}"] = _TOOL_LABELS.get(call.get("name"), str(call.get("name")))
        values[f"detail{index}"] = _one_line(call.get("detail"))
        values[f"status{index}"] = f"{'✓' if status == 'ok' else status} {_seconds(call):.2f}s"
        lines.append(f"[$text-accent]$name{index}[/]  $detail{index}  [{style}]$status{index}[/]")
    return Content.from_markup("\n".join(lines), **values)


class _ToggleLine(Static, can_focus=True):
    """A one-line disclosure control used by groups and their entries."""

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.parent.toggle()

    def on_key(self, event: events.Key) -> None:
        if event.key in ("enter", "space"):
            event.stop()
            event.prevent_default()
            self.parent.toggle()


class ToolEntry(Vertical):
    """One tool call: a summary row with raw arguments/output one click away."""

    def __init__(self, name: str, args: dict, reasoning: FoldedText | None = None,
                 cwd: Path | None = None) -> None:
        self.tool_name = name
        self.cwd = cwd
        self.args = args
        self.summary = tool_call_summary(name, args)
        self._reasoning = reasoning
        self._expanded = False
        self._status = "running…"
        self._status_class = "-running"
        super().__init__(classes="tool-entry -running")

    def compose(self) -> ComposeResult:
        yield _ToggleLine(self._header(), classes="tool-entry-header")
        children: list = []
        if self._reasoning is not None:
            children.append(self._reasoning)
        if self.tool_name == "edit_file":
            path = str(self.args.get("path") or "")
            old = str(self.args.get("old_string") or "")
            new = str(self.args.get("new_string") or "")
            children.append(EditCall(
                path, old, new,
                start_line=locate_line(path, old, new, cwd=self.cwd),
            ))
        else:
            detail = json.dumps(self.args, ensure_ascii=False, indent=2, default=str)
            children.append(ToolCall(self.tool_name, detail, expanded=True))
        body = Vertical(*children, classes="tool-entry-detail")
        body.display = False
        yield body

    def _header(self) -> Content:
        arrow = "▼" if self._expanded else "›"
        label = _TOOL_LABELS.get(self.tool_name, self.tool_name)
        return Content.from_markup(
            "[$text-muted]$arrow[/] [$text-accent]$name[/]  $summary  [$status]$status[/]",
            arrow=arrow,
            name=label,
            summary=self.summary,
            status=self._status,
        )

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self.query_one(".tool-entry-detail", Vertical).display = self._expanded
        self.query_one(".tool-entry-header", Static).update(self._header())

    async def finish(self, result: str, *, label: str = "", denied: bool = False,
                     calls: list[dict] | None = None) -> None:
        self.remove_class(self._status_class)
        self._status, self._status_class = _result_summary(result, denied)
        self.add_class(self._status_class)
        detail = self.query_one(".tool-entry-detail", Vertical)
        if calls:
            # Between the script and its output, which is where they happened.
            await detail.mount(Static(nested_calls(calls), classes="tool-nested"))
        await detail.mount(
            ToolResult(result, label=label, denied=denied, expanded=True)
        )
        self.query_one(".tool-entry-header", Static).update(self._header())


class ToolGroup(Vertical):
    """A consecutive burst of tool calls collapsed to one activity line."""

    def __init__(self, cwd: Path | None = None) -> None:
        self.cwd = cwd
        self.entries: list[ToolEntry] = []
        self._expanded = False
        super().__init__(classes="tool-group")

    def compose(self) -> ComposeResult:
        yield _ToggleLine(self._header(), classes="tool-group-header")
        body = Vertical(classes="tool-group-body")
        body.display = False
        yield body

    def _header(self) -> Content:
        arrow = "▼" if self._expanded else "▶"
        counts = Counter(_TOOL_LABELS.get(entry.tool_name, entry.tool_name) for entry in self.entries)
        parts = [f"{name} ×{count}" if count > 1 else name for name, count in counts.items()]
        if len(parts) > 4:
            parts = [*parts[:4], f"+{len(parts) - 4} kinds"]
        running = sum(entry.has_class("-running") for entry in self.entries)
        failed = sum(entry.has_class("-failed") for entry in self.entries)
        denied = sum(entry.has_class("-denied") for entry in self.entries)
        if running:
            status = "running…"
            style = "$text-accent"
        elif failed or denied:
            details = []
            if failed:
                details.append(f"{failed} failed")
            if denied:
                details.append(f"{denied} denied")
            status = " · ".join(details)
            style = "$warning"
        else:
            status = "✓"
            style = "$success"
        summary = " · ".join(parts)
        return Content.from_markup(
            "[$text-muted]$arrow[/] [$text-accent b]Tools[/]  $count calls"
            f"[$text-muted] · $summary[/]  [{style}]$status[/]",
            arrow=arrow,
            count=str(len(self.entries)),
            summary=summary,
            status=status,
        )

    async def add_call(self, name: str, args: dict, reasoning: FoldedText | None = None) -> ToolEntry:
        entry = ToolEntry(name, args, reasoning, self.cwd)
        self.entries.append(entry)
        await self.query_one(".tool-group-body", Vertical).mount(entry)
        self.query_one(".tool-group-header", Static).update(self._header())
        return entry

    def refresh_header(self) -> None:
        self.query_one(".tool-group-header", Static).update(self._header())

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self.query_one(".tool-group-body", Vertical).display = self._expanded
        self.query_one(".tool-group-header", Static).update(self._header())


class PromptInput(TextArea):
    """Multi-line prompt editor. Enter submits; Shift+Enter / Ctrl+J insert a newline.

    Up on the first line / Down on the last line walk previously submitted
    prompts, bash-style; walking past the newest entry restores the draft.
    "/" in an empty editor opens the command palette, and a leading "!" turns
    the line into a shell command the pane runs on submit.
    """

    INPUT_HELP = "Enter send · Ctrl+J newline · / commands · Esc interrupt · Shift+Tab mode"

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

    def on_text_area_changed(self) -> None:
        """Mark the editor while it holds a shell command, so the border says
        what Enter is about to do. The "!" stays in the text rather than being
        held as a mode: history, paste and the cursor all keep working on the
        line as typed."""
        self.set_class(self.text.startswith("!"), "bash")


class BlockingPanel(Vertical, can_focus=True):
    """A panel a turn is waiting on, shown in its pane until it is answered.

    What the confirm and question panels have in common for the pane. In a
    conversation it takes the prompt's place, and the keyboard with it.
    """

    def _focus_control(self, widget: Widget) -> None:
        for pane in self.app.panes:
            if pane in self.ancestors:
                self.app.focus_pane_widget(pane, widget)
                break

    def _scroll_detail_key(self, event: events.Key) -> bool:
        focused = self.app.focused
        if not isinstance(focused, VerticalScroll) or self not in focused.ancestors:
            return False
        amounts = {"up": -1, "k": -1, "down": 1, "j": 1,
                   "pageup": -focused.size.height, "pagedown": focused.size.height}
        if event.key in amounts:
            focused.scroll_relative(y=amounts[event.key], animate=False)
        elif event.key == "home":
            focused.scroll_home(animate=False)
        elif event.key == "end":
            focused.scroll_end(animate=False)
        else:
            return False
        event.prevent_default()
        event.stop()
        return True


class ConfirmPanel(BlockingPanel):
    """Inline confirmation for a dangerous tool call, shown in the asking agent's pane.

    Resolves its future with "allow" or "deny". Shows what would actually
    run/change, not just a path: the detail scrolls, and content too large to
    render is shown head and tail with the elision named — never a silent
    prefix, since what gets approved is the whole operation.
    Navigate with Up/Down or 1-2, Enter to confirm, Esc to deny.
    """

    _CLIP = 20_000
    _OPTIONS = [
        ("allow", "Yes"),
        ("deny", "No (esc)"),
    ]

    def __init__(self, tool_name: str, args: dict, future: "asyncio.Future[str]",
                 cwd: Path | None = None) -> None:
        # No ID: several panes can have a panel up at once, and a shared ID
        # would make an app-wide query resolve to whichever one is first.
        super().__init__(classes="blocking-panel confirm-panel")
        self.tool_name = tool_name
        self.args = args
        # The agent's cwd: previews resolve paths against it, exactly like the
        # execution will, so the file shown is the file touched.
        self._cwd = cwd
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
        if self._scroll_detail_key(event):
            return
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
        return abbreviate(text, limit)

    def _preview_path(self, path: str) -> Path:
        return resolve_path(path, self._cwd) if self._cwd is not None and path else Path(path)

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
                existing = self._preview_path(path).read_text(encoding="utf-8", errors="replace") if path else ""
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
                start_line=locate_line(path, old, new, cwd=self._cwd),
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


class QuestionPanel(BlockingPanel):
    """Inline question from the model (ask_user), shown in place of the prompt.

    Resolves its future with the chosen option or the typed answer, or None
    when dismissed. The choices are numbered; the last entry always lets the
    user type something else, and with no choices that is the only entry, so
    the answer box has the keyboard from the start.
    Navigate with Up/Down or 1-9, Enter to pick, Esc to dismiss.
    """

    def __init__(self, question: str, options: list[str],
                 future: "asyncio.Future[str | None]") -> None:
        super().__init__(classes="blocking-panel question-panel")
        self.question = question
        self.options = list(options)
        self._future = future
        self._selected = 0

    @property
    def _other(self) -> int:
        """Index of the free-text entry, one past the last option."""
        return len(self.options)

    def compose(self) -> ComposeResult:
        yield Static(Content.from_markup("[b]Paimon has a question[/]"))
        with VerticalScroll(id="question-detail"):
            yield Static(Content(self.question))
        yield Static(id="question-options")
        yield Input(placeholder="Type your answer", id="question-input")

    def on_mount(self) -> None:
        # Focusing is the caller's job, as for ConfirmPanel.
        self._render_options()

    def on_focus(self) -> None:
        # With the free-text entry selected the panel hands the keyboard on to
        # the answer box, so a question without options is ready to type into.
        if self._selected == self._other:
            self._focus_control(self.query_one(Input))

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if (event.widget is self.app.focused and isinstance(event.widget, Input)
                and self._selected != self._other):
            self._selected = self._other
            self._render_options()

    def _render_options(self) -> None:
        labels = [*self.options, "Something else (type below)" if self.options else "Type your answer"]
        lines = []
        for i, label in enumerate(labels):
            if i == self._selected:
                lines.append(Content.from_markup("[$text-accent b]❯ $n. $label[/]", n=str(i + 1), label=label))
            else:
                lines.append(Content.from_markup("[$text-muted]  $n. $label[/]", n=str(i + 1), label=label))
        self.query_one("#question-options", Static).update(Content("\n").join(lines))

    def _select(self, index: int) -> None:
        self._selected = index % (self._other + 1)
        self._render_options()
        if self._selected == self._other:
            self._focus_control(self.query_one(Input))
        else:
            self._focus_control(self)

    def on_key(self, event: events.Key) -> None:
        if self._scroll_detail_key(event):
            return
        key = event.key
        typing = isinstance(self.app.focused, Input)
        if key == "up" or (key == "k" and not typing):
            self._select(self._selected - 1)
        elif key in ("down", "tab") or (key == "j" and not typing):
            self._select(self._selected + 1)
        elif key == "enter" and not typing:
            if self._selected == self._other:
                self._focus_control(self.query_one(Input))
            else:
                self._resolve(self.options[self._selected])
        elif key.isdigit() and not typing and 1 <= int(key) <= self._other + 1:
            self._select(int(key) - 1)
            if self._selected != self._other:
                self._resolve(self.options[self._selected])
        elif key == "escape":
            self._resolve(None)
        else:
            return
        event.prevent_default()
        event.stop()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        answer = event.value.strip()
        if answer:
            self._resolve(answer)

    def _resolve(self, answer: str | None) -> None:
        if not self._future.done():
            self._future.set_result(answer)
