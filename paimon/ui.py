"""Reusable UI components for the Paimon TUI."""

from textual.content import Content
from textual.widgets import Markdown, Static


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
