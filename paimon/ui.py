"""Reusable UI components for the Paimon TUI."""

from textual.content import Content
from textual.widgets import Markdown, Static


class UserMessage(Static):
    """Visually distinct user prompt."""

    def __init__(self, body: str) -> None:
        super().__init__(
            Content.from_markup("[$text-muted]You[/]\n$body", body=body),
            classes="user-message",
        )


class AssistantMessage(Markdown):
    """Markdown-rendered assistant response with a consistent heading."""

    def __init__(self, body: str) -> None:
        super().__init__(self._format_body(body), classes="assistant")

    def update_body(self, body: str) -> None:
        self.update(self._format_body(body))

    @staticmethod
    def _format_body(body: str) -> str:
        return f"**Paimon**\n\n{body}"
