"""What a tool call came to: the text the model reads, and whether it worked.

Standard library only: the grep worker raises ToolError, and imports nothing
it would have to pay for at every search.
"""

from dataclasses import dataclass
from typing import Literal

# Exactly the values pydantic-ai's ToolReturnPart.outcome takes, so a status
# is persisted as the part's outcome without translation. It is not what the
# model is sent: Agent._context sends a failed result as a plain one, so the
# text reads the same on every provider (see _failures_as_text).
Status = Literal["success", "failed", "denied", "interrupted"]


@dataclass(frozen=True)
class ToolResult:
    """One tool call's result.

    ``text`` is what the model is shown, and still says in words what went
    wrong ("Error: ...", a command's exit status line): the status travels
    beside it so nothing has to read it back out of the text. ``detail`` is a
    short word for a renderer ("exit 2", "timed out"); it defaults to the
    status itself whenever the call did not succeed, and is empty when it did.
    """

    text: str
    status: Status = "success"
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status != "success" and not self.detail:
            object.__setattr__(self, "detail", self.status)

    @property
    def ok(self) -> bool:
        return self.status == "success"


class ToolError(Exception):
    """Raised by a tool executor to fail the call with ``text`` as its result.

    The executors return plain text when they succeed; raising this is how
    one says it did not, without having to wrap every success.
    """

    def __init__(self, text: str, detail: str = "failed") -> None:
        super().__init__(text)
        self.text = text
        self.detail = detail

    def result(self) -> ToolResult:
        return ToolResult(self.text, "failed", self.detail)
