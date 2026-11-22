"""What every tool module shares: the per-agent context and the registry entry."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from ..session import Session


MAX_OUTPUT = 30_000  # truncate tool output sent back to the model


def one_line(value: object, limit: int = 90) -> str:
    """``value`` collapsed onto one line and clipped to ``limit`` characters."""
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def generic_summary(args: dict) -> str:
    """How a call is described when its tool does not say: the first
    argument that reads as its subject, never a raw JSON object."""
    for key in ("question", "prompt", "expression", "code", "job_id"):
        if args.get(key):
            return one_line(args[key])
    path = one_line(args.get("path"))
    if path:
        return path
    count = len(args)
    return f"{count} argument{'s' if count != 1 else ''}"


@dataclass
class ToolContext:
    """Per-agent state a tool needs beyond its own arguments.

    One instance belongs to one ``Agent``, which is what keeps the shell
    overflow files it collects from being readable by every other agent in the
    process (see gate()).
    """

    # Resolved paths of the overflow files this agent's own commands produced.
    shell_outputs: set = field(default_factory=set)
    # The agent's own session log, for search_history/read_history. None where
    # there is no log to search (bare execute_tool calls, tests); the history
    # tools then return a readable error instead of failing.
    session: Session | None = None


# A confirm callback returns True to allow a dangerous tool, False to deny.
ConfirmFn = Callable[[str, dict], Awaitable[bool]]

# A review callback judges a call auto mode will not let through on its own:
# None lets it run, a string is the refusal the model is shown.
ReviewFn = Callable[[str, dict], Awaitable[str | None]]


@dataclass(frozen=True)
class Tool:
    """One tool: what the model is told about it, its executor, and gating class.

    ``params`` is the TypedDict of its arguments. ``run`` takes (args, cwd,
    mode, ctx) and returns the result text, a ToolResult when it has more to
    say than success, or an awaitable of either; it fails the call by raising
    ToolError. It is None for tools the agent loop handles itself
    (write_todos, the agent tools). ``access`` drives gate(): "read" runs freely inside cwd,
    "write" runs freely inside cwd in auto mode, "execute" is held outside
    yolo except for commands safe_command() recognizes as read-only,
    "background" is "execute" with no such exception, "none" is never gated.

    ``parallel`` marks a tool that changes nothing and depends on no other
    call in the same response, so the loop may run it alongside its
    neighbours instead of after them.

    ``summary`` describes a call of the tool on one line from its arguments,
    for the TUI's tool entries; None leaves it to generic_summary().
    """

    description: str
    params: type
    run: Callable[[dict, Path, str, ToolContext], object] | None
    access: str = "none"
    parallel: bool = False
    summary: Callable[[dict], str] | None = None

    def summarize(self, args: dict) -> str:
        """One human line for a call of this tool with ``args``."""
        return (self.summary or generic_summary)(args)


@dataclass(frozen=True, kw_only=True)
class Caller:
    """Who makes a tool call, and on what terms: what gate(), authorize(),
    run_tool() and execute_tool() need besides the call itself.

    Built per call by whoever holds the live values (an Agent's mode, confirm
    hook and tool set all change during its life), so it is never kept.

    ``registry`` narrows the available tools (an agent's own set); None means
    the full REGISTRY. ``safe_commands`` lets shell commands recognized as
    clearly read-only through (see safe_command); False holds every command.
    ``ctx`` carries the calling agent's own state, so gating decisions that
    depend on what this agent did earlier stay scoped to it. ``confirm`` and
    ``review`` are who decides a held call (see authorize).
    """

    cwd: Path
    mode: str
    registry: dict[str, Tool] | None = None
    safe_commands: bool = True
    ctx: ToolContext | None = None
    confirm: ConfirmFn | None = None
    review: ReviewFn | None = None
