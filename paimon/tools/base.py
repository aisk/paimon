"""What every tool module shares: the per-agent context and the registry entry."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..session import Session


MAX_OUTPUT = 30_000  # truncate tool output sent back to the model


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
    """

    description: str
    params: type
    run: Callable[[dict, Path, str, ToolContext], object] | None
    access: str = "none"
