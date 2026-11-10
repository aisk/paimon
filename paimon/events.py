"""The typed events a turn is reported in, and a persisted history replayed as them.

``Agent.run`` yields them live; ``replay_events`` rebuilds them from a session
log. Free of the agent itself, so whatever only renders them (a UI, the turn
driver) does not have to import the loop that produces them.
"""

import json
from dataclasses import dataclass, field

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from .session import JOB, SHELL, SUMMARY, job_text, shell_text, synthetic_kind
from .tools.registry import normalize_todos
from .tools.result import Status, ToolResult

# ---- Events yielded by Agent.run -------------------------------------------


@dataclass
class TextDelta:
    text: str


@dataclass
class ReasoningDelta:
    text: str


@dataclass
class ToolStart:
    id: str
    name: str
    args: dict


@dataclass
class ToolEnd:
    id: str
    name: str
    result: str
    # How the call went, the same value its ToolReturnPart.outcome persists:
    # "success", "failed", "denied" or "interrupted". ``detail`` is the short
    # word a renderer shows for anything but success ("exit 2", "timed out"),
    # so none of them has to read it back out of the result text.
    status: Status = "success"
    detail: str = ""
    # What a run_code script called, in call order: one dict per nested call
    # with its tool "name", one-line "detail", "status" (ok, error, denied or
    # cancelled) and "seconds". Empty for every other tool.
    calls: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        # The same default a ToolResult has, for events built directly.
        if self.status != "success" and not self.detail:
            self.detail = self.status

    @property
    def denied(self) -> bool:
        return self.status == "denied"

    @classmethod
    def of(cls, call_id: str, name: str, result: ToolResult,
           calls: list[dict] | None = None) -> "ToolEnd":
        return cls(call_id, name, result.text, result.status, result.detail, calls or [])


@dataclass
class TodosUpdate:
    todos: list[dict]


@dataclass
class RequestStats:
    """Output speed and cache usage of one finished model request, from
    provider-reported usage. ``input_tokens`` includes the cached portion
    (pydantic-ai normalizes providers that report them separately), so the
    cache hit rate is ``cache_read_tokens / input_tokens``. Providers that
    do not report caching leave both cache fields at zero.
    """

    output_tokens: int
    seconds: float
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class ToolBudgetExhausted:
    """The turn hit its tool-call budget: the remaining calls were refused
    with an explicit result and the turn ends without another model request.
    """

    limit: int


@dataclass
class TurnEnd:
    pass


@dataclass
class ContextCompacted:
    tokens_before: int
    tokens_after: int


@dataclass
class ContextCompactionFailed:
    error: str


@dataclass
class ModelRetry:
    """A transient model failure is about to be retried after ``delay`` seconds."""

    attempt: int
    max_attempts: int
    delay: float
    error: str


# ---- Replay-only events (history has no deltas, so these mark structure) ----


@dataclass
class UserInput:
    text: str


@dataclass
class CompactionNotice:
    """A compaction checkpoint encountered while replaying history."""


@dataclass
class JobNotice:
    """What became of an agent or a background command this session started.

    The first line says which job and how it ended; an agent that finished
    follows it with its answer. Not replay-only: it is written into the
    history, so it is both yielded live and rebuilt when the session is
    resumed.
    """

    text: str


@dataclass
class ShellRun:
    """A command the user ran themselves with the "!" prefix.

    Like JobNotice this is not replay-only: the run is persisted as a
    synthetic user message, so it is both yielded when it happens and rebuilt
    when the session is resumed.
    """

    command: str
    output: str


# Everything ``Agent.run`` and ``replay_events`` can yield. Renderers dispatch
# on isinstance; the alias exists so a type checker can flag an unhandled one.
AgentEvent = (
    TextDelta | ReasoningDelta | ToolStart | ToolEnd | TodosUpdate
    | RequestStats | ToolBudgetExhausted | TurnEnd
    | ContextCompacted | ContextCompactionFailed | ModelRetry | UserInput
    | CompactionNotice | JobNotice | ShellRun
)


def parse_args(args: object) -> dict:
    """Tool-call arguments as a dict, tolerating malformed JSON from the model."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args:
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _nested_calls(part: ToolReturnPart) -> list[dict]:
    """The nested calls persisted with a run_code result, tolerating old logs."""
    calls = part.metadata.get("calls") if isinstance(part.metadata, dict) else None
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []


def _returned(part: ToolReturnPart) -> ToolResult:
    """A persisted return part as the result it recorded; see _settle."""
    detail = part.metadata.get("detail") if isinstance(part.metadata, dict) else None
    return ToolResult(str(part.content or "(no output)"), part.outcome,
                      detail if isinstance(detail, str) else "")


def replay_events(messages: list[ModelMessage]) -> list[AgentEvent]:
    """Persisted messages replayed as the events a live ``Agent.run`` yields.

    Lets a UI render resumed history through the same code path as live turns.
    Tools run serially, so each call replays as its ToolStart immediately
    followed by its ToolEnd (looked up in the next request), matching the live
    order — not all starts of a batch first. The persisted ``outcome`` field
    and its detail restore each call's status as it was shown live.
    """
    events: list[AgentEvent] = []
    consumed: set[str] = set()  # tool returns already replayed with their call
    for index, message in enumerate(messages):
        kind = synthetic_kind(message)
        if kind == SUMMARY:
            events.append(CompactionNotice())
            continue
        if kind == JOB:
            events.append(JobNotice(job_text(message)))
            continue
        if kind == SHELL:
            events.append(ShellRun(*shell_text(message)))
            continue
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str) and part.content:
                    events.append(UserInput(part.content))
                elif (isinstance(part, ToolReturnPart)
                      and part.tool_call_id not in consumed
                      and part.tool_name != "write_todos"):
                    # A return whose call is gone (compaction cut mid-turn):
                    # better an unpaired result than a silent hole.
                    events.append(ToolEnd.of(part.tool_call_id, part.tool_name,
                                             _returned(part), _nested_calls(part)))
        elif isinstance(message, ModelResponse):
            following = messages[index + 1] if index + 1 < len(messages) else None
            returns: dict[str, ToolReturnPart] = {}
            if isinstance(following, ModelRequest):
                returns = {part.tool_call_id: part for part in following.parts
                           if isinstance(part, ToolReturnPart)}
            for part in message.parts:
                if isinstance(part, ThinkingPart) and part.content:
                    events.append(ReasoningDelta(part.content))
                elif isinstance(part, TextPart) and part.content:
                    events.append(TextDelta(part.content))
                elif isinstance(part, ToolCallPart):
                    args = parse_args(part.args)
                    result = returns.get(part.tool_call_id)
                    if result is not None:
                        consumed.add(part.tool_call_id)
                    if part.tool_name == "write_todos":
                        todos = normalize_todos(args.get("todos"))
                        # Live, a write_todos that actually ran yields only a
                        # TodosUpdate; one that was rejected or never executed
                        # yields a normal failed call.
                        if todos is not None and (result is None or result.outcome == "success"):
                            events.append(TodosUpdate(todos))
                            continue
                    events.append(ToolStart(part.tool_call_id, part.tool_name, args))
                    if result is not None:
                        events.append(ToolEnd.of(part.tool_call_id, part.tool_name,
                                                 _returned(result), _nested_calls(result)))
    return events
