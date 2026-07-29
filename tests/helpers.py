"""Shared fixtures for the test suite."""

import dataclasses
from pathlib import Path

from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from paimon import agent as agent_module
from paimon.session import Session

# Constructor arguments for one instance of every event ``Agent.run`` and
# ``replay_events`` can yield. Every renderer is checked against this list, so
# a new event has to be given a sample here before the suite will pass.
EVENT_SAMPLES = {
    "TextDelta": ("hi",),
    "ReasoningDelta": ("hm",),
    "ToolStart": ("call-1", "shell", {"command": "ls"}),
    "ToolEnd": ("call-1", "shell", "out"),
    "TodosUpdate": ([{"content": "a", "status": "pending"}],),
    "SessionHandoff": ("do the next thing",),
    "TurnEnd": (),
    "ContextCompacted": (10, 5),
    "ContextCompactionFailed": ("boom",),
    "ModelRetry": (1, 4, 2.0, "HTTP 429"),
    "UserInput": ("hi",),
    "CompactionNotice": (),
}

# Events a renderer may legitimately draw nothing for.
SILENT_EVENTS = {"TurnEnd", "SessionHandoff"}


def agent_events() -> list[object]:
    """One instance of every event dataclass defined in paimon.agent."""
    events = []
    for name in dir(agent_module):
        attribute = getattr(agent_module, name)
        if not (isinstance(attribute, type) and dataclasses.is_dataclass(attribute)):
            continue
        if attribute.__module__ != agent_module.__name__:
            continue
        if name not in EVENT_SAMPLES:
            raise AssertionError(f"add an EVENT_SAMPLES entry for the new event {name}")
        events.append(attribute(*EVENT_SAMPLES[name]))
    if len(events) != len(EVENT_SAMPLES):
        raise AssertionError("EVENT_SAMPLES has entries that are no longer events")
    return events


def make_session(cwd: Path) -> Session:
    """A persisted session file in cwd, as Session.create would make."""
    session = Session(cwd / "session.jsonl", "session-id", cwd)
    session.append({
        "type": "session",
        "id": "session-id",
        "cwd": str(cwd),
        "created_at": "2026-01-01T00:00:00+00:00",
    })
    return session


def stub_model(tool_name: str | None = None, arguments: str = "{}") -> FunctionModel:
    """Model stub: streams one tool call on the first request (when tool_name
    is given), then a bare text turn."""
    requests = 0

    async def stream(messages, info: AgentInfo):
        nonlocal requests
        requests += 1
        if tool_name is not None and requests == 1:
            yield {0: DeltaToolCall(name=tool_name, json_args=arguments, tool_call_id="call-1")}
        else:
            yield "done"

    return FunctionModel(stream_function=stream)
