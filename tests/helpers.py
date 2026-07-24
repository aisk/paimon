"""Shared fixtures for the test suite."""

from pathlib import Path

from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from paimon.session import FORMAT_VERSION, Session


def make_session(cwd: Path) -> Session:
    """A persisted session file in cwd, as Session.create would make."""
    session = Session(cwd / "session.jsonl", "session-id", cwd)
    session.append({
        "type": "session",
        "version": FORMAT_VERSION,
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
