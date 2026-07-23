"""Shared fixtures for the test suite."""

from pathlib import Path
from types import SimpleNamespace

from paimon.session import Session


def make_session(cwd: Path) -> Session:
    """A persisted session file in cwd, as Session.create would make."""
    session = Session(cwd / "session.jsonl", "session-id", cwd)
    session.append({
        "type": "session",
        "version": 1,
        "id": "session-id",
        "cwd": str(cwd),
        "created_at": "2026-01-01T00:00:00+00:00",
    })
    return session


def stub_completion(tool_name: str | None = None, arguments: str = "{}"):
    """litellm.acompletion replacement: streams one tool call on the first
    request (when tool_name is given), then a bare text turn."""
    requests = 0

    async def completion(**_kwargs):
        nonlocal requests
        requests += 1

        async def stream(with_call: bool):
            if with_call:
                call = SimpleNamespace(
                    index=0, id="call-1",
                    function=SimpleNamespace(name=tool_name, arguments=arguments),
                )
                delta = SimpleNamespace(content=None, tool_calls=[call], reasoning_content=None)
            else:
                delta = SimpleNamespace(content="done", tool_calls=[], reasoning_content=None)
            yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

        return stream(tool_name is not None and requests == 1)

    return completion
