"""Model stubs, event samples and persisted sessions for agent tests."""

import asyncio
import json
import typing
from pathlib import Path

from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from paimon import agent as agent_module
from paimon.config import Config
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
    "RequestStats": (120, 2.5, 2000, 1600, 300),
    "ToolBudgetExhausted": (3,),
    "TurnEnd": (),
    "ContextCompacted": (10, 5),
    "ContextCompactionFailed": ("boom",),
    "ModelRetry": (1, 4, 2.0, "HTTP 429"),
    "UserInput": ("hi",),
    "CompactionNotice": (),
    "JobNotice": ("agent a1f2 finished:\nthe parser is fine",),
    "ShellRun": ("ls", "a.txt"),
}

# Events a renderer may legitimately draw nothing for.
SILENT_EVENTS = {"TurnEnd", "SessionHandoff", "RequestStats", "ToolBudgetExhausted"}


def agent_events() -> list[object]:
    """One instance of every event in paimon.agent's AgentEvent union."""
    events = []
    for attribute in sorted(typing.get_args(agent_module.AgentEvent), key=lambda e: e.__name__):
        name = attribute.__name__
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


def open_agent(cwd: Path, **kwargs) -> agent_module.Agent:
    """An Agent on a fresh persisted session in cwd, with a stub model configured."""
    session = make_session(cwd)
    session.append_system_prompt("snapshot")
    return agent_module.Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"), **kwargs)


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


def session_records(session: Session) -> list[dict]:
    """Every raw record in the session log, in order."""
    return [json.loads(line) for line in
            session.path.read_text(encoding="utf-8").splitlines()]


def spawning_model(prompts: list[str], *, gate: asyncio.Event | None = None,
                   answer: str = "done", fail: str | None = None) -> FunctionModel:
    """One stub for a parent and the agents it starts.

    The parent's first request spawns one agent per prompt and every later one
    answers "ok". A request whose conversation opens with one of the prompts
    is a child's: it waits for ``gate`` when there is one, then answers
    ``answer`` or raises ``fail``.
    """
    requests = 0

    async def stream(messages, info: AgentInfo):
        nonlocal requests
        asked = [part.content for message in messages for part in message.parts
                 if isinstance(part, UserPromptPart)]
        if any(prompt in asked for prompt in prompts):
            if gate is not None:
                await gate.wait()
            if fail:
                raise RuntimeError(fail)
            yield answer
            return
        requests += 1
        if requests == 1:
            yield {index: DeltaToolCall(name="spawn_agent", tool_call_id=f"call-{index}",
                                        json_args=json.dumps({"prompt": prompt}))
                   for index, prompt in enumerate(prompts)}
        else:
            yield "ok"

    return FunctionModel(stream_function=stream)
