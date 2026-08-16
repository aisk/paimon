"""One question asked over a session's context, off to the side of its turn.

An aside is a read-only model request: it sends the system prompt, the history
as it stands right now, and one extra user message, and hands back the answer
as text. Nothing it does is recorded — not in ``Agent.history``, not in the
session log — so the conversation continues as if it had never been asked.

That is what makes it safe to ask while a turn is already waiting on the model.
The turn owns the history; an aside only reads a snapshot of it and opens a
request of its own, so neither one waits for the other.
"""

import asyncio
from typing import AsyncIterator, Optional

from pydantic_ai.direct import model_request_stream
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters

from . import retry
from .errors import PaimonError
from .llm import user_agent

DEFAULT_MAX_TOKENS = 2_048

# Prepended to the question so the model answers it rather than taking it for
# the next instruction and resuming the work.
DEFAULT_INSTRUCTIONS = (
    "[aside] Answer this out-of-band question about the conversation so far. "
    "Reply in plain text, do not call tools, and do not continue the task. "
    "Nothing you write here is added to the conversation."
)

# The recap: what the UI asks for on its own once the user has gone quiet after
# a turn that did some work. Lives here rather than in the pane that schedules
# it, so the prose stays next to the mechanism that sends it and testable
# without a terminal.
RECAP_INSTRUCTIONS = (
    "[recap] The user stepped away and is coming back to a screen full of tool "
    "output. In two to five short lines, say where things stand: what this "
    "session is working toward overall, what was just done, and the obvious "
    "next step. Answer in the language the user has been writing in. No "
    "headings, no code blocks, no preamble, and do not offer to help."
)

RECAP_QUESTION = "Where does this session stand right now?"


class AsideError(PaimonError):
    """An aside finished without an answer in it."""


def usable_history(history: list[ModelMessage]) -> list[ModelMessage]:
    """A snapshot of ``history`` that is valid to send on its own.

    A turn appends its response before it appends the tool results answering
    it, and yields to the event loop in between, so a snapshot taken at that
    moment ends on tool calls that nothing has answered yet, which every API
    rejects. Those trailing responses are dropped, so the aside simply does not
    see the calls that are still in flight.

    Shallow by design. Later appends cannot reach this list, and a tool result
    filled in afterwards only swaps one string for another in a part this list
    shares, which leaves the message valid either way.
    """
    answered = {
        part.tool_call_id
        for message in history if isinstance(message, ModelRequest)
        for part in message.parts if isinstance(part, ToolReturnPart)
    }
    snapshot = list(history)
    while snapshot and isinstance(snapshot[-1], ModelResponse) and any(
        isinstance(part, ToolCallPart) and part.tool_call_id not in answered
        for part in snapshot[-1].parts
    ):
        snapshot.pop()
    return snapshot


async def stream(
    model: Model,
    system_prompt: str,
    history: list[ModelMessage],
    question: str,
    *,
    instructions: str = DEFAULT_INSTRUCTIONS,
    tool_definitions: Optional[list] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> AsyncIterator[str]:
    """Ask ``question`` over this context, yielding the answer as text deltas.

    ``history`` is sent as given; sanitizing it is the caller's job (see
    ``usable_history``). Thinking is dropped rather than yielded: an aside is
    an answer to read, not a turn to render.

    The tool definitions are passed on so the request has the same shape as a
    normal one, since a history full of tool calls and results is safest
    replayed to a request that still declares the tools. The instructions ask
    for plain text all the same, and a model that calls a tool anyway has
    answered nothing, which raises ``AsideError``.
    """
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart(content=system_prompt)]),
        *history,
        ModelRequest(parts=[UserPromptPart(content=f"{instructions}\n\n{question}")]),
    ]
    parameters = ModelRequestParameters(
        function_tools=list(tool_definitions or []), allow_text_output=True
    )
    settings = {"max_tokens": max_tokens, "extra_headers": {"User-Agent": user_agent()}}

    attempt = 0
    answered = False
    while True:
        # Retried only while nothing has reached the caller, for the reason
        # the turn loop retries that way: a restart would repeat the deltas
        # already yielded.
        started = False
        try:
            async with model_request_stream(
                model, messages, model_settings=settings, model_request_parameters=parameters
            ) as events:
                async for event in events:
                    if isinstance(event, PartStartEvent):
                        part = event.part
                        if isinstance(part, TextPart) and part.content:
                            started = answered = True
                            yield part.content
                    elif isinstance(event, PartDeltaEvent):
                        delta = event.delta
                        if isinstance(delta, TextPartDelta) and delta.content_delta:
                            started = answered = True
                            yield delta.content_delta
            break
        except Exception as exc:  # noqa: BLE001 — classified by retry.is_transient
            attempt += 1
            if started or attempt >= retry.MAX_ATTEMPTS or not retry.is_transient(exc):
                raise
            await asyncio.sleep(retry.backoff(attempt))

    if not answered:
        raise AsideError("The model answered the aside with no text")


async def ask(
    model: Model,
    system_prompt: str,
    history: list[ModelMessage],
    question: str,
    **kwargs,
) -> str:
    """``stream`` collected into one string, for callers with nothing to render."""
    return "".join([delta async for delta in stream(model, system_prompt, history, question, **kwargs)])
