"""Small, loss-tolerant context compaction helpers.

The session log remains append-only.  Compaction only changes the list of
messages sent to the model: old messages become a checkpoint summary while a
recent suffix is kept verbatim.
"""

import asyncio
import json
import weakref
from dataclasses import dataclass
from typing import Optional

from genai_prices.data_snapshot import get_snapshot
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ThinkingPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model

from .errors import PaimonError
from .llm import ask_once
from .session import summary_message


class CompactionError(PaimonError):
    """A checkpoint could not be made. The context is left as it was."""


_TOOL_RESULT_LIMIT = 2_000

# Output budget of the summary request itself.
_SUMMARY_MAX_TOKENS = 2_048

# Input bound for the summary request when the model's window is unknown.
_DEFAULT_SUMMARY_INPUT_TOKENS = 60_000

# How many compactions may be in flight at once. Every agent in the process
# shares one event loop, and compaction is the largest request any of them
# sends: without a ceiling, eight panes filling up together fire eight
# whole-history requests at the provider at the same moment, which is exactly
# when a rate limit is least affordable (three failures and compaction is off
# for the rest of that turn).
_MAX_CONCURRENT_COMPACTIONS = 3

# Keyed by event loop: an asyncio primitive binds to the loop that first blocks
# on it, and the test suite runs a fresh loop per test.
_slots: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _slot() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphore = _slots.get(loop)
    if semaphore is None:
        semaphore = _slots[loop] = asyncio.Semaphore(_MAX_CONCURRENT_COMPACTIONS)
    return semaphore


@dataclass
class CompactionResult:
    summary: str
    kept_messages: list[ModelMessage]
    tokens_before: int
    tokens_after: int

    @property
    def messages(self) -> list[ModelMessage]:
        """The context that replaces the old one: checkpoint plus recent tail."""
        return [summary_message(self.summary), *self.kept_messages]


def _window_by_name(model_name: str) -> Optional[int]:
    """The window genai-prices records for this model name under any provider.

    pydantic-ai matches on the provider as well, which misses a known model
    served from an endpoint it does not recognize: a proxy, a local gateway.
    """
    try:
        _, info = get_snapshot().find_provider_model(model_name, None, None, None)
    except LookupError:
        return None
    return info.context_window


def context_window(model: Optional[Model], fallback: Optional[int] = None) -> Optional[int]:
    """The window to compact against: what is on record, else the fallback.

    The record is pydantic-ai's own profile for the model, filled from
    genai-prices.  The fallback is one number for the whole config, so it
    only stands in for a model the record does not know: agents on different
    models share a config, and a known model must keep its own window.  None
    means the window is unknown, which disables auto-compaction; callers
    surface that state rather than let it look like compaction is working.
    """
    known = model is not None and (model.context_window or _window_by_name(model.model_name))
    if known:
        return known
    return fallback if fallback and fallback > 0 else None


def count_tokens(messages: list[ModelMessage], tool_schemas: Optional[list[dict]] = None,
                 system_prompt: Optional[str] = None) -> int:
    """Approximate context tokens as serialized UTF-8 bytes / 4.

    Bytes rather than characters: CJK text runs about one token per
    character, and at three UTF-8 bytes each the byte count keeps the
    estimate close where a character count would be several times too low.
    The estimate must cover everything a request actually sends, so the
    system prompt (absent from the history) is counted here too.
    """
    payload = json.dumps(ModelMessagesTypeAdapter.dump_python(messages, mode="json"), ensure_ascii=False)
    if tool_schemas:
        payload += json.dumps(tool_schemas, ensure_ascii=False, default=str)
    if system_prompt:
        payload += system_prompt
    return max(1, (len(payload.encode("utf-8")) + 3) // 4)


def should_compact(tokens: int, window: Optional[int], reserve_tokens: int) -> bool:
    return window is not None and tokens > window - reserve_tokens


def _is_tool_return(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, ToolReturnPart) for part in message.parts
    )


def find_cut_index(messages: list[ModelMessage], keep_recent_tokens: int) -> int:
    """Find the first recent message to retain.

    The walk is intentionally approximate.  A tool-return request is never used
    as a boundary, so a model response stays attached to all of its results.
    """
    accumulated = 0
    for index in range(len(messages) - 1, -1, -1):
        accumulated += count_tokens([messages[index]])
        if accumulated < keep_recent_tokens:
            continue

        cut = index
        while cut > 0 and _is_tool_return(messages[cut]):
            cut -= 1
        return cut
    return 0


def _serialize_messages(messages: list[ModelMessage]) -> str:
    """Message-per-line JSON for the summary prompt, with reasoning dropped and
    tool results truncated."""
    serialized: list[str] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            message = ModelResponse(
                parts=[part for part in message.parts if not isinstance(part, ThinkingPart)],
                model_name=message.model_name,
                timestamp=message.timestamp,
            )
        raw = ModelMessagesTypeAdapter.dump_python([message], mode="json")[0]
        for part in raw.get("parts") or []:
            content = part.get("content")
            if part.get("part_kind") == "tool-return" and isinstance(content, str) and len(content) > _TOOL_RESULT_LIMIT:
                part["content"] = content[:_TOOL_RESULT_LIMIT] + "\n[tool result truncated for summary]"
        serialized.append(json.dumps(raw, ensure_ascii=False))
    return "\n".join(serialized)


def _bounded(serialized: str, window: Optional[int]) -> str:
    """Fit the serialized conversation into the summary request's own window.

    The compaction request must never itself overflow — it is what runs when
    the context is already too big. Keep the head (the goal, the constraints)
    and the tail (recent work) and elide the middle, cut at message-line
    boundaries.
    """
    budget_tokens = (window - _SUMMARY_MAX_TOKENS - 2_000 if window
                     else _DEFAULT_SUMMARY_INPUT_TOKENS)
    budget_bytes = max(8_000, budget_tokens * 4)
    byte_length = len(serialized.encode("utf-8"))
    if byte_length <= budget_bytes:
        return serialized
    # Slicing below works on characters, so scale the byte budget by the
    # text's average character width. The budget already carries a
    # 2000-token slack, which absorbs head and tail being wider than the
    # average.
    budget_chars = budget_bytes * len(serialized) // byte_length
    head = serialized[: budget_chars // 4]
    if "\n" in head:
        head = head[: head.rfind("\n")]
    tail = serialized[-(budget_chars - len(head)):]
    cut = tail.find("\n")
    if cut != -1:
        tail = tail[cut + 1:]
    omitted = len(serialized) - len(head) - len(tail)
    return (f"{head}\n[... roughly {omitted} characters from the middle of the "
            f"conversation omitted so this summary request fits the model ...]\n{tail}")


async def compact(
    messages: list[ModelMessage],
    *,
    model: Model,
    keep_recent_tokens: int,
    tokens_before: int,
    tool_schemas: Optional[list[dict]] = None,
    system_prompt: Optional[str] = None,
    window: Optional[int] = None,
) -> Optional[CompactionResult]:
    """Summarize the old prefix and return a new effective context.

    ``window`` bounds the summary request's own input (see ``_bounded``); when
    None a conservative default applies.
    """
    cut = find_cut_index(messages, keep_recent_tokens)
    if cut <= 0:
        return None

    old_messages = messages[:cut]
    kept_messages = messages[cut:]
    prompt = f"""Summarize this coding-agent conversation as a checkpoint for another model.
Do not continue the conversation or answer its questions. Be concise, but preserve exact
file paths, commands, errors, user requirements, completed work, and the next steps.

Use these sections:
## Goal
## Constraints
## Progress
## Key Decisions
## Next Steps
## Critical Context

<conversation>
{_bounded(_serialize_messages(old_messages), window)}
</conversation>"""

    async with _slot():
        summary = await ask_once(
            model,
            [ModelRequest(parts=[
                SystemPromptPart(content="You create context checkpoint summaries for an AI coding agent."),
                UserPromptPart(content=prompt),
            ])],
            max_tokens=_SUMMARY_MAX_TOKENS,
        )
    if not summary.strip():
        raise CompactionError("Context compaction returned an empty summary")
    result = CompactionResult(summary.strip(), kept_messages, tokens_before, tokens_after=0)
    result.tokens_after = count_tokens(result.messages, tool_schemas, system_prompt)
    return result
