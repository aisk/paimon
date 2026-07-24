"""Small, loss-tolerant context compaction helpers.

The session log remains append-only.  Compaction only changes the list of
messages sent to the model: old messages become a checkpoint summary while a
recent suffix is kept verbatim.
"""

import json
from dataclasses import dataclass
from typing import Optional

from pydantic_ai.direct import model_request
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters


SUMMARY_PREFIX = "The conversation before this point was compacted into this checkpoint:\n\n"
_TOOL_RESULT_LIMIT = 2_000


@dataclass
class CompactionResult:
    summary: str
    kept_messages: list[ModelMessage]
    tokens_before: int
    tokens_after: int


def summary_message(summary: str) -> ModelRequest:
    """Return the synthetic user message placed at the start of compacted context."""
    return ModelRequest(parts=[UserPromptPart(content=SUMMARY_PREFIX + summary)])


def is_summary_message(message: ModelMessage) -> bool:
    return (
        isinstance(message, ModelRequest)
        and any(
            isinstance(part, UserPromptPart)
            and isinstance(part.content, str)
            and part.content.startswith(SUMMARY_PREFIX)
            for part in message.parts
        )
    )


def context_window(override: Optional[int] = None) -> Optional[int]:
    """The configured input window; None disables compaction."""
    return override if override and override > 0 else None


def count_tokens(messages: list[ModelMessage], tool_schemas: Optional[list[dict]] = None) -> int:
    """Approximate context tokens as serialized characters / 4."""
    payload = json.dumps(ModelMessagesTypeAdapter.dump_python(messages, mode="json"), ensure_ascii=False)
    if tool_schemas:
        payload += json.dumps(tool_schemas, ensure_ascii=False, default=str)
    return max(1, (len(payload) + 3) // 4)


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


async def compact(
    messages: list[ModelMessage],
    *,
    model: Model,
    keep_recent_tokens: int,
    tokens_before: int,
    tool_schemas: Optional[list[dict]] = None,
) -> Optional[CompactionResult]:
    """Summarize the old prefix and return a new effective context."""
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
{_serialize_messages(old_messages)}
</conversation>"""

    response = await model_request(
        model,
        [ModelRequest(parts=[
            SystemPromptPart(content="You create context checkpoint summaries for an AI coding agent."),
            UserPromptPart(content=prompt),
        ])],
        model_settings={"max_tokens": 2_048},
        model_request_parameters=ModelRequestParameters(allow_text_output=True),
    )
    summary = "".join(part.content for part in response.parts if isinstance(part, TextPart))
    if not summary.strip():
        raise RuntimeError("Context compaction returned an empty summary")
    summary = summary.strip()
    compacted_messages = [summary_message(summary), *kept_messages]
    tokens_after = count_tokens(compacted_messages, tool_schemas)
    return CompactionResult(summary, kept_messages, tokens_before, tokens_after)
