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

from .errors import PaimonError
from .llm import user_agent
from .model_windows import CONTEXT_WINDOWS
from .session import is_agents_message, summary_message


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


# Fallback window sizes by model-name fragment, consulted only when the
# generated CONTEXT_WINDOWS table has no entry for the model. First match wins,
# so the more specific fragment goes first. Deliberately coarse and
# conservative: a value that is too small only compacts early, while none at
# all would silently disable the compaction safety net.
_KNOWN_WINDOWS: tuple[tuple[str, int], ...] = (
    ("gpt-4.1", 1_000_000),
    ("gpt-4o", 128_000),
    ("gpt-5", 272_000),
    ("glm-4.6", 200_000),
    ("claude", 200_000),
    ("gemini", 1_000_000),
    ("o1", 200_000),
    ("o3", 200_000),
    ("o4-mini", 200_000),
    ("grok", 256_000),
    ("deepseek", 128_000),
    ("glm", 128_000),
    ("kimi", 128_000),
    ("moonshot", 128_000),
    ("qwen", 128_000),
    ("mistral", 128_000),
    ("llama", 128_000),
)


def context_window(model: Optional[str], override: Optional[int] = None) -> Optional[int]:
    """The window to compact against: the override, else a built-in estimate.

    The estimate first looks the model up in the generated table, which knows
    thousands of models exactly, and falls back to the coarse fragment table
    for anything it has never heard of.  None means the window is unknown,
    which disables auto-compaction; callers surface that state rather than let
    it look like compaction is working.
    """
    if override and override > 0:
        return override
    if not model:
        return None
    name = model.lower()
    # "zai:glm-4.6" is looked up as "glm-4.6", but the qualified name is tried
    # first because a few catalogue entries carry a namespace of their own.
    for candidate in (name, name.partition(":")[2], name.partition("/")[2]):
        window = CONTEXT_WINDOWS.get(candidate)
        if window:
            return window
    for fragment, window in _KNOWN_WINDOWS:
        if fragment in name:
            return window
    return None


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
        # Agent status lines are about a moment, not about the work: the
        # summary prompt asks for current status, and a checkpoint that
        # preserved "a1f2 finished" from three hours ago would keep saying it
        # for the rest of the session.
        if is_agents_message(message):
            continue
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
        response = await model_request(
            model,
            [ModelRequest(parts=[
                SystemPromptPart(content="You create context checkpoint summaries for an AI coding agent."),
                UserPromptPart(content=prompt),
            ])],
            model_settings={"max_tokens": _SUMMARY_MAX_TOKENS,
                            "extra_headers": {"User-Agent": user_agent()}},
            model_request_parameters=ModelRequestParameters(allow_text_output=True),
        )
    summary = "".join(part.content for part in response.parts if isinstance(part, TextPart))
    if not summary.strip():
        raise CompactionError("Context compaction returned an empty summary")
    result = CompactionResult(summary.strip(), kept_messages, tokens_before, tokens_after=0)
    result.tokens_after = count_tokens(result.messages, tool_schemas, system_prompt)
    return result
