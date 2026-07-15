"""Small, loss-tolerant context compaction helpers.

The session log remains append-only.  Compaction only changes the list of
messages sent to the model: old messages become a checkpoint summary while a
recent suffix is kept verbatim.
"""

import json
from dataclasses import dataclass
from typing import Optional

import litellm


SUMMARY_NAME = "paimon_context_summary"
SUMMARY_PREFIX = "The conversation before this point was compacted into this checkpoint:\n\n"
_TOOL_RESULT_LIMIT = 2_000


@dataclass
class CompactionResult:
    summary: str
    kept_messages: list[dict]
    tokens_before: int
    tokens_after: int


def summary_message(summary: str) -> dict:
    """Return the synthetic user message placed at the start of compacted context."""
    return {"role": "user", "name": SUMMARY_NAME, "content": SUMMARY_PREFIX + summary}


def context_window(model: Optional[str], override: Optional[int] = None) -> Optional[int]:
    """Return the configured or LiteLLM-known input window for *model*."""
    if override and override > 0:
        return override
    if not model:
        return None
    try:
        value = litellm.get_model_info(model).get("max_input_tokens")
    except Exception:  # noqa: BLE001 - unknown/custom models are expected here
        return None
    return int(value) if isinstance(value, (int, float)) and value > 0 else None


def count_tokens(model: Optional[str], messages: list[dict], tool_schemas: Optional[list[dict]] = None) -> int:
    """Count context tokens, falling back to a dependency-free approximation."""
    try:
        return int(litellm.token_counter(model=model or "", messages=messages, tools=tool_schemas))
    except Exception:  # noqa: BLE001 - custom model names may have no tokenizer
        payload = json.dumps(messages, ensure_ascii=False, default=str)
        if tool_schemas:
            payload += json.dumps(tool_schemas, ensure_ascii=False, default=str)
        return max(1, (len(payload) + 3) // 4)


def should_compact(tokens: int, window: Optional[int], reserve_tokens: int) -> bool:
    return window is not None and tokens > window - reserve_tokens


def find_cut_index(messages: list[dict], keep_recent_tokens: int, model: Optional[str]) -> int:
    """Find the first recent message to retain.

    The walk is intentionally approximate.  A tool result is never used as a
    boundary, so an assistant tool call remains attached to all of its results.
    """
    accumulated = 0
    for index in range(len(messages) - 1, -1, -1):
        accumulated += count_tokens(model, [messages[index]])
        if accumulated < keep_recent_tokens:
            continue

        cut = index
        while cut > 0 and messages[cut].get("role") == "tool":
            cut -= 1
        return cut
    return 0


def _serialize_messages(messages: list[dict]) -> str:
    serialized: list[str] = []
    for message in messages:
        copy = dict(message)
        if copy.get("role") == "tool":
            content = str(copy.get("content") or "")
            if len(content) > _TOOL_RESULT_LIMIT:
                copy["content"] = content[:_TOOL_RESULT_LIMIT] + "\n[tool result truncated for summary]"
        serialized.append(json.dumps(copy, ensure_ascii=False, default=str))
    return "\n".join(serialized)


async def compact(
    messages: list[dict],
    *,
    model: Optional[str],
    api_base: Optional[str],
    api_key: Optional[str],
    keep_recent_tokens: int,
    tokens_before: int,
    tool_schemas: Optional[list[dict]] = None,
) -> Optional[CompactionResult]:
    """Summarize the old prefix and return a new effective context."""
    cut = find_cut_index(messages, keep_recent_tokens, model)
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

    response = await litellm.acompletion(
        model=model,
        api_base=api_base,
        api_key=api_key,
        messages=[
            {"role": "system", "content": "You create context checkpoint summaries for an AI coding agent."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2_048,
    )
    summary = response.choices[0].message.content
    if not isinstance(summary, str) or not summary.strip():
        raise RuntimeError("Context compaction returned an empty summary")
    summary = summary.strip()
    compacted_messages = [summary_message(summary), *kept_messages]
    tokens_after = count_tokens(model, compacted_messages, tool_schemas)
    return CompactionResult(summary, kept_messages, tokens_before, tokens_after)
