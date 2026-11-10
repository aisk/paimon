"""Session-log records and tool calls rendered as plain text.

Shared by ``paimon log``, the history tools and every surface that shows a
tool call on one line. Seq is a record's 1-based physical line number.
"""

from __future__ import annotations

import json
from collections.abc import Collection

from .session import synthetic_kind


def summarize_call(name: str, args: dict, limit: int | None = None) -> str:
    """One-line detail for a tool call, shared by the TUI and headless output.

    With a limit the detail is collapsed onto a single line and truncated,
    for outputs that cannot reflow (a terminal stream) unlike a TUI widget.
    """
    detail = str(args.get("command") or args.get("path") or args.get("question") or args.get("query")
                 or args.get("code") or json.dumps(args, ensure_ascii=False))
    if limit is None:
        return detail
    detail = " ".join(detail.split())
    return detail if len(detail) <= limit else detail[: limit - 1] + "…"


LOG_DETAIL_WIDTH = 120


def _chars(text: str) -> str:
    n = len(text)
    return f"{n / 1000:.1f}k chars" if n >= 1000 else f"{n} chars"


def _clip(text: object, full: bool) -> str:
    if full:
        return str(text)
    line = " ".join(str(text).split())
    return line if len(line) <= LOG_DETAIL_WIDTH else line[: LOG_DETAIL_WIDTH - 1] + "…"


def _result_status(part: dict) -> str | None:
    """The short word for a persisted tool return that did not succeed
    ("failed", "exit 2", "denied"), None for one that did.

    Read from the outcome and detail the agent recorded with the part, never
    from its text.
    """
    outcome = part.get("outcome")
    if not isinstance(outcome, str) or outcome == "success":
        return None
    metadata = part.get("metadata")
    detail = metadata.get("detail") if isinstance(metadata, dict) else None
    return detail if isinstance(detail, str) and detail else outcome


def _render_part(seq: int, part: dict, full: bool, author: str = "user") -> str:
    kind = part.get("part_kind")
    content = part.get("content")
    if kind == "user-prompt":
        label, detail = author, _clip(content, full)
    elif kind == "text":
        label, detail = "assistant", _clip(content, full)
    elif kind == "thinking":
        text = str(content or "")
        label, detail = "thinking", text if full else f"({_chars(text)})"
    elif kind == "tool-call":
        args = part.get("args")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"args": args}
        name = str(part.get("tool_name") or "?")
        detail = summarize_call(name, args if isinstance(args, dict) else {},
                                limit=None if full else LOG_DETAIL_WIDTH)
        label = f"tool_call {name}"
    elif kind == "tool-return":
        text = content if isinstance(content, str) else json.dumps(
            content, ensure_ascii=False, default=str)
        label = f"tool_result {part.get('tool_name') or '?'}"
        status = _result_status(part)
        if status is not None:
            label += f" [{status}]"
        detail = text if full else f"({_chars(text)}) {_clip(text, False)}"
    elif kind == "retry-prompt":
        label, detail = "retry_prompt", _clip(content, full)
    else:
        label, detail = str(kind or "?"), ""
    return f"[{seq}] {label}  {detail}".rstrip()


def _render_message(seq: int, record: dict, full: bool, skip: Collection[int]) -> list[str]:
    lines = []
    message = record.get("message") or {}
    # A message paimon wrote in the user's place is labelled as what it is.
    author = synthetic_kind(message) or "user"
    for index, part in enumerate(message.get("parts") or []):
        if isinstance(part, dict) and index not in skip:
            lines.append(_render_part(seq, part, full, author))
    replaces = record.get("replaces")
    if lines and isinstance(replaces, str):
        lines[0] += f" (replaces {replaces[:8]})"
    return lines


def superseded_seqs(entries: list) -> dict[int, int]:
    """Seq of every message revision a later record replaced → the final seq.

    The log is append-only, so in a format 1 log a tool result exists first as
    the pre-seeded "Interrupted by user." placeholder and then as one
    replacement per completed tool. Only the last revision reflects what
    actually happened; anything rendering the log as events must skip the
    seqs mapped here or it shows interruptions and intermediate snapshots that
    never occurred. Newer logs settle one slot at a time instead; see
    settled_parts.
    """
    id_seq: dict[str, int] = {}
    replacements: dict[str, list[int]] = {}
    for seq, record in entries:
        if not isinstance(record, dict) or record.get("type") != "message":
            continue
        if isinstance(record.get("id"), str):
            id_seq[record["id"]] = seq
        replaced = record.get("replaces")
        if isinstance(replaced, str):
            replacements.setdefault(replaced, []).append(seq)
    superseded: dict[int, int] = {}
    for replaced_id, seqs in replacements.items():
        final = seqs[-1]
        for seq in (id_seq.get(replaced_id), *seqs[:-1]):
            if seq is not None:
                superseded[seq] = final
    return superseded


def settled_parts(entries: list) -> dict[int, dict[int, int]]:
    """Seq of every tool request whose slots later records settled →
    {slot index: seq of the ``tool_result`` record that settled it}.

    The pre-seeded placeholder of a settled slot is not what happened, so a
    renderer skips those parts and shows each result at its own seq instead;
    a slot never settled keeps its placeholder, which then is the truth.
    """
    id_seq: dict[str, int] = {}
    settled: dict[int, dict[int, int]] = {}
    for seq, record in entries:
        if not isinstance(record, dict):
            continue
        kind = record.get("type")
        if kind == "message" and isinstance(record.get("id"), str):
            id_seq[record["id"]] = seq
        elif kind == "tool_result":
            target, index = id_seq.get(record.get("of")), record.get("index")
            if target is not None and isinstance(index, int):
                settled.setdefault(target, {})[index] = seq
    return settled


def render_record(seq: int, record: dict | None, full: bool,
                  skip: Collection[int] = ()) -> list[str]:
    """``record`` as text lines; ``skip`` names message parts left out (the
    slots settled_parts reports as answered elsewhere)."""
    if record is None:
        return [f"[{seq}] <corrupt>"]
    kind = record.get("type")
    if kind == "message":
        return _render_message(seq, record, full, skip)
    if kind == "tool_result":
        part = record.get("part")
        return [_render_part(seq, part, full)] if isinstance(part, dict) else [f"[{seq}] tool_result"]
    if kind == "session":
        return [f"[{seq}] session {str(record.get('id') or '')[:8]} "
                f"created {record.get('created_at') or '?'}"]
    if kind == "system_prompt":
        content = str(record.get("content") or "")
        return [f"[{seq}] system_prompt {content if full else f'({_chars(content)})'}"]
    if kind == "compaction":
        kept = record.get("kept_messages")
        return [f"[{seq}] compacted: {record.get('tokens_before') or 0:,} tokens "
                f"→ summary + {len(kept) if isinstance(kept, list) else 0} kept"]
    if kind == "turn_end":
        line = f"[{seq}] turn_end {record.get('outcome') or '?'}"
        error = record.get("error")
        if isinstance(error, str) and error:
            line += f"  {_clip(error, full)}"
        lines = [line]
        partial = record.get("partial_text")
        if isinstance(partial, str) and partial:
            lines.append(f"[{seq}] partial  "
                         + (partial if full else f"({_chars(partial)}) {_clip(partial, False)}"))
        return lines
    if kind == "model_retry":
        return [f"[{seq}] model_retry attempt {record.get('attempt') or '?'}  "
                f"{_clip(record.get('error') or '', full)}".rstrip()]
    if kind == "compaction_failed":
        return [f"[{seq}] compaction_failed  {_clip(record.get('error') or '', full)}".rstrip()]
    return [f"[{seq}] {kind or '?'}"]
