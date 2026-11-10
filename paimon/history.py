"""Search and read access to a session's own log, behind the history tools.

The session log is append-only, so everything compaction dropped from the
model's context is still on disk. search_history greps that log and
read_history returns full records by seq, the same 1-based line numbers
``paimon log`` prints.
"""

from __future__ import annotations

import json
import re

from .logtext import render_record, settled_parts, superseded_seqs
from .session import Session, synthetic_kind
from .tools.result import ToolError

_HISTORY_TOOLS = ("search_history", "read_history")
_SEARCH_CONTEXT = 100  # chars of context shown on each side of a match
_DEFAULT_SEARCH_RESULTS = 20
_MAX_SEARCH_RESULTS = 100
_MAX_READ_RECORDS = 20


def _surviving_entries(session: Session) -> list[tuple[int, dict]]:
    """Message, tool result and compaction records with revisions applied,
    seq attached.

    Unlike Session.messages(), a compaction record does not reset the list:
    recovering what compaction dropped is the whole point here. A replaced
    record keeps its original position but carries the replacement's seq, so
    read_history on that seq returns exactly what search matched. A settled
    slot is searched in its own ``tool_result`` record, at that record's seq,
    and blanked out of the placeholder request it settles.
    """
    survivors: list[tuple[int, dict]] = []
    positions: dict[str, int] = {}
    for seq, record in session.entries():
        if record is None:
            continue
        kind = record.get("type")
        if kind == "compaction":
            survivors.append((seq, record))
        elif kind == "tool_result" and isinstance(record.get("part"), dict):
            target, index = record.get("of"), record.get("index")
            if isinstance(target, str) and target in positions and isinstance(index, int):
                held_seq, held = survivors[positions[target]]
                parts = list((held.get("message") or {}).get("parts") or [])
                if 0 <= index < len(parts):
                    parts[index] = None
                    held = {**held, "message": {**held["message"], "parts": parts}}
                    survivors[positions[target]] = (held_seq, held)
            survivors.append((seq, record))
        elif kind == "message" and isinstance(record.get("message"), dict):
            replaced = record.get("replaces")
            if isinstance(replaced, str) and replaced in positions:
                survivors[positions[replaced]] = (seq, record)
            else:
                if isinstance(record.get("id"), str):
                    positions[record["id"]] = len(survivors)
                survivors.append((seq, record))
    return survivors


def _searchable_parts(record: dict):
    """(label, text) for each part of a message record worth searching.

    The history tools' own calls and results are skipped: they quote earlier
    matches, so searching them would return every past search over again.
    """
    if record.get("type") == "tool_result":
        message = {"parts": [record.get("part")]}
    else:
        message = record.get("message") or {}
    # A job notice or a "!" run is labelled as such, never as the user's words.
    author = synthetic_kind(message) or "user"
    for part in message.get("parts") or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("part_kind")
        content = part.get("content")
        if kind == "user-prompt":
            yield author, str(content or "")
        elif kind == "text":
            yield "assistant", str(content or "")
        elif kind == "thinking":
            yield "thinking", str(content or "")
        elif kind in ("tool-call", "tool-return"):
            name = str(part.get("tool_name") or "?")
            if name in _HISTORY_TOOLS:
                continue
            if kind == "tool-call":
                args = part.get("args")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False, default=str)
                yield f"tool_call {name}", f"{name} {args}"
            else:
                text = content if isinstance(content, str) else json.dumps(
                    content, ensure_ascii=False, default=str)
                yield f"tool_result {name}", text


def _snippet(text: str, match: re.Match) -> str:
    start = max(0, match.start() - _SEARCH_CONTEXT)
    end = min(len(text), match.end() + _SEARCH_CONTEXT)
    piece = " ".join(text[start:end].split())
    return f"{'…' if start > 0 else ''}{piece}{'…' if end < len(text) else ''}"


def search(session: Session, args: dict) -> str:
    query = str(args.get("query") or "")
    if not query:
        raise ToolError("Error: query must not be empty")
    try:
        limit = max(1, min(int(args.get("max_results") or _DEFAULT_SEARCH_RESULTS), _MAX_SEARCH_RESULTS))
    except (TypeError, ValueError):
        raise ToolError("Error: max_results must be an integer") from None
    note = ""
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        note = " (query is not a valid regex; searched literally)"
    hits = []
    for seq, record in _surviving_entries(session):
        if record.get("type") == "compaction":
            parts = [("compaction summary", str(record.get("summary") or ""))]
        else:
            parts = _searchable_parts(record)
        for label, text in parts:
            match = pattern.search(text)
            if match:
                hits.append(f"[{seq}] {label}: {_snippet(text, match)}")
    if not hits:
        return f"No matches in the session log{note}."
    shown = hits[:limit]
    header = (f"{len(hits)} matching parts{note}"
              + (f", showing the first {len(shown)}" if len(hits) > len(shown) else "")
              + ". Use read_history with a seq to see a full record.")
    return "\n".join([header, *shown])


def read(session: Session, args: dict, budget: int) -> str:
    """Full records from seq on, cut at a record boundary once past budget chars."""
    try:
        seq = int(args["seq"])
        count = max(1, min(int(args.get("count") or 1), _MAX_READ_RECORDS))
    except (KeyError, TypeError, ValueError):
        raise ToolError("Error: seq (and optional count) must be integers") from None
    entries = session.entries()
    if seq < 1 or seq > len(entries):
        raise ToolError(f"Error: seq out of range (the log has {len(entries)} records)")
    superseded = superseded_seqs(entries)
    settled = settled_parts(entries)
    lines: list[str] = []
    used = 0
    for number, record in entries[seq - 1 : seq - 1 + count]:
        if number in superseded:
            # The content at this seq was replaced later (a placeholder or an
            # intermediate snapshot); returning it would present a result that
            # never happened as real.
            lines.append(f"[{number}] (superseded revision; the final version "
                         f"is at seq {superseded[number]})")
            continue
        skip = settled.get(number, {})
        rendered = "\n".join(render_record(number, record, full=True, skip=skip))
        if skip and not rendered:
            # Every placeholder of this tool request was settled later; the
            # same reasoning as for a superseded revision applies.
            seqs = ", ".join(str(at) for at in sorted(skip.values()))
            rendered = (f"[{number}] (placeholder results, all settled later; "
                        f"the final ones are at seq {seqs})")
        if used + len(rendered) > budget:
            if not lines:
                rendered = rendered[:budget] + "\n... (record truncated)"
                lines.append(rendered)
            else:
                lines.append(f"... (stopped before seq {number}: output budget reached, "
                             "read fewer records at once)")
            break
        lines.append(rendered)
        used += len(rendered)
    return "\n".join(lines)
