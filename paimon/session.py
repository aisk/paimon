"""Append-only JSONL session persistence."""

# Deferred annotations: the ``list`` classmethod shadows the builtin in the
# class body, which would otherwise break ``list[dict]`` annotations below it.
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, UserPromptPart

from . import lockfile

# A compaction checkpoint is a synthetic user message. Its shape belongs here
# rather than to whatever decides *when* to compact: replaying a log has to
# rebuild it, and previews have to recognize it as not-a-user-message.
SUMMARY_PREFIX = "The conversation before this point was compacted into this checkpoint:\n\n"

# The other synthetic user message: a one-line report on the agents this
# session started, prepended to a turn so the model learns that one of them
# finished without anything having to interrupt the user.
AGENTS_PREFIX = "[agents] "


class SessionError(RuntimeError):
    """A session cannot be opened. Callers catch this to report and move on."""


class SessionBusyError(SessionError):
    """The session is already active in another process."""


class SessionIncompleteError(SessionError):
    """The session log is missing the system prompt snapshot it needs to resume."""


def summary_message(summary: str) -> ModelRequest:
    """The synthetic user message placed at the start of compacted context."""
    return ModelRequest(parts=[UserPromptPart(content=SUMMARY_PREFIX + summary)])


def _has_prefixed_user_text(message: ModelMessage, prefix: str) -> bool:
    return (
        isinstance(message, ModelRequest)
        and any(
            isinstance(part, UserPromptPart)
            and isinstance(part.content, str)
            and part.content.startswith(prefix)
            for part in message.parts
        )
    )


def is_summary_message(message: ModelMessage) -> bool:
    return _has_prefixed_user_text(message, SUMMARY_PREFIX)


def agents_message(summary: str) -> ModelRequest:
    """The synthetic user message carrying an agent status line.

    A message of its own rather than something glued onto the user's prompt:
    the first user message is the session's title everywhere it is listed, and
    a turn assembled from several queued prompts has no single place to glue it
    onto anyway.
    """
    return ModelRequest(parts=[UserPromptPart(content=AGENTS_PREFIX + summary)])


def is_agents_message(message: ModelMessage) -> bool:
    return _has_prefixed_user_text(message, AGENTS_PREFIX)


def agents_text(message: ModelMessage) -> str:
    """The status line of an agents message, without its marker prefix."""
    for part in getattr(message, "parts", []):
        if (isinstance(part, UserPromptPart) and isinstance(part.content, str)
                and part.content.startswith(AGENTS_PREFIX)):
            return part.content[len(AGENTS_PREFIX):]
    return ""


def is_synthetic_user_text(content: str) -> bool:
    """Whether a user-prompt string is one paimon wrote rather than the user.

    Previews, titles and "where does a turn start" all have to skip these.
    """
    return content.startswith(SUMMARY_PREFIX) or content.startswith(AGENTS_PREFIX)


def dump_message(message: ModelMessage) -> dict:
    """One ModelMessage as plain JSON-compatible data."""
    return ModelMessagesTypeAdapter.dump_python([message], mode="json")[0]


def load_messages(raw: list) -> list[ModelMessage]:
    return list(ModelMessagesTypeAdapter.validate_python(raw))


def data_dir() -> Path:
    override = os.environ.get("PAIMON_DATA_HOME")
    if override:
        return Path(override)
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return Path(xdg_data) / "paimon"
    return Path.home() / ".local" / "share" / "paimon"


def sessions_dir() -> Path:
    return data_dir() / "sessions"


def _project_dir(cwd: Path) -> Path:
    resolved = cwd.resolve()
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:16]
    name = resolved.name or "root"
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return sessions_dir() / f"{safe_name}-{digest}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resume_hint(session_id: str) -> str:
    """The command that resumes a session, to print when one ends.

    An id prefix is enough for ``--resume`` to find it, and is what the
    session file itself is named after.
    """
    return f"paimon -r {session_id[:8]}"


class Session:
    """A session backed by an append-only JSONL event log."""

    def __init__(self, path: Path, session_id: str, cwd: Path, parent: Optional[str] = None):
        self.path = path
        self.id = session_id
        self.cwd = cwd.resolve()
        # The session that spawned this one, when it belongs to a subagent.
        # Children share the project directory with the session that started
        # them, so they are hidden from listings unless asked for.
        self.parent = parent
        # Whether this session holds the process claim on its file. Claims are
        # refcounted per process, so unlocking twice would drop one somebody
        # else has since taken; this is what makes unlock() idempotent.
        self.locked = False

    def lock(self) -> None:
        """Mark this session active: only one agent may run a session.

        Listing and previews never lock; the Agent locks on construction.
        Raises SessionBusyError if the session is already open, here or in
        another process.
        """
        # The process-wide lock refcounts, so it says yes to a second Agent in
        # this same process — which is exactly the case that corrupts a log:
        # two histories evolving apart while both append to one append-only
        # file replay as a single interleaved conversation.
        if lockfile.held(self.path):
            raise SessionBusyError(f"session {self.id[:8]} is already open in this window")
        if not lockfile.acquire(self.path):
            raise SessionBusyError(f"session {self.id[:8]} is already active in another process")
        self.locked = True

    def unlock(self) -> None:
        """Release this session's claim, once; the lock drops with the last holder."""
        if not self.locked:
            return
        self.locked = False
        lockfile.release(self.path)

    @classmethod
    def create(cls, cwd: Path, parent: Optional[str] = None) -> "Session":
        session_id = str(uuid4())
        directory = _project_dir(cwd)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session = cls(directory / f"{timestamp}-{session_id[:8]}.jsonl", session_id, cwd, parent)
        header = {"type": "session", "id": session_id,
                  "cwd": str(session.cwd), "created_at": _now()}
        if parent:
            header["parent"] = parent
        session.append(header)
        return session

    def fork(self) -> "Session":
        """Copy this session's log into a new session with a fresh id.

        The copy keeps every line except the old header verbatim, so ``log``
        line numbers stay aligned between the two sessions.
        """
        session_id = str(uuid4())
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.path.parent / f"{timestamp}-{session_id[:8]}.jsonl"
        header = {"type": "session", "id": session_id,
                  "cwd": str(self.cwd), "created_at": _now()}
        # A fork of a subagent's transcript is still subagent material, so it
        # inherits the parent and stays out of the listings for the same reason.
        if self.parent:
            header["parent"] = self.parent
        lines = [json.dumps(header, ensure_ascii=False, separators=(",", ":"))]
        with self.path.open(encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = None
                if isinstance(record, dict) and record.get("type") == "session":
                    continue
                lines.append(line.rstrip("\n"))
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, ("\n".join(lines) + "\n").encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        return Session(path, session_id, self.cwd, self.parent)

    @classmethod
    def list(cls, cwd: Path, include_children: bool = False) -> list["Session"]:
        """Sessions that have at least one message, newest first by mtime.

        Reads each log only far enough to see the header and a first message,
        so listing stays cheap with many long sessions.

        Sessions started by a subagent are left out by default. They share this
        project directory with the session that spawned them — the directory is
        keyed by cwd, and a subagent inherits its parent's — so an afternoon of
        parallel work would otherwise bury the user's own sessions, and
        ``paimon -c`` would resume one of them.
        """
        directory = _project_dir(cwd)
        if not directory.is_dir():
            return []
        paths = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        sessions = []
        for path in paths:
            records = cls._iter_records(path)
            header = next(records, None)
            if (header is None or header.get("type") != "session"
                    or not isinstance(header.get("id"), str)):
                continue
            parent = header.get("parent")
            parent = parent if isinstance(parent, str) else None
            if parent and not include_children:
                continue
            if any(record.get("type") == "message" for record in records):
                sessions.append(cls(path, header["id"], cwd, parent))
        return sessions

    @staticmethod
    def _iter_records(path: Path):
        """The log's records in order, skipping unparseable lines.

        An unreadable file yields nothing; consumers that can stop early
        (previews, listing) should iterate this instead of _read_records.
        """
        try:
            with path.open(encoding="utf-8") as file:
                for line in file:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        yield record
        except OSError:
            return

    @staticmethod
    def _read_records(path: Path) -> list[dict]:
        return list(Session._iter_records(path))

    def messages(self) -> list[ModelMessage]:
        raw_messages: list[dict] = []
        positions: dict[str, int] = {}
        for record in self._iter_records(self.path):
            if record.get("type") == "compaction":
                summary = record.get("summary")
                kept_messages = record.get("kept_messages")
                if isinstance(summary, str) and isinstance(kept_messages, list):
                    kept = [message for message in kept_messages if isinstance(message, dict)]
                    raw_messages = [dump_message(summary_message(summary)), *kept]
                    # Compaction snapshots are final: later replacement records
                    # only refer to messages appended after this checkpoint.
                    positions = {}
                continue

            message = record.get("message")
            if record.get("type") != "message" or not isinstance(message, dict):
                continue
            replaced = record.get("replaces")
            if replaced in positions:
                raw_messages[positions[replaced]] = message
            else:
                if isinstance(record.get("id"), str):
                    positions[record["id"]] = len(raw_messages)
                raw_messages.append(message)
        return load_messages(raw_messages)

    def system_prompt(self) -> Optional[str]:
        """Return the system prompt snapshot stored for this session."""
        for record in self._iter_records(self.path):
            if record.get("type") == "system_prompt" and isinstance(record.get("content"), str):
                return record["content"]
        return None

    def created_at(self) -> Optional[str]:
        """ISO timestamp from the session header record, if present."""
        header = next(self._iter_records(self.path), None)
        if (header is not None and header.get("type") == "session"
                and isinstance(header.get("created_at"), str)):
            return header["created_at"]
        return None

    def first_user_text(self) -> Optional[str]:
        """The first user message, for picker previews."""
        for record in self._iter_records(self.path):
            message = record.get("message")
            if record.get("type") != "message" or not isinstance(message, dict):
                continue
            if message.get("kind") != "request":
                continue
            for part in message.get("parts") or []:
                content = part.get("content") if isinstance(part, dict) else None
                if (isinstance(part, dict) and part.get("part_kind") == "user-prompt"
                        and isinstance(content, str) and not is_synthetic_user_text(content)):
                    return content
        return None

    def append_system_prompt(self, content: str) -> None:
        """Persist the system prompt generated when the session is first loaded."""
        self.append({
            "type": "system_prompt",
            "timestamp": _now(),
            "content": content,
        })

    def append_message(self, message: ModelMessage, replaces: Optional[str] = None) -> str:
        record_id = str(uuid4())
        record = {"type": "message", "id": record_id, "timestamp": _now(), "message": dump_message(message)}
        if replaces:
            record["replaces"] = replaces
        self.append(record)
        return record_id

    def append_compaction(self, summary: str, kept_messages: list[ModelMessage], tokens_before: int) -> None:
        """Persist a checkpoint without deleting any earlier JSONL records."""
        self.append({
            "type": "compaction",
            "id": str(uuid4()),
            "timestamp": _now(),
            "summary": summary,
            "kept_messages": [dump_message(message) for message in kept_messages],
            "tokens_before": tokens_before,
        })

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
