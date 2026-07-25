"""Append-only JSONL session persistence.

Format version 2 stores pydantic-ai ``ModelMessage`` JSON in message records.
Version-1 (litellm dict) sessions are not converted; they are skipped when
listing sessions to resume.
"""

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

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from . import lockfile
from .compaction import SUMMARY_PREFIX, summary_message

FORMAT_VERSION = 2


class SessionBusyError(RuntimeError):
    """The session is already active in another process."""


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

    def __init__(self, path: Path, session_id: str, cwd: Path):
        self.path = path
        self.id = session_id
        self.cwd = cwd.resolve()

    def lock(self) -> None:
        """Mark this session active: only one process may run a session.

        Listing and previews never lock; the Agent locks on construction.
        Raises SessionBusyError if another process holds the session.
        """
        if not lockfile.acquire(self.path):
            raise SessionBusyError(f"session {self.id[:8]} is already active in another process")

    def unlock(self) -> None:
        """Release this process's claim; the lock drops with the last holder."""
        lockfile.release(self.path)

    @classmethod
    def create(cls, cwd: Path) -> "Session":
        session_id = str(uuid4())
        directory = _project_dir(cwd)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session = cls(directory / f"{timestamp}-{session_id[:8]}.jsonl", session_id, cwd)
        session.append({"type": "session", "version": FORMAT_VERSION, "id": session_id,
                        "cwd": str(session.cwd), "created_at": _now()})
        return session

    @classmethod
    def _scan(cls, cwd: Path) -> list[tuple["Session", list[dict]]]:
        """Valid sessions for cwd with their records, newest first by mtime."""
        directory = _project_dir(cwd)
        if not directory.is_dir():
            return []
        paths = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [(cls(path, str(records[0]["id"]), cwd), records)
                for path in paths
                if (records := cls._read_records(path))
                and records[0].get("type") == "session"
                and records[0].get("version") == FORMAT_VERSION]

    @classmethod
    def list(cls, cwd: Path) -> list["Session"]:
        """Sessions that have at least one message, newest first."""
        return [session for session, records in cls._scan(cwd)
                if any(record.get("type") == "message" for record in records)]

    @staticmethod
    def _read_records(path: Path) -> list[dict]:
        records = []
        try:
            with path.open(encoding="utf-8") as file:
                for line in file:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        except OSError:
            return []
        return records

    def messages(self) -> list[ModelMessage]:
        raw_messages: list[dict] = []
        positions: dict[str, int] = {}
        for record in self._read_records(self.path):
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
        for record in self._read_records(self.path):
            if record.get("type") == "system_prompt" and isinstance(record.get("content"), str):
                return record["content"]
        return None

    def created_at(self) -> Optional[str]:
        """ISO timestamp from the session header record, if present."""
        records = self._read_records(self.path)
        if records and records[0].get("type") == "session" and isinstance(records[0].get("created_at"), str):
            return records[0]["created_at"]
        return None

    def first_user_text(self) -> Optional[str]:
        """The first user message, for picker previews."""
        for record in self._read_records(self.path):
            message = record.get("message")
            if record.get("type") != "message" or not isinstance(message, dict):
                continue
            if message.get("kind") != "request":
                continue
            for part in message.get("parts") or []:
                content = part.get("content") if isinstance(part, dict) else None
                if (isinstance(part, dict) and part.get("part_kind") == "user-prompt"
                        and isinstance(content, str) and not content.startswith(SUMMARY_PREFIX)):
                    return content
        return None

    def append_system_prompt(self, content: str) -> None:
        """Persist the system prompt generated when the session is first loaded."""
        self.append({
            "type": "system_prompt",
            "version": 1,
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
