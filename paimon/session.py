"""Append-only JSONL session persistence."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

FORMAT_VERSION = 1


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


class Session:
    """A session backed by an append-only JSONL event log."""

    def __init__(self, path: Path, session_id: str, cwd: Path):
        self.path = path
        self.id = session_id
        self.cwd = cwd.resolve()

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
    def latest(cls, cwd: Path) -> Optional["Session"]:
        directory = _project_dir(cwd)
        if not directory.is_dir():
            return None
        paths = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths:
            records = cls._read_records(path)
            if records and records[0].get("type") == "session" and records[0].get("version") == FORMAT_VERSION:
                return cls(path, str(records[0]["id"]), cwd)
        return None

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

    def messages(self) -> list[dict]:
        messages: list[dict] = []
        positions: dict[str, int] = {}
        for record in self._read_records(self.path):
            message = record.get("message")
            if record.get("type") != "message" or not isinstance(message, dict):
                continue
            replaced = record.get("replaces")
            if replaced in positions:
                messages[positions[replaced]] = message
            else:
                if isinstance(record.get("id"), str):
                    positions[record["id"]] = len(messages)
                messages.append(message)
        return messages

    def append_message(self, message: dict, replaces: Optional[str] = None) -> str:
        record_id = str(uuid4())
        record = {"type": "message", "id": record_id, "timestamp": _now(), "message": message}
        if replaces:
            record["replaces"] = replaces
        self.append(record)
        return record_id

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
