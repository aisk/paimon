"""Expansion of ``@path`` mentions into versioned file context."""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass
from pathlib import Path


# Keep automatic attachments small enough that one accidental @ on a generated
# file cannot consume the whole request. The model can use read_file for more.
MAX_MENTION_BYTES = 24_000
MAX_MENTION_LINES = 200

_OPENING_TAG = re.compile(r"<mentioned_file\b(?P<attributes>[^>]*)>")
_ATTRIBUTE = re.compile(r'([\w-]+)="([^"]*)"')


@dataclass(frozen=True)
class MentionedVersion:
    path: str
    sha256: str
    exposure: str


class MentionExpander:
    """Expand file mentions and remember versions already present in context."""

    def __init__(self, cwd: Path, messages: list[dict] | None = None):
        self.cwd = cwd.resolve()
        self._versions: dict[tuple[str, str], MentionedVersion] = {}
        if messages:
            self.restore(messages)

    def restore(self, messages: list[dict]) -> None:
        """Restore prior file inclusions from persisted, expanded user messages."""
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str):
                continue
            for tag in _OPENING_TAG.finditer(content):
                attributes = {name: html.unescape(value) for name, value in _ATTRIBUTE.findall(tag.group("attributes"))}
                if attributes.get("data-paimon-mention") != "1":
                    continue
                version = attributes.get("version", "")
                exposure = attributes.get("exposure")
                path = attributes.get("path")
                if not path or not exposure or not version.startswith("sha256:"):
                    continue
                sha256 = version.removeprefix("sha256:")
                if len(sha256) == 64 and all(char in "0123456789abcdef" for char in sha256):
                    self._versions[(path, sha256)] = MentionedVersion(path, sha256, exposure)

    def expand(self, text: str) -> str:
        """Replace token-start ``@path`` mentions. Spaces are written as ``\\ ``."""
        result: list[str] = []
        cursor = 0
        index = 0
        while index < len(text):
            if text[index] != "@" or (index > 0 and not text[index - 1].isspace()):
                index += 1
                continue
            end = index + 1
            path_chars: list[str] = []
            while end < len(text):
                char = text[end]
                if char == "\\" and end + 1 < len(text) and text[end + 1] == " ":
                    path_chars.append(" ")
                    end += 2
                    continue
                if char.isspace():
                    break
                path_chars.append(char)
                end += 1
            if not path_chars:
                index += 1
                continue
            result.append(text[cursor:index])
            requested = "".join(path_chars)
            result.append(self._expand_path(requested))
            cursor = end
            index = end
        result.append(text[cursor:])
        return "".join(result)

    def _expand_path(self, requested: str) -> str:
        path = Path(requested)
        if not path.is_absolute():
            path = self.cwd / path
        try:
            path = path.resolve(strict=True)
        except (OSError, RuntimeError):
            return self._error(requested, "not_found")
        if not path.is_file():
            return self._error(requested, "not_a_file")
        try:
            data = path.read_bytes()
        except OSError:
            return self._error(requested, "unreadable")

        canonical_path = str(path)
        sha256 = hashlib.sha256(data).hexdigest()
        previous = self._versions.get((canonical_path, sha256))
        if previous:
            return self._reference(previous)

        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        if len(data) <= MAX_MENTION_BYTES and len(lines) <= MAX_MENTION_LINES:
            exposure = "full"
            body = text
            extra = ""
        else:
            exposure = "preview"
            body, included_lines = self._preview(lines)
            extra = f' included_lines="1-{included_lines}" total_lines="{len(lines)}"'

        mentioned = MentionedVersion(canonical_path, sha256, exposure)
        self._versions[(canonical_path, sha256)] = mentioned
        attributes = self._attributes(canonical_path, sha256, exposure) + extra
        return f"<mentioned_file {attributes}>\n{body}\n</mentioned_file>"

    @staticmethod
    def _preview(lines: list[str]) -> tuple[str, int]:
        selected: list[str] = []
        used = 0
        for line in lines[:MAX_MENTION_LINES]:
            encoded = line.encode("utf-8")
            remaining = MAX_MENTION_BYTES - used
            if remaining <= 0:
                break
            if len(encoded) > remaining:
                selected.append(encoded[:remaining].decode("utf-8", errors="ignore"))
                used = MAX_MENTION_BYTES
                break
            selected.append(line)
            used += len(encoded)
        return "".join(selected), len(selected)

    @staticmethod
    def _attributes(path: str, sha256: str, exposure: str) -> str:
        return (
            'data-paimon-mention="1" '
            f'path="{html.escape(path, quote=True)}" '
            f'version="sha256:{sha256}" exposure="{exposure}"'
        )

    @staticmethod
    def _error(requested: str, reason: str) -> str:
        return (
            '<mentioned_file data-paimon-mention="1" '
            f'requested="{html.escape(requested, quote=True)}" status="{reason}" />'
        )

    @staticmethod
    def _reference(mentioned: MentionedVersion) -> str:
        attributes = MentionExpander._attributes(mentioned.path, mentioned.sha256, mentioned.exposure)
        return f'<mentioned_file {attributes} status="previously_mentioned" />'
