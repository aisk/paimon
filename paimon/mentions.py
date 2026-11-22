"""Expansion of ``@path`` mentions into file context."""

from __future__ import annotations

import html
import re
from pathlib import Path

from .tools.files import read_image, resolve_path
from .tools.result import Image, ToolError


# Inline only files small enough that one accidental @ cannot dominate the
# request. Anything larger is referenced by path; the model uses read_file.
MAX_INLINE_BYTES = 4_000

# Punctuation that may separate a mention from the rest of a sentence. CJK text
# puts no space after it, so a mention runs on until the next whitespace.
_PUNCTUATION = ".,;:!?'\"()[]{}<>，。、；：！？…（）【】《》“”‘’"


def expand_mentions(text: str, cwd: Path, images: list[Image] | None = None) -> str:
    """Replace token-start ``@path`` mentions. Spaces are written as ``\\ ``.

    A mentioned image is added to ``images``, for the caller to attach to the
    message; without a list to add to it is referenced by path like any binary.
    """
    cwd = cwd.resolve()
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
        requested = "".join(path_chars)
        resolved = _resolve(requested, cwd, images)
        # A mention that names no readable file is left verbatim: it is more
        # likely ordinary text (a decorator, an email address) than a mistake.
        if resolved is None:
            index = end
            continue
        tag, kept = resolved
        # Any dropped tail is punctuation, never an escaped space, so one
        # decoded character always gives back one source character.
        end -= len(requested) - kept
        result.append(text[cursor:index])
        result.append(tag)
        cursor = end
        index = end
    result.append(text[cursor:])
    return "".join(result)


def mentioned_images(text: str, cwd: Path) -> tuple[Image, ...]:
    """The images ``text`` mentions, in the order it does."""
    images: list[Image] = []
    expand_mentions(text, cwd, images)
    return tuple(images)


# A markdown image, its target bare or in angle brackets, with or without a title.
_LINKED = re.compile(r"""!\[[^\]]*\]\(\s*(?:<([^>]+)>|([^)\s]+))(?:\s+["'][^)]*)?\s*\)""")
_FENCED = re.compile(r"^(```|~~~).*?^\1", re.S | re.M)


def linked_images(text: str, cwd: Path) -> tuple[Image, ...]:
    """The local images an answer shows with ``![alt](path)``, in order.

    One that is not there, is not an image or is too large is passed over:
    the answer still names it, which is all a missing picture can do.
    """
    images: list[Image] = []
    for match in _LINKED.finditer(_FENCED.sub("", text)):
        target = match[1] or match[2]
        if re.match(r"[a-z][a-z0-9+.-]*://", target, re.I):
            continue
        try:
            path = resolve_path(target, cwd)
            image = read_image(path) if path.is_file() else None
        except (OSError, RuntimeError, ValueError, ToolError):
            continue
        if image is not None:
            images.append(image)
    return tuple(images)


def _resolve(requested: str, cwd: Path, images: list[Image] | None) -> tuple[str, int] | None:
    """A mention's tag and the length of the path it consumed, or None."""
    tag = _expand_path(requested, cwd, images)
    if tag is not None:
        return tag, len(requested)
    # The whole token is not a file, so punctuation inside it probably ends the
    # path ("看看 @a.py，还有"). Retry at each punctuation boundary, longest
    # prefix first; a file genuinely named "foo." already matched above.
    for cut in range(len(requested) - 1, 0, -1):
        if requested[cut] not in _PUNCTUATION:
            continue
        tag = _expand_path(requested[:cut], cwd, images)
        if tag is not None:
            return tag, cut
    return None


def _expand_path(requested: str, cwd: Path, images: list[Image] | None) -> str | None:
    """The tag for a mention, or None when it does not name a readable file."""
    path = Path(requested)
    if not path.is_absolute():
        path = cwd / path
    try:
        path = path.resolve(strict=True)
        if not path.is_file():
            return None
        if images is not None:
            escaped = html.escape(str(path), quote=True)
            try:
                image = read_image(path)
            except ToolError:
                return f'<mentioned_image path="{escaped}" error="too large to attach" />'
            if image is not None:
                images.append(image)
                return f'<mentioned_image path="{escaped}" />'
        data = path.read_bytes()
    except (OSError, RuntimeError):
        return None

    reference = f'<mentioned_file path="{html.escape(str(path), quote=True)}"'
    if len(data) > MAX_INLINE_BYTES:
        return reference + " />"
    try:
        body = data.decode("utf-8")  # strict: binaries fall back to path only
    except UnicodeDecodeError:
        return reference + " />"
    return f"{reference}>\n{body}\n</mentioned_file>"
