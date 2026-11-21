"""The file tools, and resolving a tool's path argument against the agent's cwd.

Standard library only: the grep worker imports this module.
"""

import codecs
import fnmatch
import os
import re
import threading
import time
from pathlib import Path
from typing import Iterator

from .result import ToolError


def _resolve(path: str, cwd: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else cwd / p


def resolve_path(path: str, cwd: Path) -> Path:
    """The path a tool would actually touch, resolved against the agent's cwd.

    Public so UI previews share the exact resolution the execution uses; two
    implementations would eventually preview one file and modify another.
    """
    return _resolve(path, cwd)


def _real(path: Path) -> Path | None:
    """The path with symlinks resolved, or None when it cannot be resolved.

    None never compares equal to a recorded path, so an unresolvable path
    is held like any other instead of raising inside the gate.
    """
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _inside(path: Path, cwd: Path) -> bool:
    """True if path (symlinks resolved) is inside cwd.

    A path that cannot be resolved at all (a symlink loop raises RuntimeError,
    not OSError) counts as outside: gate() then holds the call instead
    of letting the exception escape the tool-call error boundary and end the
    turn.
    """
    try:
        return path.resolve().is_relative_to(cwd.resolve())
    except (OSError, RuntimeError, ValueError):
        return False


def _decode_preserving(raw: bytes, errors: str = "strict") -> tuple[str, str, bool]:
    """(LF-normalized text, line ending to restore, had a UTF-8 BOM).

    The model reads and writes LF, so matching and editing happen on the LF
    copy; the original ending and BOM are put back on write, and a CRLF file
    stays CRLF with only the target span changed. Reading and writing bytes
    keeps this independent of the platform and its locale — text mode would
    normalize every line ending in the file just for touching one line.
    Mixed-ending files are re-written uniformly with the majority ending.

    Strict by default: text decoded with replacement characters and written
    back would turn every byte that is not UTF-8 into U+FFFD, all over a file
    of which only one span was meant to change. Only callers that never write
    the text back may pass ``errors="replace"``.
    """
    bom = raw.startswith(codecs.BOM_UTF8)
    if bom:
        raw = raw[len(codecs.BOM_UTF8):]
    text = raw.decode("utf-8", errors=errors)
    crlf = text.count("\r\n")
    bare = text.count("\n") - crlf
    newline = "\r\n" if crlf > bare else "\n"
    return text.replace("\r\n", "\n"), newline, bom


def _encode_preserving(text: str, newline: str, bom: bool) -> bytes:
    if newline != "\n":
        text = text.replace("\n", newline)
    raw = text.encode("utf-8")
    return codecs.BOM_UTF8 + raw if bom else raw


# Read a line at a time and never more of one than this, so a file with no
# newline in it at all (a minified bundle, a database dump) costs this much
# memory rather than its whole size.
_READ_CHUNK = 64 * 1024


def _stopped(stop: threading.Event | None) -> bool:
    return stop is not None and stop.is_set()


def _numbered_lines(source, stop: threading.Event | None) -> Iterator[tuple[int, bytes, bool]]:
    """(line number, the line's first bytes without its ending, whether more
    of it was left unread) for every line of a file opened in binary mode.

    Only "\n" ends a line, "\r\n" being one way to write it: str.splitlines()
    also breaks at form feeds, \x1c-\x1e, NEL and the Unicode separators, and
    drops them, so line numbers would disagree with grep's and an old_string
    copied from this output would not be found by edit_file.
    """
    number = 0
    while not _stopped(stop):
        raw = source.readline(_READ_CHUNK)
        if not raw:
            return
        number += 1
        ended = raw.endswith(b"\n")
        clipped = False
        # The rest of a longer line is passed over, never held.
        while not ended and len(raw) == _READ_CHUNK and not _stopped(stop):
            more = source.readline(_READ_CHUNK)
            if not more:
                break
            clipped, ended = True, more.endswith(b"\n")
        if number == 1 and raw.startswith(codecs.BOM_UTF8):
            raw = raw[len(codecs.BOM_UTF8):]
        if raw.endswith(b"\n"):
            raw = raw[:-1].removesuffix(b"\r")
        yield number, raw, clipped


def _read_file(args: dict, cwd: Path, budget: int | None = None,
               stop: threading.Event | None = None) -> str:
    """The file's lines, numbered, read as a stream.

    ``budget`` is how many characters the result may run to. Reading ends
    there, with a line saying where to go on from, instead of the whole file
    being loaded for a result that is cut down to a fraction of it afterwards.
    ``stop`` ends the read early for a caller that is no longer waiting.
    """
    path = _resolve(args["path"], cwd)
    if not path.exists():
        raise ToolError(f"Error: file not found: {path}")
    # Refused rather than clamped or ignored: a limit of 0 read as "no limit"
    # hands back the whole file, and an offset of 0 is a model counting from
    # zero, which deserves to hear that lines count from 1.
    offset = int(args.get("offset", 1))
    if offset < 1:
        raise ToolError("Error: offset is a 1-indexed line number; it must be at least 1.")
    limit = args.get("limit")
    if limit is not None and int(limit) < 1:
        raise ToolError("Error: limit must be at least 1; leave it out to read to the end.")
    last = offset - 1 + int(limit) if limit is not None else None

    selected: list[str] = []
    size = width = 0
    note = ""
    with path.open("rb") as source:
        for number, raw, clipped in _numbered_lines(source, stop):
            if number < offset:
                continue
            if last is not None and number > last:
                break
            # Lossy on purpose: this text is shown, never written back.
            line = raw.decode("utf-8", errors="replace")
            if clipped:
                line += f" ... (line {number} is longer than {_READ_CHUNK} bytes; the rest is left out)"
            # Every number is padded to the width of the last one, so a line
            # whose number is a digit longer widens all the lines before it.
            digits = len(str(number))
            size += len(selected) * (digits - width)
            width = digits
            cost = len(line) + width + 3  # two spaces and the newline
            if budget is not None and size + cost > budget:
                if selected:
                    note = (f"... (stopped before line {number} to keep the result short; "
                            f"read on with offset={number})")
                    break
                line = line[:max(budget - 200, 0)]
                note = (f"... (line {number} is cut short to fit the result; "
                        f"read on with offset={number + 1})")
                selected.append(line)
                break
            selected.append(line)
            size += cost
    if not selected:
        return "(file is empty or offset past end of file)"
    width = len(str(offset + len(selected) - 1))
    text = "\n".join(f"{offset + i:>{width}}  {line}" for i, line in enumerate(selected))
    return f"{text}\n{note}" if note else text


def _write_file(args: dict, cwd: Path) -> str:
    path = _resolve(args["path"], cwd)
    content = str(args["content"]).replace("\r\n", "\n")
    newline, bom = "\n", False
    if path.is_file():
        # A full rewrite keeps the file's own conventions, the same way an
        # edit does; new files are plain LF UTF-8. Only the conventions are
        # taken from the old bytes, so a file that is not UTF-8 decodes
        # lossily here and is simply replaced with UTF-8 content.
        try:
            _, newline, bom = _decode_preserving(path.read_bytes(), errors="replace")
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_encode_preserving(content, newline, bom))
    n = content.count("\n") + 1
    return f"Wrote {n} lines to {path}"


def _edit_file(args: dict, cwd: Path) -> str:
    path = _resolve(args["path"], cwd)
    if not path.exists():
        raise ToolError(f"Error: file not found: {path}")
    try:
        text, newline, bom = _decode_preserving(path.read_bytes())
    except UnicodeDecodeError:
        raise ToolError(f"Error: {path} is not valid UTF-8 text, and editing it here would corrupt "
                        "every byte outside the edit. Use the shell to change it in its own "
                        "encoding.") from None
    old = str(args["old_string"]).replace("\r\n", "\n")
    new = str(args["new_string"]).replace("\r\n", "\n")
    count = text.count(old)
    if count == 0:
        raise ToolError("Error: old_string not found in file.")
    if count > 1:
        raise ToolError(f"Error: old_string is not unique (found {count} times). "
                        "Add more context to make it unique.")
    path.write_bytes(_encode_preserving(text.replace(old, new, 1), newline, bom))
    return f"Edited {path}"


# Directories never worth walking into for a code-search glob; they bury real
# results under dependency/VCS/build noise. Pass include_ignored=true to search them anyway.
_GLOB_IGNORE = {
    ".git", ".hg", ".svn",  # VCS
    ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",  # Python
    "node_modules", ".next", ".nuxt",  # JS/TS
    "target", "dist", "build", "out",  # build output (Rust/Java/JS/...)
    ".cache", ".gradle", ".idea",  # caches & IDE
}


# How long a glob may walk. The walk runs off the event loop, where a caller
# that stops waiting cannot interrupt it, so this is also what ends a walk
# nobody is waiting for any more.
_GLOB_TIMEOUT = 10.0

_GLOB_MAGIC = re.compile(r"[*?\[]")


class _GlobTimeout(Exception):
    pass


def _glob_files(base: Path, parts: tuple[str, ...], skip: set[str],
                expired) -> Iterator[Path]:
    """Files under ``base`` matching the pattern's ``parts``, the way
    Path.glob matches them, except that a directory named in ``skip`` is never
    entered: filtering its files out afterwards means walking all of
    node_modules to throw every result away.

    "**" matches any number of directories and does not follow a symlink to
    one, so the walk ends; a named or wildcard part does follow it.
    """
    matchers = [None if part == "**" or not _GLOB_MAGIC.search(part)
                else re.compile(fnmatch.translate(os.path.normcase(part))).match
                for part in parts]

    def entries(directory: Path) -> list[os.DirEntry]:
        try:
            with os.scandir(directory) as found:
                return list(found)
        except OSError:
            return []

    def is_dir(entry: os.DirEntry, follow: bool) -> bool:
        try:
            return entry.is_dir(follow_symlinks=follow)
        except OSError:
            return False

    def walk(directory: Path, index: int) -> Iterator[Path]:
        if expired():
            raise _GlobTimeout
        part, last = parts[index], index == len(parts) - 1
        if part == "**":
            yield from walk(directory, index + 1)
            for entry in entries(directory):
                if entry.name not in skip and is_dir(entry, follow=False):
                    yield from walk(Path(entry.path), index)
        elif (match := matchers[index]) is None:
            child = directory / part
            if last:
                if child.is_file():
                    yield child
            elif part not in skip and child.is_dir():
                yield from walk(child, index + 1)
        else:
            for entry in entries(directory):
                if not match(os.path.normcase(entry.name)):
                    continue
                if last:
                    try:
                        if entry.is_file():
                            yield Path(entry.path)
                    except OSError:
                        continue
                elif entry.name not in skip and is_dir(entry, follow=True):
                    yield from walk(Path(entry.path), index + 1)

    yield from walk(base, 0)


def _glob(args: dict, cwd: Path, sandboxed: bool = False,
          stop: threading.Event | None = None) -> str:
    base = _resolve(args["path"], cwd) if args.get("path") else cwd
    if not base.is_dir():
        raise ToolError(f"Error: not a directory: {base}")
    pattern = Path(str(args["pattern"]))
    if pattern.anchor or not pattern.parts:
        raise ToolError("Error: pattern must be a relative glob such as '**/*.py'; "
                        "pass the directory to search as path.")
    # "a/**/**/b" is "a/**/b", and walking it twice finds everything twice.
    parts = tuple(part for i, part in enumerate(pattern.parts)
                  if not (part == "**" and i and pattern.parts[i - 1] == "**"))
    if parts[-1] == "**":
        # Everything below, as Path.glob reads it from Python 3.13 on; before
        # that it meant directories only, which lists no file at all.
        parts += ("*",)
    # The ignore list names directories: a file called "build" is matched.
    skip = set() if args.get("include_ignored") else _GLOB_IGNORE
    deadline = time.monotonic() + _GLOB_TIMEOUT

    def expired() -> bool:
        return _stopped(stop) or time.monotonic() >= deadline

    # Timed in one pass, and a file gone since it was listed is left out:
    # it is no longer there to match, and one stat failing must not fail
    # the whole call.
    dated = []
    try:
        for p in dict.fromkeys(_glob_files(base, parts, skip, expired)):
            # sandboxed keeps symlinks under base from listing files outside it
            if sandboxed and not _inside(p, base):
                continue
            try:
                dated.append((p.stat().st_mtime, p))
            except OSError:
                continue
    except _GlobTimeout:
        raise ToolError(f"Error: glob timed out after {_GLOB_TIMEOUT:g} seconds with "
                        f"{len(dated)} files matched so far. Narrow the pattern or the "
                        "path.") from None
    if not dated:
        return "(no files matched)"
    dated.sort(key=lambda item: item[0], reverse=True)
    return "\n".join(str(p) for _, p in dated)
