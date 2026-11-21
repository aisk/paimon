"""The file tools, and resolving a tool's path argument against the agent's cwd.

Standard library only: the grep worker imports this module.
"""

import codecs
import fnmatch
import os
import re
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


def _read_file(args: dict, cwd: Path, budget: int | None = None) -> str:
    """The file's lines, numbered.

    Read as a stream, and no further than ``budget`` characters of result: a
    longer file ends with a line saying where to read on from, instead of
    being loaded whole for a result that is cut to a fraction of it afterwards.
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

    lines: list[str] = []
    chars = 0
    note = ""
    with path.open("rb") as source:
        # A file read as bytes ends a line at "\n" and nowhere else.
        # str.splitlines() also breaks at form feeds, \x1c-\x1e, NEL and the
        # Unicode separators, and drops them, so line numbers would disagree
        # with grep's and an old_string copied from this output would not be
        # found by edit_file.
        for number, raw in enumerate(source, 1):
            if number < offset:
                continue
            if last is not None and number > last:
                break
            if number == 1:
                raw = raw.removeprefix(codecs.BOM_UTF8)
            raw = raw[:-2] if raw.endswith(b"\r\n") else raw.removesuffix(b"\n")
            # Lossy on purpose: this text is shown, never written back.
            line = raw.decode("utf-8", errors="replace")
            chars += len(line)
            # Each line also carries its number, two spaces and a newline.
            framing = (len(lines) + 1) * (len(str(number)) + 3)
            if budget is not None and chars + framing > budget:
                if lines:
                    note = (f"... (stopped before line {number} to keep the result short; "
                            f"read on with offset={number})")
                else:
                    lines.append(line[:budget])
                    note = (f"... (line {number} is cut short to fit the result; "
                            f"read on with offset={number + 1})")
                break
            lines.append(line)
    if not lines:
        return "(file is empty or offset past end of file)"
    width = len(str(offset + len(lines) - 1))
    text = "\n".join(f"{offset + i:>{width}}  {line}" for i, line in enumerate(lines))
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


# How long a glob may walk. It runs in a thread, which nothing can interrupt,
# so this is also what ends a walk whose caller has stopped waiting.
_GLOB_TIMEOUT = 10.0

_GLOB_MAGIC = re.compile(r"[*?\[]")


def _subdirectories(directory: Path) -> list[Path]:
    """The directories in ``directory``, leaving out symlinks to one."""
    try:
        with os.scandir(directory) as entries:
            return [Path(entry.path) for entry in entries
                    if entry.is_dir(follow_symlinks=False)]
    except OSError:
        return []


def _named(directory: Path, part: str) -> list[Path]:
    """What one part of a pattern names in ``directory``."""
    if not _GLOB_MAGIC.search(part):
        return [directory / part]
    try:
        return [directory / name for name in fnmatch.filter(os.listdir(directory), part)]
    except OSError:
        return []


def _glob_files(directory: Path, parts: tuple[str, ...], skip: set[str],
                deadline: float) -> Iterator[Path]:
    """The files under ``directory`` that ``parts`` match, as Path.glob matches
    them, except that a directory named in ``skip`` is never entered:
    filtering its files out afterwards means walking all of node_modules to
    throw every result away.
    """
    if time.monotonic() >= deadline:
        raise ToolError(f"Error: glob timed out after {_GLOB_TIMEOUT:g} seconds. "
                        "Narrow the pattern or the path.")
    part, rest = parts[0], parts[1:]
    if part == "**":
        # No directory at all, then each one below. Not through a symlink,
        # which is what makes the walk end.
        yield from _glob_files(directory, rest, skip, deadline)
        for child in _subdirectories(directory):
            if child.name not in skip:
                yield from _glob_files(child, parts, skip, deadline)
        return
    for child in _named(directory, part):
        if not rest:
            if child.is_file():
                yield child
        elif child.name not in skip and child.is_dir():
            yield from _glob_files(child, rest, skip, deadline)


def _glob(args: dict, cwd: Path, sandboxed: bool = False) -> str:
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
    found = _glob_files(base, parts, skip, time.monotonic() + _GLOB_TIMEOUT)
    # Timed in one pass, and a file gone since it was listed is left out:
    # it is no longer there to match, and one stat failing must not fail
    # the whole call.
    dated = []
    for p in dict.fromkeys(found):  # "**/a/**" reaches a file by more than one route
        # sandboxed keeps symlinks under base from listing files outside it
        if sandboxed and not _inside(p, base):
            continue
        try:
            dated.append((p.stat().st_mtime, p))
        except OSError:
            continue
    if not dated:
        return "(no files matched)"
    dated.sort(key=lambda item: item[0], reverse=True)
    return "\n".join(str(p) for _, p in dated)
