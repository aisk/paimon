"""The file tools, and resolving a tool's path argument against the agent's cwd.

Standard library only: the grep worker imports this module.
"""

import codecs
from pathlib import Path

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


def _read_file(args: dict, cwd: Path) -> str:
    path = _resolve(args["path"], cwd)
    if not path.exists():
        raise ToolError(f"Error: file not found: {path}")
    text, _, _ =_decode_preserving(path.read_bytes(), errors="replace")
    lines = text.splitlines()
    offset = max(1, int(args.get("offset", 1)))
    limit = args.get("limit")
    end = offset - 1 + int(limit) if limit else len(lines)
    selected = lines[offset - 1 : end]
    if not selected:
        return "(file is empty or offset past end of file)"
    width = len(str(offset + len(selected) - 1))
    return "\n".join(f"{offset + i:>{width}}  {line}" for i, line in enumerate(selected))


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


def _glob(args: dict, cwd: Path, sandboxed: bool = False) -> str:
    base = _resolve(args["path"], cwd) if args.get("path") else cwd
    if not base.is_dir():
        raise ToolError(f"Error: not a directory: {base}")
    skip = set() if args.get("include_ignored") else _GLOB_IGNORE
    matches = [
        p
        for p in base.glob(args["pattern"])
        if p.is_file()
        and not skip.intersection(p.relative_to(base).parts)
        # sandboxed keeps symlinks under base from listing files outside it
        and (not sandboxed or _inside(p, base))
    ]
    if not matches:
        return "(no files matched)"
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return "\n".join(str(p) for p in matches)
