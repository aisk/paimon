"""The grep tool.

Standard library only, and so is everything it imports: the search runs in a
worker process (paimon._grep_worker) started for every call, and pulling in
pydantic or pydantic-ai there would cost about a second per search.
"""

import asyncio
import codecs
import fnmatch
import json
import os
import re
import sys
from pathlib import Path

from .files import _GLOB_IGNORE, _inside, _resolve
from .result import ToolError


_GREP_DEFAULT_RESULTS = 50
_GREP_MAX_RESULTS = 200
_GREP_MAX_LINE = 300  # chars of a matched line shown before it is clipped
# Files larger than this are skipped: a multi-hundred-megabyte artifact would
# stall the loop for a search that is about source code.
_GREP_MAX_FILE_BYTES = 10_000_000


def _grep_files(base: Path, name_filter: str | None, sandboxed: bool):
    """Files under ``base``, in stable directory and filename order."""
    # Prune before descent; sort only one directory at a time. Directory
    # symlinks are not followed, including cycles.
    for root, dirs, names in os.walk(base, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in _GLOB_IGNORE)
        for name in sorted(names):
            if name in _GLOB_IGNORE or (name_filter and not fnmatch.fnmatch(name, name_filter)):
                continue
            path = Path(root) / name
            if sandboxed and not _inside(path, base):
                continue
            if path.is_file():
                yield path


def _grep(args: dict, cwd: Path, sandboxed: bool = False) -> str:
    """Synchronous worker implementation; tool calls use _grep_async."""
    try:
        pattern = re.compile(str(args["pattern"]))
    except re.error as exc:
        raise ToolError(f"Error: invalid regular expression: {exc}") from None
    base = _resolve(args["path"], cwd) if args.get("path") else cwd
    try:
        limit = int(args.get("max_results") or _GREP_DEFAULT_RESULTS)
    except (TypeError, ValueError):
        limit = _GREP_DEFAULT_RESULTS
    limit = max(1, min(limit, _GREP_MAX_RESULTS))
    if base.is_file():
        files = iter([base])
    elif base.is_dir():
        files = _grep_files(base, str(args["glob"]) if args.get("glob") else None, sandboxed)
    else:
        raise ToolError(f"Error: no such file or directory: {base}")

    lines: list[str] = []
    skipped_large = 0
    truncated = False
    for path in files:
        try:
            with path.open("rb") as source:
                if os.fstat(source.fileno()).st_size > _GREP_MAX_FILE_BYTES:
                    skipped_large += 1
                    continue
                if b"\0" in source.read(8192):
                    continue  # binary
                source.seek(0)
                consumed = 0
                lineno = 0
                while True:
                    # Bound reads even when the file grows after the size check.
                    raw = source.readline(_GREP_MAX_FILE_BYTES - consumed + 1)
                    if not raw:
                        break
                    consumed += len(raw)
                    if consumed > _GREP_MAX_FILE_BYTES:
                        skipped_large += 1
                        break
                    lineno += 1
                    if lineno == 1:
                        raw = raw.removeprefix(codecs.BOM_UTF8)
                    # Only LF/CRLF delimit lines, not Unicode separators.
                    line = raw.decode("utf-8", errors="replace")
                    if line.endswith("\n"):
                        line = line[:-1].removesuffix("\r")
                    if not pattern.search(line):
                        continue
                    if len(lines) == limit:
                        truncated = True
                        break
                    shown = line.strip()
                    if len(shown) > _GREP_MAX_LINE:
                        shown = shown[:_GREP_MAX_LINE] + "…"
                    lines.append(f"{path}:{lineno}:{shown}")
        except OSError:
            continue
        if truncated:
            break
    if not lines:
        note = f" ({skipped_large} large files skipped)" if skipped_large else ""
        return f"(no matches){note}"
    if truncated:
        lines.append(f"[first {limit} matches shown; narrow the pattern, filter by "
                     "glob, or raise max_results]")
    if skipped_large:
        lines.append(f"[{skipped_large} files over {_GREP_MAX_FILE_BYTES // 1_000_000}MB skipped]")
    return "\n".join(lines)


_GREP_TIMEOUT = 10.0


async def _grep_async(args: dict, cwd: Path, sandboxed: bool = False) -> str:
    """Use a killable process: a regex can hold the GIL indefinitely.

    The worker writes the result text as a JSON string, or an object with the
    error text when the search failed (see paimon._grep_worker).
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "paimon._grep_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        payload = json.dumps([args, str(cwd), sandboxed]).encode("utf-8")
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(payload), _GREP_TIMEOUT)
        except asyncio.TimeoutError:
            raise ToolError("Error: search timed out; simplify the regular expression or "
                            "narrow the path/glob.", detail="timed out") from None
        if proc.returncode:
            raise ToolError("Error: search worker failed: "
                            f"{stderr.decode('utf-8', errors='replace')[:1000]}")
        result = json.loads(stdout)
        if isinstance(result, dict):
            raise ToolError(str(result.get("error")))
        return result
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        await proc.wait()
