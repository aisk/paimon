"""Tool definitions and execution.

Each tool is described with an OpenAI-style JSON schema (sent to the model) and
implemented by a small Python function. ``execute_tool`` dispatches by name.
"""

import asyncio
import os
import signal
from pathlib import Path

# Tools whose side effects warrant a user confirmation before running.
DANGEROUS = {"bash", "write_file", "edit_file"}

MAX_OUTPUT = 30_000  # truncate tool output sent back to the model

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file and return its contents with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the working directory or absolute."},
                    "offset": {"type": "integer", "description": "1-indexed line to start from (optional)."},
                    "limit": {"type": "integer", "description": "Maximum number of lines to read (optional)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact substring in a file. old_string must appear exactly once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string", "description": "Exact text to replace (must be unique in the file)."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern (e.g. '**/*.py', 'src/**/*.ts'). Returns matching paths sorted by most recently modified first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern. Use '**' to match any number of directories."},
                    "path": {"type": "string", "description": "Base directory to search in (optional, defaults to the working directory)."},
                    "include_ignored": {"type": "boolean", "description": "Search inside noise dirs like node_modules/.venv/.git too (optional, default false)."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the working directory and return its combined stdout/stderr. Use this for listing, searching (grep/find/ls), git, running tests, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_todos",
            "description": (
                "Create or update the task list for a multi-step task. Always pass the COMPLETE list; "
                "it overwrites the previous one. Use it to plan work and show progress on tasks with 3+ "
                "steps; skip it for trivial single-step requests. Keep exactly one task in_progress at a time, "
                "and mark a task completed as soon as it is done."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The complete task list, in order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "Short description of the task."},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                            },
                            "required": ["content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
]


_TODO_MARKERS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def render_todos(todos: list[dict]) -> str:
    """Plain-text rendering of the todo list, used as the tool result the model sees."""
    if not todos:
        return "(todo list cleared)"
    return "\n".join(f"{_TODO_MARKERS.get(t.get('status'), '[ ]')} {t.get('content', '')}" for t in todos)


def _resolve(path: str, cwd: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else cwd / p


def _read_file(args: dict, cwd: Path) -> str:
    path = _resolve(args["path"], cwd)
    if not path.exists():
        return f"Error: file not found: {path}"
    lines = path.read_text(errors="replace").splitlines()
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"])
    n = args["content"].count("\n") + 1
    return f"Wrote {n} lines to {path}"


def _edit_file(args: dict, cwd: Path) -> str:
    path = _resolve(args["path"], cwd)
    if not path.exists():
        return f"Error: file not found: {path}"
    text = path.read_text()
    old = args["old_string"]
    count = text.count(old)
    if count == 0:
        return "Error: old_string not found in file."
    if count > 1:
        return f"Error: old_string is not unique (found {count} times). Add more context to make it unique."
    path.write_text(text.replace(old, args["new_string"], 1))
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


def _glob(args: dict, cwd: Path) -> str:
    base = _resolve(args["path"], cwd) if args.get("path") else cwd
    if not base.is_dir():
        return f"Error: not a directory: {base}"
    skip = set() if args.get("include_ignored") else _GLOB_IGNORE
    matches = [
        p
        for p in base.glob(args["pattern"])
        if p.is_file() and not skip.intersection(p.relative_to(base).parts)
    ]
    if not matches:
        return "(no files matched)"
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return "\n".join(str(p) for p in matches)


_KILL_GRACE = 2.0  # seconds to wait after SIGTERM before forcing SIGKILL


def _signal_group(pgid: int, sig: int, proc: asyncio.subprocess.Process) -> None:
    """Send a signal to the whole process group, falling back to the child alone."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Terminate the command and any children (its process group), escalating
    SIGTERM -> SIGKILL. The SIGKILL backstop is scheduled on the event loop so it
    still fires even though the turn that called this is being cancelled."""
    if proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    _signal_group(pgid, signal.SIGTERM, proc)
    asyncio.get_running_loop().call_later(_KILL_GRACE, _signal_group, pgid, signal.SIGKILL, proc)


async def _bash(args: dict, cwd: Path) -> str:
    # start_new_session puts the child in its own process group so we can kill
    # the whole tree (the shell plus anything it spawns) on timeout/interrupt.
    proc = await asyncio.create_subprocess_shell(
        args["command"],
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        _kill_tree(proc)
        return "Error: command timed out after 120s."
    except asyncio.CancelledError:
        _kill_tree(proc)
        raise
    out = stdout.decode(errors="replace")
    status = f"(exit code {proc.returncode})"
    return f"{out}\n{status}" if out.strip() else status


async def execute_tool(name: str, args: dict, cwd: Path) -> str:
    """Dispatch a tool call. Always returns a string for the model."""
    try:
        if name == "read_file":
            return _read_file(args, cwd)
        if name == "write_file":
            return _write_file(args, cwd)
        if name == "edit_file":
            return _edit_file(args, cwd)
        if name == "glob":
            return _glob(args, cwd)
        if name == "bash":
            result = await _bash(args, cwd)
        else:
            return f"Error: unknown tool {name!r}"
    except Exception as exc:  # noqa: BLE001 — surface any tool error to the model
        return f"Error executing {name}: {exc}"
    if len(result) > MAX_OUTPUT:
        result = result[:MAX_OUTPUT] + f"\n... (truncated, {len(result) - MAX_OUTPUT} more chars)"
    return result
