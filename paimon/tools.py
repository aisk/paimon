"""Tool definitions and execution.

Each tool is one ``Tool`` entry in ``REGISTRY``: the OpenAI-style JSON schema
sent to the model, the function that runs it, and the access class that drives
permission gating. Adding a tool means adding one registry entry.
"""

import asyncio
import inspect
import json
import os
import shlex
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

from pydantic_ai.tools import ToolDefinition

# A confirm callback returns True to allow a dangerous tool, False to deny.
ConfirmFn = Callable[[str, dict], Awaitable[bool]]

# Permission modes: read (confirm writes, shell and reads outside cwd),
# edit (auto-approve writes inside cwd), yolo (no confirmation at all).
# In read and edit modes, shell commands recognized by safe_command() run
# without confirmation unless the safe_commands toggle is off.
MODES = ("read", "edit", "yolo")

MAX_OUTPUT = 30_000  # truncate tool output sent back to the model


@dataclass(frozen=True)
class Tool:
    """One tool: its model-facing schema, executor, and gating class.

    ``run`` takes (args, cwd, mode) and returns a string or an awaitable of
    one; it is None for tools the agent loop handles itself (write_todos).
    ``access`` drives gate(): "read" runs freely inside cwd, "write" is
    auto-approved inside cwd in edit mode, "execute" needs confirmation
    outside yolo except for commands safe_command() recognizes as read-only,
    "none" is never gated, "always" needs confirmation even in yolo mode.
    """

    schema: dict
    run: Optional[Callable[[dict, Path, str], object]]
    access: str = "none"


_TODO_MARKERS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def render_todos(todos: list[dict]) -> str:
    """Plain-text rendering of the todo list, used as the tool result the model sees."""
    if not todos:
        return "(todo list cleared)"
    return "\n".join(f"{_TODO_MARKERS.get(t.get('status'), '[ ]')} {t.get('content', '')}" for t in todos)


def summarize_call(name: str, args: dict, limit: Optional[int] = None) -> str:
    """One-line detail for a tool call, shared by the TUI and headless output.

    With a limit the detail is collapsed onto a single line and truncated,
    for outputs that cannot reflow (a terminal stream) unlike a TUI widget.
    """
    detail = str(args.get("command") or args.get("path") or json.dumps(args, ensure_ascii=False))
    if limit is None:
        return detail
    detail = " ".join(detail.split())
    return detail if len(detail) <= limit else detail[: limit - 1] + "…"


def _resolve(path: str, cwd: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else cwd / p


def _inside(path: Path, cwd: Path) -> bool:
    """True if path (symlinks resolved) is inside cwd."""
    try:
        return path.resolve().is_relative_to(cwd.resolve())
    except OSError:
        return False


# Any of these in the raw command string means the argv view built below may
# not be what the shell actually runs (substitution, redirection, chaining,
# escaping, brace expansion, glob character classes that can match ".."), so
# the command is never auto-allowed. A false rejection only costs one
# confirmation prompt, so this errs hard toward rejection.
_SHELL_METACHARS = frozenset("$`;&|<>()\\{[\n\r")


@dataclass(frozen=True)
class _SafeCommandSpec:
    """What still needs checking for a command that is read-only by nature."""

    deny_flags: tuple[str, ...] = ()  # flags that write, execute, or hang
    check_paths: bool = True  # non-flag args must resolve inside cwd


# Commands with no exec/write capability at argv level, or whose few
# dangerous flags are denied below. Deliberately absent: sed (-i), awk
# (system()), sort (-o writes), uniq (second positional is an output file),
# xargs, tee, less/more (! shell escape), curl/wget (network), env/printenv
# (would dump API keys into model context).
_SAFE_COMMANDS: dict[str, _SafeCommandSpec] = {
    "ls": _SafeCommandSpec(),
    "pwd": _SafeCommandSpec(),
    "cat": _SafeCommandSpec(),
    "head": _SafeCommandSpec(),
    "tail": _SafeCommandSpec(deny_flags=("-f", "-F", "--follow")),  # hangs until timeout
    "wc": _SafeCommandSpec(),
    "stat": _SafeCommandSpec(),
    "file": _SafeCommandSpec(deny_flags=("-C", "--compile")),  # -C writes a .mgc file
    "which": _SafeCommandSpec(check_paths=False),  # args are command names, not paths
    "du": _SafeCommandSpec(),
    "df": _SafeCommandSpec(),
    "grep": _SafeCommandSpec(),
    "rg": _SafeCommandSpec(deny_flags=("--pre", "--hostname-bin")),  # both execute programs
    # The complete set of GNU find actions with side effects; everything
    # else prints to stdout or filters.
    "find": _SafeCommandSpec(deny_flags=("-exec", "-execdir", "-ok", "-okdir", "-delete",
                                         "-fls", "-fprint", "-fprint0", "-fprintf")),
    "tree": _SafeCommandSpec(deny_flags=("-o",)),  # -o writes the listing to a file
    "diff": _SafeCommandSpec(),
    "readlink": _SafeCommandSpec(),
    "realpath": _SafeCommandSpec(),
    "echo": _SafeCommandSpec(check_paths=False),
    "uname": _SafeCommandSpec(check_paths=False),
    "whoami": _SafeCommandSpec(check_paths=False),
    # -s sets the clock; -f reads a file, so its paths are checked like any other.
    "date": _SafeCommandSpec(deny_flags=("-s", "--set")),
    "nproc": _SafeCommandSpec(check_paths=False),
}

# "grep" is deliberately absent: its -O/--open-files-in-pager runs an
# arbitrary command, and git's option bundling (-nOsh) makes that hard to
# screen reliably. Plain grep and rg cover the same ground.
_SAFE_GIT_SUBCOMMANDS = frozenset({
    "status", "log", "diff", "show", "blame", "shortlog", "describe",
    "rev-parse", "ls-files", "ls-tree", "branch", "remote", "tag",
})
# --ext-diff/--textconv run configured external programs, --show-signature
# runs gpg.program, --no-index reads files outside the repo. (--output and
# --output-directory write files; matched by prefix below.)
_UNSAFE_GIT_FLAGS = ("--ext-diff", "--textconv", "--no-index", "--show-signature")
# Options that read a file named on the command line, which would otherwise
# escape the cwd check git's positional args cannot get (refs vs paths).
_GIT_DENY_FLAGS = {
    "blame": ("--contents",),
    "ls-files": ("-X", "--exclude-from"),
}
# Subcommands whose bare/flag form lists but whose positional args mutate
# (`git branch NAME` creates, `git tag NAME` tags, `git remote add` adds):
# every token after the subcommand must come from the listed set.
_GIT_FLAGS_ONLY = {
    "branch": frozenset({"-a", "--all", "-r", "--remotes", "-v", "-vv",
                         "--verbose", "-l", "--list", "--show-current"}),
    "remote": frozenset({"-v", "--verbose"}),
    "tag": frozenset({"-l", "--list", "-n"}),
}


def _flag_denied(tok: str, deny_flags: tuple[str, ...]) -> bool:
    """True if tok matches a denied flag, including combined shorts ("-20f")."""
    if tok.split("=", 1)[0] in deny_flags:
        return True
    if tok.startswith("--"):
        return False
    letters = {f[1] for f in deny_flags if len(f) == 2}
    return any(ch in letters for ch in tok[1:])


def _hidden_glob(tok: str) -> bool:
    """True if a path segment could expand to ".." and escape the cwd check.

    The token is checked literally, so "cwd/.*" looks contained, but the
    shell expands it to "." and "..". Only patterns whose segment starts with
    a dot can match "..", since "*" does not match leading dots.
    """
    return any(seg.startswith(".") and ("*" in seg or "?" in seg) for seg in tok.split("/"))


def _safe_git(argv: list[str]) -> bool:
    """argv is everything after "git". Global options before the subcommand
    (-c, -C, --git-dir, --exec-path, ...) all start with "-" and are rejected
    by the membership test on argv[0]."""
    if not argv or argv[0] not in _SAFE_GIT_SUBCOMMANDS:
        return False
    flags_only = _GIT_FLAGS_ONLY.get(argv[0])
    if flags_only is not None:
        return all(tok in flags_only for tok in argv[1:])
    deny = _UNSAFE_GIT_FLAGS + _GIT_DENY_FLAGS.get(argv[0], ())
    for tok in argv[1:]:
        if not tok.startswith("-"):
            continue  # a ref or a pathspec; git's own repo boundary applies
        if tok.split("=", 1)[0].startswith("--output") or _flag_denied(tok, deny):
            return False
        if "/" in tok:
            return False  # an attached path value (--contents=/etc/passwd)
    return True


def safe_command(command: str, cwd: Path) -> bool:
    """Conservatively decide whether a shell command is clearly read-only.

    A guardrail against agent mistakes, not a security boundary: anything not
    positively recognized returns False and the caller falls back to the
    normal confirmation flow, so a miss costs a prompt, never a denial.
    """
    if any(ch in _SHELL_METACHARS for ch in command):
        return False
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if not argv:
        return False
    # Tilde expands at word start and after "=" in bash; reject both
    # positions rather than guess what /bin/sh does. Mid-token tildes
    # (git's HEAD~1) stay allowed.
    if any(tok.startswith("~") or "=~" in tok for tok in argv):
        return False
    if argv[0] == "git":
        return _safe_git(argv[1:])
    spec = _SAFE_COMMANDS.get(argv[0])  # exact match: "./ls", "/bin/ls", "FOO=1 ls" miss
    if spec is None:
        return False
    for tok in argv[1:]:
        if tok.startswith("-"):
            # A flag carrying an attached path (-f/etc/passwd, --from-file=/x)
            # would never reach the containment check below.
            if _flag_denied(tok, spec.deny_flags) or "/" in tok:
                return False
        elif _hidden_glob(tok):
            return False
        elif spec.check_paths and not _inside(_resolve(tok, cwd), cwd):
            return False
    return True


def gate(name: str, args: dict, mode: str, cwd: Path,
         registry: Optional[dict[str, Tool]] = None,
         safe_commands: bool = True) -> str:
    """Decide whether a tool call runs freely ("allow") or needs user confirmation ("confirm").

    ``safe_commands`` auto-allows shell commands recognized as clearly
    read-only (see safe_command); False restores confirm-everything behavior.
    """
    tool = (REGISTRY if registry is None else registry).get(name)
    if tool is not None and tool.access == "always":
        return "confirm"
    if mode == "yolo" or tool is None or tool.access == "none":
        return "allow"
    if tool.access == "execute":
        if safe_commands and safe_command(str(args.get("command") or ""), cwd):
            return "allow"
        return "confirm"
    # A missing/malformed path resolves to cwd itself; the tool then fails on its own.
    inside = _inside(_resolve(str(args.get("path") or ""), cwd), cwd)
    if tool.access == "read":
        return "allow" if inside else "confirm"
    if tool.access == "write" and mode == "edit" and inside:
        return "allow"
    return "confirm"


async def run_tool(name: str, args: dict, cwd: Path, mode: str,
                   confirm: Optional[ConfirmFn] = None,
                   registry: Optional[dict[str, Tool]] = None,
                   safe_commands: bool = True) -> tuple[str, bool]:
    """Gate, optionally confirm, then execute a tool call.

    Returns ``(result, denied)``. This is the enforcement point: a call that
    needs confirmation is denied when no confirm hook is available, so a
    headless Agent cannot bypass the permission mode. ``registry`` narrows the
    available tools (an agent's own set); None means the full REGISTRY.
    """
    if gate(name, args, mode, cwd, registry, safe_commands=safe_commands) == "confirm":
        allowed = await confirm(name, args) if confirm else False
        if not allowed:
            return "User denied this operation.", True
    return await execute_tool(name, args, cwd, mode=mode, registry=registry), False


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


def _glob(args: dict, cwd: Path, sandboxed: bool = False) -> str:
    base = _resolve(args["path"], cwd) if args.get("path") else cwd
    if not base.is_dir():
        return f"Error: not a directory: {base}"
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


_COMMAND_TIMEOUT = 120.0
_KILL_GRACE = 2.0  # seconds to wait after SIGTERM before forcing SIGKILL
_KILL_TIMEOUT = 2.0  # maximum wait to reap the process after SIGKILL


def _signal_group(pgid: int, sig: int, proc: asyncio.subprocess.Process) -> None:
    """Send a signal to the whole process group, falling back to the child alone."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


async def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Terminate and reap a command tree without allowing cleanup to hang."""
    if proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    _signal_group(pgid, signal.SIGTERM, proc)
    try:
        await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=_KILL_GRACE)
        return
    except asyncio.TimeoutError:
        _signal_group(pgid, signal.SIGKILL, proc)
    try:
        await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=_KILL_TIMEOUT)
    except asyncio.TimeoutError:
        # The OS should eventually reap it; most importantly, cleanup cannot
        # block the agent indefinitely.
        pass


async def _shell(args: dict, cwd: Path) -> str:
    # start_new_session puts the child in its own process group so we can kill
    # the whole tree (the shell plus anything it spawns) on timeout/interrupt.
    proc = await asyncio.create_subprocess_shell(
        args["command"],
        cwd=str(cwd),
        # Without this a command that reads stdin (a bare "cat") would block
        # until the timeout, eating the user's keystrokes in the TUI.
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_COMMAND_TIMEOUT)
    except asyncio.TimeoutError:
        await _kill_tree(proc)
        return f"Error: command timed out after {_COMMAND_TIMEOUT:g}s."
    except asyncio.CancelledError:
        await _kill_tree(proc)
        raise
    out = stdout.decode(errors="replace")
    status = f"(exit code {proc.returncode})"
    return f"{out}\n{status}" if out.strip() else status


async def execute_tool(name: str, args: dict, cwd: Path, mode: str = "yolo",
                       registry: Optional[dict[str, Tool]] = None) -> str:
    """Run a registered tool. Always returns a string for the model."""
    tool = (REGISTRY if registry is None else registry).get(name)
    if tool is None or tool.run is None:
        return f"Error: unknown tool {name!r}"
    try:
        result = tool.run(args, cwd, mode)
        if inspect.isawaitable(result):
            result = await result
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface any tool error to the model
        return f"Error executing {name}: {exc}"
    if len(result) > MAX_OUTPUT:
        result = result[:MAX_OUTPUT] + f"\n... (truncated, {len(result) - MAX_OUTPUT} more chars)"
    return result


REGISTRY: dict[str, Tool] = {
    "read_file": Tool(
        access="read",
        run=lambda args, cwd, mode: _read_file(args, cwd),
        schema={
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
    ),
    "write_file": Tool(
        access="write",
        run=lambda args, cwd, mode: _write_file(args, cwd),
        schema={
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
    ),
    "edit_file": Tool(
        access="write",
        run=lambda args, cwd, mode: _edit_file(args, cwd),
        schema={
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
    ),
    "glob": Tool(
        access="read",
        run=lambda args, cwd, mode: _glob(args, cwd, sandboxed=mode != "yolo"),
        schema={
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
    ),
    "shell": Tool(
        access="execute",
        run=lambda args, cwd, mode: _shell(args, cwd),
        schema={
            "type": "function",
            "function": {
                "name": "shell",
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
    ),
    # Stateful: mutates agent-held state, so the agent loop runs it itself.
    "write_todos": Tool(
        run=None,
        schema={
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
    ),
    # Stateful: ends the session, so the agent loop handles it.
    "start_new_session": Tool(
        run=None,
        access="always",
        schema={
            "type": "function",
            "function": {
                "name": "start_new_session",
                "description": (
                    "Hand off to a fresh session: end this one and start a new empty session "
                    "whose first user message is your prompt. Use when most of the conversation "
                    "so far is irrelevant to the next phase of work — a focused handoff prompt "
                    "beats carrying a long history forward. The prompt must be self-contained: "
                    "state the goal, key file paths, decisions already made, and current status; "
                    "the new session has no memory of this one. Requires explicit user "
                    "confirmation and only works in the interactive UI (always denied in "
                    "non-interactive runs)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The first user message for the new session; must be self-contained.",
                        },
                    },
                    "required": ["prompt"],
                },
            },
        },
    ),
}

def schemas(registry: dict[str, Tool]) -> list[dict]:
    """The OpenAI-style schema list for a registry, in registry order."""
    return [tool.schema for tool in registry.values()]


def definitions(registry: dict[str, Tool]) -> list[ToolDefinition]:
    """The same schemas as pydantic-ai tool definitions."""
    return [
        ToolDefinition(
            name=tool.schema["function"]["name"],
            description=tool.schema["function"]["description"],
            parameters_json_schema=tool.schema["function"]["parameters"],
        )
        for tool in registry.values()
    ]
