"""Tool definitions and execution.

Each tool is one ``Tool`` entry in ``REGISTRY``: the OpenAI-style JSON schema
sent to the model, the function that runs it, and the access class that drives
permission gating. Adding a tool means adding one registry entry.
"""

import asyncio
import codecs
import fnmatch
import inspect
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Awaitable, Callable, Optional

from pydantic_ai.tools import ToolDefinition

from .session import Session, data_dir

# A confirm callback returns True to allow a dangerous tool, False to deny.
ConfirmFn = Callable[[str, dict], Awaitable[bool]]

# Permission modes: read (confirm writes, shell and reads outside cwd),
# edit (auto-approve writes inside cwd), yolo (no confirmation at all).
# In read and edit modes, shell commands recognized by safe_command() run
# without confirmation unless the safe_commands toggle is off.
MODES = ("read", "edit", "yolo")

MAX_OUTPUT = 30_000  # truncate tool output sent back to the model

# Bounds for wait_for_job. A wait that cannot expire is a deadlock waiting to
# happen: the agent being waited on may be blocked on a permission prompt in a
# tab nobody is looking at, and the caller has to get control back to say so.
DEFAULT_WAIT_TIMEOUT = 60.0
MAX_WAIT_TIMEOUT = 600.0


@dataclass
class ToolContext:
    """Per-agent state a tool needs beyond its own arguments.

    One instance belongs to one ``Agent``, which is what keeps the shell
    overflow files it collects from being readable by every other agent in the
    process (see gate()).
    """

    # Resolved paths of the overflow files this agent's own commands produced.
    shell_outputs: set = field(default_factory=set)
    # The agent's own session log, for search_history/read_history. None where
    # there is no log to search (bare execute_tool calls, tests); the history
    # tools then return a readable error instead of failing.
    session: Optional[Session] = None


@dataclass(frozen=True)
class Tool:
    """One tool: its model-facing schema, executor, and gating class.

    ``run`` takes (args, cwd, mode, ctx) and returns a string or an awaitable
    of one; it is None for tools the agent loop handles itself (write_todos,
    the agent tools). ``access`` drives gate(): "read" runs freely inside cwd,
    "write" is auto-approved inside cwd in edit mode, "execute" needs
    confirmation outside yolo except for commands safe_command() recognizes as
    read-only, "background" is "execute" with no such exception, "none" is
    never gated, "always" needs confirmation even in yolo mode.
    """

    schema: dict
    run: Optional[Callable[[dict, Path, str, ToolContext], object]]
    access: str = "none"


_TODO_MARKERS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def normalize_todos(value: object) -> Optional[list[dict]]:
    """The ``todos`` argument as a task list, or None when it is malformed.

    The model writes this argument, so an unexpected shape has to become a
    tool error rather than an exception: every consumer (the result text, the
    TUI panel, the JSON stream) assumes a list of objects.
    """
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        return None
    return value


def render_todos(todos: list[dict]) -> str:
    """Plain-text rendering of the todo list, used as the tool result the model sees."""
    if not todos:
        return "(todo list cleared)"
    return "\n".join(f"{_TODO_MARKERS.get(t.get('status'), '[ ]')} {t.get('content', '')}" for t in todos)


def _validate_value(value: object, schema: dict, path: str) -> Optional[str]:
    expected = schema.get("type")
    if expected == "string":
        if not isinstance(value, str):
            return f"'{path}' must be a string"
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"'{path}' must be an integer"
    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"'{path}' must be a number"
    elif expected == "boolean":
        if not isinstance(value, bool):
            return f"'{path}' must be a boolean"
    elif expected == "array":
        if not isinstance(value, list):
            return f"'{path}' must be an array"
        items = schema.get("items")
        if isinstance(items, dict):
            for index, element in enumerate(value):
                error = _validate_value(element, items, f"{path}[{index}]")
                if error is not None:
                    return error
    elif expected == "object":
        if not isinstance(value, dict):
            return f"'{path}' must be an object"
        return _validate_object(value, schema, path)
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        choices = ", ".join(repr(choice) for choice in enum)
        return f"'{path}' must be one of {choices}"
    return None


def _validate_object(value: dict, schema: dict, path: str = "") -> Optional[str]:
    for key in schema.get("required") or []:
        if key not in value:
            where = f" in '{path}'" if path else ""
            return f"missing required argument '{key}'{where}"
    properties = schema.get("properties") or {}
    for key, item in value.items():
        subschema = properties.get(key)
        if isinstance(subschema, dict):
            error = _validate_value(item, subschema, f"{path}.{key}" if path else key)
            if error is not None:
                return error
    return None


def validate_args(name: str, args: dict, toolset: dict) -> Optional[str]:
    """Why ``args`` do not fit the tool's declared schema, or None when they do.

    The one validation layer every ToolCallPart passes before gating or
    dispatch, agent-handled tools included, so a malformed argument becomes a
    tool error result instead of an exception ending the turn. Covers exactly
    the JSON-schema subset REGISTRY declares (types, required, enum, array
    items); unknown extra arguments pass — models add stray keys, and
    rejecting them costs more than ignoring them.
    """
    tool = toolset.get(name)
    if tool is None:
        return f"unknown tool {name!r}"
    parameters = (tool.schema.get("function") or {}).get("parameters") or {}
    return _validate_object(args, parameters)


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


# Session-log record rendering, shared by ``paimon log`` (commands.py) and the
# read_history tool. Seq is the record's 1-based physical line number.

LOG_DETAIL_WIDTH = 120


def _chars(text: str) -> str:
    n = len(text)
    return f"{n / 1000:.1f}k chars" if n >= 1000 else f"{n} chars"


def _clip(text: object, full: bool) -> str:
    if full:
        return str(text)
    line = " ".join(str(text).split())
    return line if len(line) <= LOG_DETAIL_WIDTH else line[: LOG_DETAIL_WIDTH - 1] + "…"


def _render_message(seq: int, record: dict, full: bool) -> list[str]:
    lines = []
    for part in (record.get("message") or {}).get("parts") or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("part_kind")
        content = part.get("content")
        if kind == "user-prompt":
            label, detail = "user", _clip(content, full)
        elif kind == "text":
            label, detail = "assistant", _clip(content, full)
        elif kind == "thinking":
            text = str(content or "")
            label, detail = "thinking", text if full else f"({_chars(text)})"
        elif kind == "tool-call":
            args = part.get("args")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"args": args}
            name = str(part.get("tool_name") or "?")
            detail = summarize_call(name, args if isinstance(args, dict) else {},
                                    limit=None if full else LOG_DETAIL_WIDTH)
            label = f"tool_call {name}"
        elif kind == "tool-return":
            text = content if isinstance(content, str) else json.dumps(
                content, ensure_ascii=False, default=str)
            label = f"tool_result {part.get('tool_name') or '?'}"
            detail = text if full else f"({_chars(text)}) {_clip(text, False)}"
        elif kind == "retry-prompt":
            label, detail = "retry_prompt", _clip(content, full)
        else:
            label, detail = str(kind or "?"), ""
        lines.append(f"[{seq}] {label}  {detail}".rstrip())
    replaces = record.get("replaces")
    if lines and isinstance(replaces, str):
        lines[0] += f" (replaces {replaces[:8]})"
    return lines


def superseded_seqs(entries: list) -> dict[int, int]:
    """Seq of every message revision a later record replaced → the final seq.

    The log is append-only, so a tool result exists first as the pre-seeded
    "Interrupted by user." placeholder and then as one replacement per
    completed tool. Only the last revision reflects what actually happened;
    anything rendering the log as events must skip the seqs mapped here or it
    shows interruptions and intermediate snapshots that never occurred.
    """
    id_seq: dict[str, int] = {}
    replacements: dict[str, list[int]] = {}
    for seq, record in entries:
        if not isinstance(record, dict) or record.get("type") != "message":
            continue
        if isinstance(record.get("id"), str):
            id_seq[record["id"]] = seq
        replaced = record.get("replaces")
        if isinstance(replaced, str):
            replacements.setdefault(replaced, []).append(seq)
    superseded: dict[int, int] = {}
    for replaced_id, seqs in replacements.items():
        final = seqs[-1]
        for seq in (id_seq.get(replaced_id), *seqs[:-1]):
            if seq is not None:
                superseded[seq] = final
    return superseded


def render_record(seq: int, record: Optional[dict], full: bool) -> list[str]:
    if record is None:
        return [f"[{seq}] <corrupt>"]
    kind = record.get("type")
    if kind == "message":
        return _render_message(seq, record, full)
    if kind == "session":
        return [f"[{seq}] session {str(record.get('id') or '')[:8]} "
                f"created {record.get('created_at') or '?'}"]
    if kind == "system_prompt":
        content = str(record.get("content") or "")
        return [f"[{seq}] system_prompt {content if full else f'({_chars(content)})'}"]
    if kind == "compaction":
        kept = record.get("kept_messages")
        return [f"[{seq}] compacted: {record.get('tokens_before') or 0:,} tokens "
                f"→ summary + {len(kept) if isinstance(kept, list) else 0} kept"]
    if kind == "turn_end":
        line = f"[{seq}] turn_end {record.get('outcome') or '?'}"
        error = record.get("error")
        if isinstance(error, str) and error:
            line += f"  {_clip(error, full)}"
        lines = [line]
        partial = record.get("partial_text")
        if isinstance(partial, str) and partial:
            lines.append(f"[{seq}] partial  "
                         + (partial if full else f"({_chars(partial)}) {_clip(partial, False)}"))
        return lines
    if kind == "model_retry":
        return [f"[{seq}] model_retry attempt {record.get('attempt') or '?'}  "
                f"{_clip(record.get('error') or '', full)}".rstrip()]
    if kind == "compaction_failed":
        return [f"[{seq}] compaction_failed  {_clip(record.get('error') or '', full)}".rstrip()]
    return [f"[{seq}] {kind or '?'}"]


def _resolve(path: str, cwd: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else cwd / p


def resolve_path(path: str, cwd: Path) -> Path:
    """The path a tool would actually touch, resolved against the agent's cwd.

    Public so UI previews share the exact resolution the execution uses; two
    implementations would eventually preview one file and modify another.
    """
    return _resolve(path, cwd)


def _real(path: Path) -> Optional[Path]:
    """The path with symlinks resolved, or None when it cannot be resolved.

    None never compares equal to a recorded path, so an unresolvable path
    falls through to confirmation instead of raising inside the gate.
    """
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _inside(path: Path, cwd: Path) -> bool:
    """True if path (symlinks resolved) is inside cwd.

    A path that cannot be resolved at all (a symlink loop raises RuntimeError,
    not OSError) counts as outside: gate() then asks for confirmation instead
    of letting the exception escape the tool-call error boundary and end the
    turn.
    """
    try:
        return path.resolve().is_relative_to(cwd.resolve())
    except (OSError, RuntimeError, ValueError):
        return False


# Any of these anywhere in the raw command string — even inside quotes, since
# "$" and backticks expand inside double quotes and the rest are rare enough
# quoted that fail-closed simplicity wins — means the segment view built below
# may not be what the shell actually runs (substitution, redirection,
# escaping, brace expansion, glob character classes that can match ".."), so
# the command is never auto-allowed. A false rejection only costs one
# confirmation prompt, so this errs hard toward rejection. ";", "&" and "|"
# are not here: _split_segments handles them position-aware, so quoted forms
# like grep "a|b" stay allowed.
_SHELL_METACHARS = frozenset("$`<>()\\{[\n\r")


def _split_segments(command: str) -> Optional[list[tuple[str, str]]]:
    """Split a command at unquoted &&, ||, ";" and "|" into (operator,
    segment) pairs; the first pair's operator is "".

    Returns None for anything not worth modeling: a background "&", an empty
    segment (doubled or dangling operator), an unterminated quote. Tracking
    quotes with two booleans is sound only because backslash, "$" and
    backticks were hard-rejected before this runs, so nothing sh recognizes
    can move or escape a closing quote.
    """
    parts: list[tuple[str, str]] = []
    buf: list[str] = []
    op = ""
    in_sq = in_dq = False
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if in_sq or in_dq:
            if ch == ("'" if in_sq else '"'):
                in_sq = in_dq = False
            buf.append(ch)
        elif ch in "'\"":
            in_sq, in_dq = ch == "'", ch == '"'
            buf.append(ch)
        elif ch in ";&|":
            two = command[i:i + 2]
            if ch == "&" and two != "&&":
                return None  # background job
            seg = "".join(buf).strip()
            if not seg:
                return None  # leading operator, or a ";;"-style run
            parts.append((op, seg))
            buf = []
            op = two if two in ("&&", "||") else ch
            i += len(op) - 1
        else:
            buf.append(ch)
        i += 1
    if in_sq or in_dq:
        return None
    seg = "".join(buf).strip()
    if not seg:
        return None  # trailing operator, or an all-whitespace command
    parts.append((op, seg))
    return parts


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


def _safe_cd(argv: list[str], base: Path, root: Path) -> Optional[Path]:
    """cd is modeled only in its plainest form: one directory argument that
    stays inside root. Bare cd goes to $HOME and "cd -" to $OLDPWD, both
    outside our view, and any flag changes semantics we do not track."""
    if len(argv) != 2 or not argv[1] or argv[1].startswith("-"):
        return None
    # With CDPATH set, sh may resolve a relative target against a CDPATH
    # entry instead of the current directory; do not guess which.
    if os.environ.get("CDPATH"):
        return None
    tok = argv[1]
    # sh's cd is logical ("link/.." strips the text, not the symlink) while
    # _inside resolves symlinks first; a ".." segment is exactly where the
    # two disagree, so ban it rather than model both semantics.
    #
    # A glob target ("sub*") resolves as its literal self, so the check
    # below sees "cwd/sub*" (inside) while the shell cd's into whatever the
    # glob matches — a symlink pointing outside root would escape, and every
    # later segment would then resolve against that outside base. cd to a
    # glob is pointless anyway, so reject any "*"/"?" outright.
    if ".." in tok.split("/") or "*" in tok or "?" in tok:
        return None
    target = _resolve(tok, base)
    # Resolved so later segments check against the directory the kernel will
    # actually be in (without "..", physical and logical agree).
    return target.resolve() if _inside(target, root) else None


def _safe_simple(argv: list[str], base: Path, root: Path) -> Optional[Path]:
    """Check one operator-free command from a compound line.

    ``base`` resolves relative paths (it moves when an earlier segment was
    cd); ``root`` is the original cwd and stays the containment boundary.
    Returns the base for the next segment — a new directory for cd, ``base``
    unchanged otherwise — or None when not clearly read-only.
    """
    # Tilde expands at word start and after "=" in bash; reject both
    # positions rather than guess what /bin/sh does. Mid-token tildes
    # (git's HEAD~1) stay allowed.
    if any(tok.startswith("~") or "=~" in tok for tok in argv):
        return None
    if argv[0] == "cd":
        return _safe_cd(argv, base, root)
    if argv[0] == "git":
        return base if _safe_git(argv[1:]) else None
    spec = _SAFE_COMMANDS.get(argv[0])  # exact match: "./ls", "/bin/ls", "FOO=1 ls" miss
    if spec is None:
        return None
    for tok in argv[1:]:
        if tok.startswith("-"):
            # A flag carrying an attached path (-f/etc/passwd, --from-file=/x)
            # would never reach the containment check below.
            if _flag_denied(tok, spec.deny_flags) or "/" in tok:
                return None
        elif _hidden_glob(tok):
            return None
        elif spec.check_paths and not _inside(_resolve(tok, base), root):
            return None
    return base


def safe_command(command: str, cwd: Path) -> bool:
    """Conservatively decide whether a shell command is clearly read-only.

    A guardrail against agent mistakes, not a security boundary: anything not
    positively recognized returns False and the caller falls back to the
    normal confirmation flow, so a miss costs a prompt, never a denial.

    Compound commands pass when every &&/||/;/|-separated segment passes on
    its own. A cd segment moves the base later relative paths resolve
    against, but containment is always checked against the original cwd.
    """
    if any(ch in _SHELL_METACHARS for ch in command):
        return False
    parts = _split_segments(command)
    if parts is None:
        return False
    # cd may only chain with "&&": then the first failing segment stops the
    # whole line, so every segment that does run runs in exactly the base
    # modeled below. With ";" or "||" a failed or skipped cd leaves later
    # segments in a different directory than modeled, and in a pipeline
    # every segment runs in its own subshell, so cd would not apply at all.
    pure_and = all(op == "&&" for op, _ in parts[1:])
    base = cwd
    for _, seg in parts:
        try:
            argv = shlex.split(seg)
        except ValueError:
            return False
        if not argv or (argv[0] == "cd" and not pure_and):
            return False
        new_base = _safe_simple(argv, base, cwd)
        if new_base is None:
            return False
        base = new_base
    return True


def gate(name: str, args: dict, mode: str, cwd: Path,
         registry: Optional[dict[str, Tool]] = None,
         safe_commands: bool = True,
         ctx: Optional[ToolContext] = None) -> str:
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
    if tool.access == "background":
        # No safe_command exception here. That list is about what a command
        # reads and writes, and a process that never ends on its own is not
        # harmless just because it only reads: "tail -f" is on the deny side of
        # it precisely because it runs until the timeout — and a background
        # command has no timeout to run into.
        return "confirm"
    # A missing/malformed path resolves to cwd itself; the tool then fails on its own.
    resolved = _resolve(str(args.get("path") or ""), cwd)
    inside = _inside(resolved, cwd)
    if tool.access == "read":
        # An overflow file this agent's own command produced is paimon's record
        # of something the user already approved, so reading it back is not a
        # new escalation. The exemption is per agent on purpose: the directory
        # itself is shared by every session and project on the machine, and a
        # blanket allowance would let one agent read another one's command
        # output — a different project's, or last week's — without a prompt.
        own_output = ctx is not None and _real(resolved) in ctx.shell_outputs
        return "allow" if inside or own_output else "confirm"
    if tool.access == "write" and mode == "edit" and inside:
        return "allow"
    return "confirm"


async def run_tool(name: str, args: dict, cwd: Path, mode: str,
                   confirm: Optional[ConfirmFn] = None,
                   registry: Optional[dict[str, Tool]] = None,
                   safe_commands: bool = True,
                   ctx: Optional[ToolContext] = None) -> tuple[str, bool]:
    """Gate, optionally confirm, then execute a tool call.

    Returns ``(result, denied)``. This is the enforcement point: a call that
    needs confirmation is denied when no confirm hook is available, so a
    headless Agent cannot bypass the permission mode. ``registry`` narrows the
    available tools (an agent's own set); None means the full REGISTRY.
    ``ctx`` carries the calling agent's own state, so gating decisions that
    depend on what this agent did earlier stay scoped to it.
    """
    if gate(name, args, mode, cwd, registry, safe_commands=safe_commands, ctx=ctx) == "confirm":
        allowed = await confirm(name, args) if confirm else False
        if not allowed:
            return "User denied this operation.", True
    return await execute_tool(name, args, cwd, mode=mode, registry=registry, ctx=ctx), False


def _decode_preserving(raw: bytes) -> tuple[str, str, bool]:
    """(LF-normalized text, line ending to restore, had a UTF-8 BOM).

    The model reads and writes LF, so matching and editing happen on the LF
    copy; the original ending and BOM are put back on write, and a CRLF file
    stays CRLF with only the target span changed. Reading and writing bytes
    keeps this independent of the platform and its locale — text mode would
    normalize every line ending in the file just for touching one line.
    Mixed-ending files are re-written uniformly with the majority ending.
    """
    bom = raw.startswith(codecs.BOM_UTF8)
    if bom:
        raw = raw[len(codecs.BOM_UTF8):]
    text = raw.decode("utf-8", errors="replace")
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
        return f"Error: file not found: {path}"
    text, _, _ = _decode_preserving(path.read_bytes())
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
        # edit does; new files are plain LF UTF-8.
        try:
            _, newline, bom = _decode_preserving(path.read_bytes())
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_encode_preserving(content, newline, bom))
    n = content.count("\n") + 1
    return f"Wrote {n} lines to {path}"


def _edit_file(args: dict, cwd: Path) -> str:
    path = _resolve(args["path"], cwd)
    if not path.exists():
        return f"Error: file not found: {path}"
    text, newline, bom = _decode_preserving(path.read_bytes())
    old = str(args["old_string"]).replace("\r\n", "\n")
    new = str(args["new_string"]).replace("\r\n", "\n")
    count = text.count(old)
    if count == 0:
        return "Error: old_string not found in file."
    if count > 1:
        return f"Error: old_string is not unique (found {count} times). Add more context to make it unique."
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


_GREP_DEFAULT_RESULTS = 50
_GREP_MAX_RESULTS = 200
_GREP_MAX_LINE = 300  # chars of a matched line shown before it is clipped
# Files larger than this are skipped: a multi-hundred-megabyte artifact would
# stall the loop for a search that is about source code.
_GREP_MAX_FILE_BYTES = 10_000_000


def _grep_files(base: Path, name_filter: Optional[str], sandboxed: bool):
    """The files under ``base`` a grep looks at, in stable path order."""
    skip = _GLOB_IGNORE
    for path in sorted(base.rglob("*")):
        if not path.is_file() or skip.intersection(path.relative_to(base).parts):
            continue
        if name_filter and not fnmatch.fnmatch(path.name, name_filter):
            continue
        # sandboxed keeps symlinks under base from reaching files outside it,
        # the same way glob's listing does.
        if sandboxed and not _inside(path, base):
            continue
        yield path


def _grep(args: dict, cwd: Path, sandboxed: bool = False) -> str:
    try:
        pattern = re.compile(str(args["pattern"]))
    except re.error as exc:
        return f"Error: invalid regular expression: {exc}"
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
        return f"Error: no such file or directory: {base}"

    lines: list[str] = []
    skipped_large = 0
    truncated = False
    for path in files:
        try:
            raw = path.read_bytes() if path.stat().st_size <= _GREP_MAX_FILE_BYTES else None
        except OSError:
            continue
        if raw is None:
            skipped_large += 1
            continue
        if b"\0" in raw[:8192]:
            continue  # binary
        text, _, _ = _decode_preserving(raw)
        for lineno, line in enumerate(text.splitlines(), 1):
            if not pattern.search(line):
                continue
            shown = line.strip()
            if len(shown) > _GREP_MAX_LINE:
                shown = shown[:_GREP_MAX_LINE] + "…"
            lines.append(f"{path}:{lineno}:{shown}")
            if len(lines) >= limit:
                truncated = True
                break
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


# The session log is append-only, so everything compaction dropped from the
# model's context is still on disk. search_history greps that log and
# read_history returns full records by seq — the same 1-based line numbers
# ``paimon log`` prints.

_HISTORY_TOOLS = ("search_history", "read_history")
_SEARCH_CONTEXT = 100  # chars of context shown on each side of a match
_DEFAULT_SEARCH_RESULTS = 20
_MAX_SEARCH_RESULTS = 100
_MAX_READ_RECORDS = 20
# Self-truncation headroom: read_history cuts at record boundaries with a note
# naming the seq it stopped at, which execute_tool's generic chop cannot do.
_READ_BUDGET = MAX_OUTPUT - 1_000


def _surviving_entries(session: Session) -> list[tuple[int, dict]]:
    """Message and compaction records with ``replaces`` applied, seq attached.

    Unlike Session.messages(), a compaction record does not reset the list:
    recovering what compaction dropped is the whole point here. A replaced
    record keeps its original position but carries the replacement's seq, so
    read_history on that seq returns exactly what search matched.
    """
    survivors: list[tuple[int, dict]] = []
    positions: dict[str, int] = {}
    for seq, record in session.entries():
        if record is None:
            continue
        kind = record.get("type")
        if kind == "compaction":
            survivors.append((seq, record))
        elif kind == "message" and isinstance(record.get("message"), dict):
            replaced = record.get("replaces")
            if isinstance(replaced, str) and replaced in positions:
                survivors[positions[replaced]] = (seq, record)
            else:
                if isinstance(record.get("id"), str):
                    positions[record["id"]] = len(survivors)
                survivors.append((seq, record))
    return survivors


def _searchable_parts(record: dict):
    """(label, text) for each part of a message record worth searching.

    The history tools' own calls and results are skipped: they quote earlier
    matches, so searching them would return every past search over again.
    """
    for part in (record.get("message") or {}).get("parts") or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("part_kind")
        content = part.get("content")
        if kind == "user-prompt":
            yield "user", str(content or "")
        elif kind == "text":
            yield "assistant", str(content or "")
        elif kind == "thinking":
            yield "thinking", str(content or "")
        elif kind in ("tool-call", "tool-return"):
            name = str(part.get("tool_name") or "?")
            if name in _HISTORY_TOOLS:
                continue
            if kind == "tool-call":
                args = part.get("args")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False, default=str)
                yield f"tool_call {name}", f"{name} {args}"
            else:
                text = content if isinstance(content, str) else json.dumps(
                    content, ensure_ascii=False, default=str)
                yield f"tool_result {name}", text


def _snippet(text: str, match: re.Match) -> str:
    start = max(0, match.start() - _SEARCH_CONTEXT)
    end = min(len(text), match.end() + _SEARCH_CONTEXT)
    piece = " ".join(text[start:end].split())
    return f"{'…' if start > 0 else ''}{piece}{'…' if end < len(text) else ''}"


def _search_history(args: dict, ctx: ToolContext) -> str:
    if ctx.session is None:
        return "Error: no session log available"
    query = str(args.get("query") or "")
    if not query:
        return "Error: query must not be empty"
    try:
        limit = max(1, min(int(args.get("max_results") or _DEFAULT_SEARCH_RESULTS), _MAX_SEARCH_RESULTS))
    except (TypeError, ValueError):
        return "Error: max_results must be an integer"
    note = ""
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        note = " (query is not a valid regex; searched literally)"
    hits = []
    for seq, record in _surviving_entries(ctx.session):
        if record.get("type") == "compaction":
            parts = [("compaction summary", str(record.get("summary") or ""))]
        else:
            parts = _searchable_parts(record)
        for label, text in parts:
            match = pattern.search(text)
            if match:
                hits.append(f"[{seq}] {label}: {_snippet(text, match)}")
    if not hits:
        return f"No matches in the session log{note}."
    shown = hits[:limit]
    header = (f"{len(hits)} matching parts{note}"
              + (f", showing the first {len(shown)}" if len(hits) > len(shown) else "")
              + ". Use read_history with a seq to see a full record.")
    return "\n".join([header, *shown])


def _read_history(args: dict, ctx: ToolContext) -> str:
    if ctx.session is None:
        return "Error: no session log available"
    try:
        seq = int(args["seq"])
        count = max(1, min(int(args.get("count") or 1), _MAX_READ_RECORDS))
    except (KeyError, TypeError, ValueError):
        return "Error: seq (and optional count) must be integers"
    entries = ctx.session.entries()
    if seq < 1 or seq > len(entries):
        return f"Error: seq out of range (the log has {len(entries)} records)"
    superseded = superseded_seqs(entries)
    lines: list[str] = []
    used = 0
    for number, record in entries[seq - 1 : seq - 1 + count]:
        if number in superseded:
            # The content at this seq was replaced later (a placeholder or an
            # intermediate snapshot); returning it would present a result that
            # never happened as real.
            lines.append(f"[{number}] (superseded revision; the final version "
                         f"is at seq {superseded[number]})")
            continue
        rendered = "\n".join(render_record(number, record, full=True))
        if used + len(rendered) > _READ_BUDGET:
            if not lines:
                rendered = rendered[:_READ_BUDGET] + "\n... (record truncated)"
                lines.append(rendered)
            else:
                lines.append(f"... (stopped before seq {number}: output budget reached, "
                             "read fewer records at once)")
            break
        lines.append(rendered)
        used += len(rendered)
    return "\n".join(lines)


_COMMAND_TIMEOUT = 120.0
_KILL_GRACE = 2.0  # seconds to wait after SIGTERM before forcing SIGKILL
_KILL_TIMEOUT = 2.0  # maximum wait to reap the process after SIGKILL
_READ_CHUNK = 64 * 1024
# Once the command itself has exited, a descendant may still be writing to the
# pipe it inherited. Keep reading while bytes arrive, but never for long.
_DRAIN_GRACE = 0.1
_DRAIN_TOTAL = 2.0

_SHELL_MAX_LINES = 2_000
# The footer and the status line share the result budget with the output, so
# the byte cap is derived rather than written down. That keeps one invariant:
# a shell result never exceeds MAX_OUTPUT characters, so execute_tool's generic
# truncation — which cuts from the head, and would take back exactly the tail
# this collector went to the trouble of keeping — never fires for shell.
_SHELL_FOOTER_BUDGET = 2_000
_SHELL_MAX_BYTES = MAX_OUTPUT - _SHELL_FOOTER_BUDGET
_OVERFLOW_TTL = 7 * 24 * 60 * 60  # seconds an overflow file is kept around


def shell_output_dir() -> Path:
    """Where a command's full output goes when more of it than fits is produced."""
    return data_dir() / "shell-output"


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def _prune_overflow(directory: Path) -> None:
    """Drop overflow files old enough that no live session can still cite them."""
    cutoff = time.time() - _OVERFLOW_TTL
    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink()
        except OSError:
            pass


class _OutputTail:
    """Bounded collector for one command's streamed output.

    Keeps the last ``_SHELL_MAX_BYTES`` in memory and, once the command writes
    more than fits, mirrors every byte into an overflow file. The tail is what
    the model needs (the error and the exit code live at the end) and the file
    is how it can still get back to the first error of a long build log.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._newlines = 0
        self._open_line = False
        self._line_bytes = 0
        self._fd: Optional[int] = None
        self.path: Optional[Path] = None
        self.total_bytes = 0

    @property
    def total_lines(self) -> int:
        return self._newlines + (1 if self._open_line else 0)

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total_bytes += len(chunk)
        self._newlines += chunk.count(b"\n")
        last_newline = chunk.rfind(b"\n")
        if last_newline == -1:
            self._line_bytes += len(chunk)
        else:
            self._line_bytes = len(chunk) - last_newline - 1
        self._open_line = not chunk.endswith(b"\n")

        if self._fd is None and (self.total_bytes > _SHELL_MAX_BYTES
                                 or self.total_lines > _SHELL_MAX_LINES):
            self._start_overflow()
        self._write(chunk)
        self._buffer += chunk
        # Only bounds memory; the budget itself is applied in render(). Copying
        # on every chunk would make a chatty command quadratic in chunk count.
        if len(self._buffer) > 2 * _SHELL_MAX_BYTES:
            del self._buffer[:len(self._buffer) - _SHELL_MAX_BYTES]

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def render(self) -> str:
        """The retained tail, with a footer whenever anything was dropped."""
        data = bytes(self._buffer)
        trimmed = len(data) < self.total_bytes
        if len(data) > _SHELL_MAX_BYTES:
            data = data[len(data) - _SHELL_MAX_BYTES:]
            trimmed = True
        text = data.decode("utf-8", errors="replace")
        partial_line = False
        if trimmed:
            newline = text.find("\n")
            if newline == -1:
                # A single line longer than the whole budget: its end is all we
                # can keep, and the end is where the message usually is.
                partial_line = True
            else:
                text = text[newline + 1:]  # drop the fragment the cut left behind
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if len(lines) > _SHELL_MAX_LINES:
            lines = lines[-_SHELL_MAX_LINES:]
        text = "\n".join(lines)

        if not trimmed and len(lines) >= self.total_lines:
            return text
        where = f"; full output: {self.path}" if self.path is not None else ""
        shown = _format_size(len(text.encode("utf-8", errors="replace")))
        if partial_line:
            note = (f"showing last {shown} of line {self.total_lines:,} "
                    f"(line is {_format_size(self._line_bytes)})")
        else:
            note = (f"showing last {len(lines):,} of {self.total_lines:,} lines "
                    f"({shown} of {_format_size(self.total_bytes)})")
        return f"{text}\n\n[{note}{where}]"

    def _start_overflow(self) -> None:
        directory = shell_output_dir()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = directory / f"{stamp}-{os.getpid()}-{os.urandom(3).hex()}.log"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            _prune_overflow(directory)
            # The output can hold anything the command printed, so keep it
            # private the way config.py keeps the api key file private.
            self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError:
            return  # losing the head is bad; failing the command over it is worse
        self.path = path
        self._write(bytes(self._buffer))

    def _write(self, chunk: bytes) -> None:
        if self._fd is None or not chunk:
            return
        try:
            view = memoryview(chunk)
            while view:
                # os.write is free to accept only part of a large buffer.
                view = view[os.write(self._fd, view):]
        except OSError:
            # A half-written file must not be advertised as the full output.
            self.close()
            if self.path is not None:
                try:
                    self.path.unlink()
                except OSError:
                    pass
                self.path = None


def _signal_group(pgid: int, sig: int, proc: asyncio.subprocess.Process) -> None:
    """Send a signal to the whole process group, falling back to the child alone.

    Windows has no process groups to signal (os.killpg does not exist there);
    the tree is handled by the taskkill path in _kill_tree, and this falls
    back to killing the child itself.
    """
    if os.name == "nt":
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
        return
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


async def _taskkill_tree(proc: asyncio.subprocess.Process) -> None:
    """Windows tree kill: taskkill /T /F walks the parent-pid tree, the
    closest equivalent of killing a POSIX process group. Best effort — a
    child whose parent already exited is re-parented and cannot be found by
    anything — with a direct kill of the leader as the fallback.
    """
    try:
        killer = await asyncio.create_subprocess_exec(
            "taskkill", "/PID", str(proc.pid), "/T", "/F",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await asyncio.wait_for(killer.wait(), timeout=_KILL_GRACE + _KILL_TIMEOUT)
    except (OSError, asyncio.TimeoutError):
        pass
    if proc.returncode is None:
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=_KILL_TIMEOUT)
        except asyncio.TimeoutError:
            pass


async def _kill_tree(proc: asyncio.subprocess.Process, pgid: int) -> None:
    """Terminate and reap a command's process group without letting cleanup hang.

    The group is signalled even when the shell itself has already exited: the
    processes it backgrounded are precisely the ones that outlive it, and on a
    timeout or an interrupt the whole tree has to go.
    """
    if os.name == "nt":
        await _taskkill_tree(proc)
        return
    _signal_group(pgid, signal.SIGTERM, proc)
    if proc.returncode is None:
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=_KILL_GRACE)
        except asyncio.TimeoutError:
            pass
    # Unconditional: SIGTERM may have been trapped, and a leader that exited
    # says nothing about the rest of its group.
    _signal_group(pgid, signal.SIGKILL, proc)
    if proc.returncode is None:
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=_KILL_TIMEOUT)
        except asyncio.TimeoutError:
            # The OS should eventually reap it; most importantly, cleanup cannot
            # block the agent indefinitely.
            pass


async def _pump(stream: asyncio.StreamReader, tail: _OutputTail) -> None:
    while True:
        chunk = await stream.read(_READ_CHUNK)
        if not chunk:
            return
        tail.append(chunk)


@cache
def shell_executable() -> Optional[str]:
    """The shell the shell tool actually runs commands with.

    Bash is preferred when installed, so the bashisms models habitually write
    ([[ ]], process substitution) behave; /bin/sh is the fallback. None means
    the platform default (cmd.exe via ComSpec on Windows). The system prompt
    reports this same value — the model must never be told a shell the tool
    does not use (previously it reported $SHELL while running /bin/sh).
    """
    if os.name == "nt":
        return None
    for candidate in ("/bin/bash", "/usr/bin/bash"):
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("bash") or "/bin/sh"


async def _spawn_shell(
    command: str, cwd: Path, env: Optional[dict] = None
) -> tuple[asyncio.subprocess.Process, asyncio.StreamReader, asyncio.ReadTransport]:
    """Start ``command`` with its output on a pipe asyncio does not manage.

    With stdout=PIPE, Process.wait() on Python 3.11+ also waits for the pipe
    to reach EOF (gh-88050): a backgrounded descendant that inherited the pipe
    then holds the turn open long after the command itself exited. Handing the
    shell the write end of a plain os.pipe() keeps wait() about the shell
    alone; the read end comes back as a stream plus the transport to close
    when a descendant will not let go.
    """
    read_fd, write_fd = os.pipe()
    try:
        options = dict(
            cwd=str(cwd),
            # Without this a command that reads stdin (a bare "cat") would
            # block until the timeout, eating the user's keystrokes in the TUI.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=write_fd,
            stderr=write_fd,
            env=env,
        )
        if os.name == "nt":
            # A group of its own, so cleanup handles the tree as one unit
            # (taskkill /T in _kill_tree); start_new_session is POSIX-only.
            options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            # start_new_session puts the child in its own process group so the
            # whole tree (the shell plus anything it spawns) can be killed on
            # timeout/interrupt.
            options["start_new_session"] = True
        shell = shell_executable()
        if shell is None:
            proc = await asyncio.create_subprocess_shell(command, **options)
        else:
            # Exec form rather than create_subprocess_shell's executable
            # override: the override would run bash with argv[0] "sh", which
            # flips it into POSIX mode.
            proc = await asyncio.create_subprocess_exec(shell, "-c", command, **options)
    except BaseException:
        os.close(read_fd)
        raise
    finally:
        os.close(write_fd)
    pipe = os.fdopen(read_fd, "rb", 0)
    stream = asyncio.StreamReader()
    try:
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(stream), pipe
        )
    except BaseException:
        pipe.close()
        raise
    return proc, stream, transport


def _stop_reading(transport: asyncio.ReadTransport) -> None:
    """Close the read end of a pipe a descendant is still holding open.

    Without this the event loop keeps draining that pipe into the stream
    buffer for as long as the descendant lives, unbounded and unread.
    """
    try:
        transport.close()
    except OSError:
        pass


async def _drain(reader: asyncio.Task, tail) -> None:
    """Take what the pipe still holds now that the command itself has exited.

    Re-armed while bytes keep arriving, so a descendant in the middle of
    writing is not cut off mid-sentence, and never for long.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _DRAIN_TOTAL
    while not reader.done():
        seen = tail.total_bytes
        try:
            await asyncio.wait_for(asyncio.shield(reader), timeout=_DRAIN_GRACE)
        except asyncio.TimeoutError:
            if tail.total_bytes == seen or loop.time() >= deadline:
                return


async def _stop_reader(reader: asyncio.Task, transport: asyncio.ReadTransport) -> None:
    """End the pump task, closing the pipe if a descendant still holds it open."""
    if not reader.done():
        reader.cancel()
        _stop_reading(transport)
    try:
        await reader
    except (asyncio.CancelledError, Exception):  # noqa: BLE001 — cleanup only
        pass


async def _collect(
    proc: asyncio.subprocess.Process,
    stream: asyncio.StreamReader,
    transport: asyncio.ReadTransport,
    tail: _OutputTail,
) -> None:
    """Read output until the command exits, then drain what is still in flight.

    Deliberately never waits for stdout EOF. A backgrounded descendant inherits
    the pipe and can hold it open for as long as it runs ("svc & echo ready"),
    which would block the turn until the timeout with the command itself long
    finished.
    """
    reader = asyncio.create_task(_pump(stream, tail))
    try:
        await asyncio.wait_for(proc.wait(), timeout=_COMMAND_TIMEOUT)
        await _drain(reader, tail)
    finally:
        await _stop_reader(reader, transport)


async def _shell(args: dict, cwd: Path, ctx: Optional[ToolContext] = None) -> str:
    proc, stream, transport = await _spawn_shell(args["command"], cwd)
    # The child leads its own group, so the group id is its pid. Recorded here
    # because os.getpgid() stops working the moment that leader is reaped.
    pgid = proc.pid
    tail = _OutputTail()
    try:
        try:
            await _collect(proc, stream, transport, tail)
            status = f"(exit code {proc.returncode})"
        except asyncio.TimeoutError:
            await _kill_tree(proc, pgid)
            status = f"(timed out after {_COMMAND_TIMEOUT:g}s)"
        except asyncio.CancelledError:
            await _kill_tree(proc, pgid)
            raise
    finally:
        tail.close()
        # Recorded so this agent — and only this agent — can read the file back
        # without another confirmation; see gate().
        if ctx is not None and tail.path is not None:
            real = _real(tail.path)
            if real is not None:
                ctx.shell_outputs.add(real)
    out = tail.render()
    # The status follows the output instead of replacing it: a timeout or a
    # non-zero exit is exactly when what the command managed to print matters.
    return f"{out}\n{status}" if out.strip() else status


# ---- background commands ---------------------------------------------------
#
# A command that outlives the turn that started it: no timeout, no result, just
# a buffer its own tab renders and its agent reads. Deliberately not a PTY —
# there is no keyboard and no terminal emulation, which is also why output can
# arrive in blocks (see _line_buffered).

# What a background command keeps in memory. Larger than the shell tool's
# budget because this is a live view the user scrolls, while only the tail of
# it is ever handed to the model.
_TASK_BUFFER_BYTES = 256 * 1024


class _TaskOutput:
    """A background command's output, read incrementally by several readers.

    Bounded: the oldest bytes go once the buffer is full. Cursors are global
    byte offsets rather than buffer indexes, so a reader that fell behind the
    trimming is told how much it missed instead of being handed the wrong
    bytes.
    """

    def __init__(self, limit: int = _TASK_BUFFER_BYTES) -> None:
        self._buffer = bytearray()
        self._limit = limit
        self.total_bytes = 0

    @property
    def start(self) -> int:
        """Offset of the oldest byte still retained."""
        return self.total_bytes - len(self._buffer)

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total_bytes += len(chunk)
        self._buffer += chunk
        if len(self._buffer) > self._limit:
            del self._buffer[: len(self._buffer) - self._limit]

    def since(self, offset: int) -> tuple[bytes, int, int]:
        """Output after ``offset``, as (bytes, new cursor, bytes dropped).

        Bytes, not text: a chunk boundary falls wherever the pipe said it did,
        and decoding one that splits a character would put a replacement mark
        in the middle of the output every time. Callers hold the fragment back
        instead.
        """
        offset = max(0, min(offset, self.total_bytes))
        dropped = max(0, self.start - offset)
        return bytes(self._buffer[offset + dropped - self.start:]), self.total_bytes, dropped


def _line_buffered(command: str) -> str:
    """``command`` wrapped so its libc stdio flushes per line, where it can be.

    A background command's stdout is a pipe, so libc buffers it in blocks and a
    tab can sit empty for minutes under a chatty build before everything
    arrives at once. stdbuf works through LD_PRELOAD, which descendants
    inherit; it does nothing for Go or Rust programs, or on musl. This is a
    mitigation, not a fix — output arriving late is what a pipe instead of a
    terminal costs.
    """
    if not shutil.which("stdbuf"):
        return command
    shell = shell_executable() or "sh"
    return f"stdbuf -oL -eL {shlex.quote(shell)} -c {shlex.quote(command)}"


class BackgroundCommand:
    """One command running past the end of the turn that started it."""

    def __init__(
        self,
        command: str,
        proc: asyncio.subprocess.Process,
        stream: asyncio.StreamReader,
        transport: asyncio.ReadTransport,
        pgid: int,
    ) -> None:
        self.command = command
        self.output = _TaskOutput()
        self.exit_code: Optional[int] = None
        self.killed = False
        self._proc = proc
        self._stream = stream
        self._transport = transport
        self._pgid = pgid
        self._reading = asyncio.ensure_future(self._read())

    @property
    def running(self) -> bool:
        return self.exit_code is None

    async def _read(self) -> None:
        reader = asyncio.create_task(_pump(self._stream, self.output))
        try:
            await self._proc.wait()
            await _drain(reader, self.output)
        finally:
            await _stop_reader(reader, self._transport)
            self.exit_code = self._proc.returncode

    async def wait(self) -> Optional[int]:
        """Block until the command has exited and its output has been drained.

        Shielded, because whoever waits is liable to be cancelled — a job being
        killed, the app going down — and cancelling the read would abandon
        whatever the pipe still held.
        """
        await asyncio.shield(self._reading)
        return self.exit_code

    def kill(self) -> None:
        """Stop the whole process group. Returns at once; reaping runs on.

        Callers are all in places that cannot wait — a pane closing, a session
        being swapped out — and the escalation from SIGTERM to SIGKILL takes
        seconds of waiting the UI does not have.
        """
        if self.killed or not self.running:
            return
        self.killed = True
        asyncio.ensure_future(_kill_tree(self._proc, self._pgid))

    def terminate_now(self) -> None:
        """Signal the group without awaiting anything, on the way out.

        The app is exiting: nothing here will be reaped, and a group left
        unsignalled is reparented to init and keeps running. SIGKILL follows
        SIGTERM immediately rather than after a grace period, because there is
        no loop left to come back and finish the job.
        """
        self.killed = True
        if self._proc.returncode is None:
            if os.name == "nt":
                # Nothing left to await taskkill on; fire it detached so the
                # tree still goes down after this process is gone.
                try:
                    subprocess.Popen(["taskkill", "/PID", str(self._proc.pid), "/T", "/F"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except OSError:
                    pass
            _signal_group(self._pgid, signal.SIGTERM, self._proc)
            _signal_group(self._pgid, signal.SIGKILL, self._proc)


async def start_background(command: str, cwd: Path) -> BackgroundCommand:
    """Start ``command`` detached from the turn, in its own process group."""
    proc, stream, transport = await _spawn_shell(
        _line_buffered(command),
        cwd,
        # Python is the one interpreter that can be told to stop block
        # buffering from the outside, and the one this agent starts most.
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return BackgroundCommand(command, proc, stream, transport, proc.pid)


def tail_text(data: bytes, dropped: int = 0, limit: int = _SHELL_MAX_BYTES) -> str:
    """Output for the model: the end of it, with a note about what was cut."""
    trimmed = len(data) - limit
    if trimmed > 0:
        data = data[-limit:]
        dropped += trimmed
    text = data.decode("utf-8", errors="replace")
    if dropped > 0:
        text = f"[{_format_size(dropped)} of earlier output dropped]\n{text}"
    return text


async def execute_tool(name: str, args: dict, cwd: Path, mode: str = "yolo",
                       registry: Optional[dict[str, Tool]] = None,
                       ctx: Optional[ToolContext] = None) -> str:
    """Run a registered tool. Always returns a string for the model."""
    tool = (REGISTRY if registry is None else registry).get(name)
    if tool is None or tool.run is None:
        return f"Error: unknown tool {name!r}"
    ctx = ToolContext() if ctx is None else ctx
    try:
        result = tool.run(args, cwd, mode, ctx)
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
        run=lambda args, cwd, mode, ctx: _read_file(args, cwd),
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
        run=lambda args, cwd, mode, ctx: _write_file(args, cwd),
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
        run=lambda args, cwd, mode, ctx: _edit_file(args, cwd),
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
        run=lambda args, cwd, mode, ctx: _glob(args, cwd, sandboxed=mode != "yolo"),
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
    "grep": Tool(
        access="read",
        run=lambda args, cwd, mode, ctx: _grep(args, cwd, sandboxed=mode != "yolo"),
        schema={
            "type": "function",
            "function": {
                "name": "grep",
                "description": (
                    "Search file contents for a regular expression and return matching "
                    "lines as path:line:text, in file order. Directories are searched "
                    "recursively, skipping VCS/dependency/build noise and binary files. "
                    "Prefer this over shell grep: it needs no confirmation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Python regular expression, matched against each line."},
                        "path": {"type": "string", "description": "File or directory to search (optional, defaults to the working directory)."},
                        "glob": {"type": "string", "description": "Only search files whose name matches this pattern, e.g. '*.py' (optional)."},
                        "max_results": {"type": "integer", "description": f"Maximum matching lines to return (optional, default {_GREP_DEFAULT_RESULTS}, maximum {_GREP_MAX_RESULTS})."},
                    },
                    "required": ["pattern"],
                },
            },
        },
    ),
    "shell": Tool(
        access="execute",
        run=lambda args, cwd, mode, ctx: _shell(args, cwd, ctx),
        schema={
            "type": "function",
            "function": {
                "name": "shell",
                "description": (
                    "Run a shell command in the working directory and return its combined "
                    "stdout/stderr. Use this for listing, searching (grep/find/ls), git, running "
                    "tests, etc. Output is truncated to the last 2000 lines or ~28KB, whichever "
                    "comes first; when that happens the full output is written to a file and its "
                    "path is included in the result, so you can read the earlier part back."
                ),
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
    # "read" because these only read the agent's own session log, never a user
    # file: with no path argument the gate resolves to cwd and always allows,
    # which is right — everything in the log already went through this
    # conversation once.
    "search_history": Tool(
        access="read",
        run=lambda args, cwd, mode, ctx: _search_history(args, ctx),
        schema={
            "type": "function",
            "function": {
                "name": "search_history",
                "description": (
                    "Search this session's full history log on disk. The log keeps every "
                    "user message, assistant reply, tool call and tool result of this "
                    "session, including everything that context compaction has since "
                    "summarized away — use it to recover exact details (file paths, "
                    "command output, earlier decisions) instead of guessing from a "
                    "compaction summary. Returns one line per matching part, prefixed "
                    "with a seq number to pass to read_history for the full record."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Regular expression, matched case-insensitively. An invalid regex is searched as literal text.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum matches to return (optional, default 20, max 100).",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ),
    "read_history": Tool(
        access="read",
        run=lambda args, cwd, mode, ctx: _read_history(args, ctx),
        schema={
            "type": "function",
            "function": {
                "name": "read_history",
                "description": (
                    "Read full records from this session's history log by seq number, "
                    "as returned by search_history. Use it to see the complete content "
                    "behind a match: the whole user message, tool output or assistant "
                    "reply, even when it predates a context compaction."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seq": {
                            "type": "integer",
                            "description": "Seq of the first record to read (from search_history results).",
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of consecutive records to read (optional, default 1, max 20).",
                        },
                    },
                    "required": ["seq"],
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
    # The rest of the registry is stateful in the same way: every one of them
    # acts on the pool of jobs this process is running, which only the UI owns,
    # so the agent loop hands them to its supervisor instead of running them
    # here.
    #
    # access is written out rather than left to default so the choice is on the
    # record: all of them but run_background are "none" because they touch
    # neither the filesystem nor a process of their own, and everything a
    # started agent goes on to do is gated in that agent's own tab under the
    # mode it inherited. Adding a tool here that does reach outside the process
    # needs its own access class, the way run_background has one.
    "spawn_agent": Tool(
        run=None,
        access="none",
        schema={
            "type": "function",
            "function": {
                "name": "spawn_agent",
                "description": (
                    "Start another agent working in parallel in its own tab, in the same "
                    "working directory and (unless an agent type narrows them) with the same "
                    "tools, and return its agent id. Use it for work that is independent of "
                    "what you are doing right now — exploring a second part of the codebase, "
                    "a long test run, a self-contained refactor — and keep the number of "
                    "them small. The prompt must be self-contained: state the goal, the key "
                    "file paths, the decisions already made and what to report back, because "
                    "the new agent has no memory of this conversation and cannot ask you "
                    "anything. Do not wait for it: when it finishes while you are idle, a "
                    "status line wakes you — collect its output with read_job then, and read "
                    "every finished job in that one turn; reading a finished agent also "
                    "closes it for you. wait_for_job exists for the rare case you cannot "
                    "proceed without the result, and stop_job for an agent going the wrong "
                    "way. It cannot spawn agents of its own. Permission "
                    "prompts for its tools appear in its tab, so it can sit blocked until "
                    "the user answers them. Stopping an agent frees its slot but keeps its "
                    "session on disk; pass that session id here later to pick the "
                    "conversation back up where it ended."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The new agent's first user message; must be self-contained.",
                        },
                        "model": {
                            "type": "string",
                            "description": "Model for this agent only (optional; defaults to the current one).",
                        },
                        "session": {
                            "type": "string",
                            "description": (
                                "Session id of an agent you started earlier (quoted when it "
                                "was started and when it was stopped): resume that "
                                "conversation with its history and role intact, the prompt "
                                "becoming its next turn. Only your own agents' sessions "
                                "qualify. Optional; not combinable with 'agent'."
                            ),
                        },
                    },
                    "required": ["prompt"],
                },
            },
        },
    ),
    "send_to_agent": Tool(
        run=None,
        access="none",
        schema={
            "type": "function",
            "function": {
                "name": "send_to_agent",
                "description": (
                    "Send a follow-up instruction to an agent you started. It runs as that "
                    "agent's next turn; if it is busy the message is queued and delivered "
                    "when the current turn ends. Like the spawn prompt it has to stand on "
                    "its own \u2014 the agent cannot see this conversation. Only agents take "
                    "instructions; a background command cannot be sent anything."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "The id of an agent you started.",
                        },
                        "prompt": {"type": "string"},
                    },
                    "required": ["job_id", "prompt"],
                },
            },
        },
    ),
    # Starting a background command is the one supervised tool that reaches
    # outside the process, so it is the one with an access class of its own:
    # "background" rather than "execute" so the safe_command allowance can
    # never apply to it, and never left to default to "none", which would hand
    # the model an unconfirmed, untimed, turn-outliving way to run anything.
    "run_background": Tool(
        run=None,
        access="background",
        schema={
            "type": "function",
            "function": {
                "name": "run_background",
                "description": (
                    "Start a long-running command in its own tab and return a job id, "
                    "instead of waiting for it like the shell tool does. Use it for things "
                    "that are meant to keep running \u2014 a dev server, a file watcher, a long "
                    "build or test suite \u2014 and use shell for anything that finishes on its "
                    "own within a couple of minutes. The command gets no terminal and no "
                    "input, so it must be non-interactive; output may arrive in blocks "
                    "rather than line by line, because a pipe is not a terminal. Nothing "
                    "reaches you on its own: call read_job for new output, wait_for_job to "
                    "wait for it to finish, and stop_job when you are done with it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "description": {
                            "type": "string",
                            "description": "A few words naming the job; it labels the tab.",
                        },
                    },
                    "required": ["command", "description"],
                },
            },
        },
    ),
    # The three below work on either kind of job. They are one tool each rather
    # than one per kind because the ids share a space (see Supervisor._new_id):
    # the model holds a mixed list of them and a per-kind tool would only give
    # it a way to guess wrong. None is gated \u2014 each one reaches only the jobs
    # this same agent started.
    "read_job": Tool(
        run=None,
        access="none",
        schema={
            "type": "function",
            "function": {
                "name": "read_job",
                "description": (
                    "Read what a job you started has produced since you last read it (mode "
                    "'new', the default) or everything it has produced (mode 'all'), "
                    "together with its state. For an agent that is its assistant text only "
                    "\u2014 not its thinking and not its tool output \u2014 so ask it in the spawn "
                    "prompt to end with the summary you need. Reading a finished agent's "
                    "output also closes it and frees its tab; the result quotes the session "
                    "id that spawn_agent can resume should you need it again. For a "
                    "background command it is the tail of its output, plus the exit code "
                    "once it has stopped."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "The id of an agent or a background command you started.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["new", "all"],
                            "description": "'new' (default) since your last read, or 'all'.",
                        },
                    },
                    "required": ["job_id"],
                },
            },
        },
    ),
    "wait_for_job": Tool(
        run=None,
        access="none",
        schema={
            "type": "function",
            "function": {
                "name": "wait_for_job",
                "description": (
                    "Wait until a job you started is no longer running, then return its "
                    "state. This never blocks forever: when the timeout runs out it returns "
                    "'running', and it returns early with 'needs_confirm' when an agent is "
                    "stuck on a permission prompt in its own tab \u2014 say so, because only the "
                    "user can clear that. Read what it produced with read_job afterwards."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "The id of an agent or a background command you started.",
                        },
                        "timeout": {
                            "type": "number",
                            "description": (
                                f"Seconds to wait (optional, default {DEFAULT_WAIT_TIMEOUT:g}, "
                                f"maximum {MAX_WAIT_TIMEOUT:g})."
                            ),
                        },
                    },
                    "required": ["job_id"],
                },
            },
        },
    ),
    "stop_job": Tool(
        run=None,
        access="none",
        schema={
            "type": "function",
            "function": {
                "name": "stop_job",
                "description": (
                    "Stop a job you started, and everything it started. What it produced "
                    "stays readable afterwards. Stop what you no longer need: an agent going "
                    "the wrong way keeps spending tokens, and a background command keeps "
                    "running and keeps a tab open until the app exits."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "The id of an agent or a background command you started.",
                        },
                    },
                    "required": ["job_id"],
                },
            },
        },
    ),
}

# Tools that only mean anything with the UI supervising a pool of panes: the
# agent loop refuses them without a supervisor, and headless leaves them out of
# its toolset entirely rather than offering the model something that cannot
# work. A background task is in the list for a second reason: headless runs one
# turn under asyncio.run, which cancels everything still alive on the way out,
# so a task started there would be a process group nobody ever kills.
SUPERVISED_TOOLS = ("spawn_agent", "send_to_agent", "run_background",
                    "read_job", "wait_for_job", "stop_job")

# What a spawned agent must not be given: every job tool, and the handoff.
#
# Every job tool, because depth stays 1 — only the conversation the user is
# actually in starts agents and leaves processes running behind it — and a
# subagent that can start nothing has nothing to read, wait for or stop either.
# The ownership check would answer "unknown" to all three anyway; leaving them
# in the schema would only spend tokens describing tools that cannot apply.
#
# The handoff, because its access is "always" and the confirmation appears in
# the subagent's own tab, where the user is likely to approve it: approving it
# swaps the session out from under the id the parent holds, and the parent is
# never told.
SUBAGENT_DENIED = ("start_new_session", *SUPERVISED_TOOLS)


def without(registry: dict[str, Tool], names) -> dict[str, Tool]:
    """A copy of ``registry`` with ``names`` removed.

    Narrowing the registry — rather than the agent-handled table, which is
    class-level and shared — is what actually disables a tool for one agent:
    the loop rejects any name outside its own toolset before dispatching.
    """
    return {name: tool for name, tool in registry.items() if name not in set(names)}


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
