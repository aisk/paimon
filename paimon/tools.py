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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

from pydantic_ai.tools import ToolDefinition

from .session import data_dir

# A confirm callback returns True to allow a dangerous tool, False to deny.
ConfirmFn = Callable[[str, dict], Awaitable[bool]]

# Permission modes: read (confirm writes, shell and reads outside cwd),
# edit (auto-approve writes inside cwd), yolo (no confirmation at all).
# In read and edit modes, shell commands recognized by safe_command() run
# without confirmation unless the safe_commands toggle is off.
MODES = ("read", "edit", "yolo")

MAX_OUTPUT = 30_000  # truncate tool output sent back to the model

# Bounds for wait_for_agent. A wait that cannot expire is a deadlock waiting to
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


@dataclass(frozen=True)
class Tool:
    """One tool: its model-facing schema, executor, and gating class.

    ``run`` takes (args, cwd, mode, ctx) and returns a string or an awaitable
    of one; it is None for tools the agent loop handles itself (write_todos,
    the agent tools). ``access`` drives gate(): "read" runs freely inside cwd,
    "write" is auto-approved inside cwd in edit mode, "execute" needs
    confirmation outside yolo except for commands safe_command() recognizes as
    read-only, "none" is never gated, "always" needs confirmation even in yolo
    mode.
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
    """Send a signal to the whole process group, falling back to the child alone."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


async def _kill_tree(proc: asyncio.subprocess.Process, pgid: int) -> None:
    """Terminate and reap a command's process group without letting cleanup hang.

    The group is signalled even when the shell itself has already exited: the
    processes it backgrounded are precisely the ones that outlive it, and on a
    timeout or an interrupt the whole tree has to go.
    """
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


def _stop_reading(proc: asyncio.subprocess.Process) -> None:
    """Close the read end of a pipe a descendant is still holding open.

    Without this the event loop keeps draining that pipe into the stream
    buffer for as long as the descendant lives, unbounded and unread.
    """
    transport = getattr(proc.stdout, "_transport", None)
    if transport is not None:
        try:
            transport.close()
        except OSError:
            pass


async def _collect(proc: asyncio.subprocess.Process, tail: _OutputTail) -> None:
    """Read output until the command exits, then drain what is still in flight.

    Deliberately never waits for stdout EOF. A backgrounded descendant inherits
    the pipe and can hold it open for as long as it runs ("svc & echo ready"),
    which would block the turn until the timeout with the command itself long
    finished.
    """
    reader = asyncio.create_task(_pump(proc.stdout, tail))
    try:
        await asyncio.wait_for(proc.wait(), timeout=_COMMAND_TIMEOUT)
        # The command is gone. Give the pipe a moment to hand over what it
        # still holds, re-armed while bytes keep arriving so a descendant in
        # the middle of writing is not cut off mid-sentence.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _DRAIN_TOTAL
        while not reader.done():
            seen = tail.total_bytes
            try:
                await asyncio.wait_for(asyncio.shield(reader), timeout=_DRAIN_GRACE)
            except asyncio.TimeoutError:
                if tail.total_bytes == seen or loop.time() >= deadline:
                    break
    finally:
        if not reader.done():
            reader.cancel()
            _stop_reading(proc)
        try:
            await reader
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 — cleanup only
            pass


async def _shell(args: dict, cwd: Path, ctx: Optional[ToolContext] = None) -> str:
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
    # The child leads its own group, so the group id is its pid. Recorded here
    # because os.getpgid() stops working the moment that leader is reaped.
    pgid = proc.pid
    tail = _OutputTail()
    try:
        try:
            await _collect(proc, tail)
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
    # The four below are stateful in the same way: they act on the pool of
    # agents this process is running, which only the UI owns, so the agent loop
    # hands them to its supervisor instead of running them here.
    #
    # access is written out rather than left to default to "none" so the choice
    # is on the record: none of the four touches the filesystem or spawns a
    # process on its own, and everything a started agent goes on to do is gated
    # in that agent's own tab under the mode it inherited. Adding a tool here
    # that does reach outside the process needs its own access class.
    "spawn_agent": Tool(
        run=None,
        access="none",
        schema={
            "type": "function",
            "function": {
                "name": "spawn_agent",
                "description": (
                    "Start another agent working in parallel in its own tab, with the same "
                    "tools and working directory, and return its agent id. Use it for work "
                    "that is independent of what you are doing right now — exploring a second "
                    "part of the codebase, a long test run, a self-contained refactor — and "
                    "keep the number of them small. The prompt must be self-contained: state "
                    "the goal, the key file paths, the decisions already made and what to "
                    "report back, because the new agent has no memory of this conversation "
                    "and cannot ask you anything. Nothing it produces reaches you on its own: "
                    "call read_agent to collect it, and wait_for_agent to wait for it. It "
                    "cannot spawn agents of its own. Permission prompts for its tools appear "
                    "in its tab, so it can sit blocked until the user answers them."
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
                    "its own — the agent cannot see this conversation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["agent_id", "prompt"],
                },
            },
        },
    ),
    "read_agent": Tool(
        run=None,
        access="none",
        schema={
            "type": "function",
            "function": {
                "name": "read_agent",
                "description": (
                    "Read what an agent has written since you last read it (mode 'new', the "
                    "default) or its whole answer so far (mode 'all'), together with its "
                    "state. Only its assistant text comes back — not its thinking and not its "
                    "tool output — so ask it in the spawn prompt to end with the summary you "
                    "need."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["new", "all"],
                            "description": "'new' (default) since your last read, or 'all'.",
                        },
                    },
                    "required": ["agent_id"],
                },
            },
        },
    ),
    "wait_for_agent": Tool(
        run=None,
        access="none",
        schema={
            "type": "function",
            "function": {
                "name": "wait_for_agent",
                "description": (
                    "Wait until an agent is no longer running, then return its state. This "
                    "never blocks forever: when the timeout runs out it returns 'running', "
                    "and it returns early with 'needs_confirm' when the agent is stuck on a "
                    "permission prompt in its own tab — say so, because only the user can "
                    "clear that. Read its output with read_agent afterwards."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "timeout": {
                            "type": "number",
                            "description": (
                                f"Seconds to wait (optional, default {DEFAULT_WAIT_TIMEOUT:g}, "
                                f"maximum {MAX_WAIT_TIMEOUT:g})."
                            ),
                        },
                    },
                    "required": ["agent_id"],
                },
            },
        },
    ),
}

# Tools that only mean anything with the UI supervising a pool of panes: the
# agent loop refuses them without a supervisor, and headless leaves them out of
# its toolset entirely rather than offering the model something that cannot work.
SUPERVISED_TOOLS = ("spawn_agent", "send_to_agent", "read_agent", "wait_for_agent")

# What a spawned agent must not be given: spawning (depth stays 1) and the
# handoff, which would swap the session out from under the id its parent holds.
SUBAGENT_DENIED = ("spawn_agent", "start_new_session")


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
