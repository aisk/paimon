"""Whether a tool call may run: permission modes, the read-only shell command
recognizer, the gate, and running a call once it is authorized."""

import os
import shlex
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from .base import Tool, ToolContext
from .files import _inside, _real, _resolve
from .registry import REGISTRY, execute_tool
from .result import ToolResult
from .shell import git_bash_root, shell_executable


# A confirm callback returns True to allow a dangerous tool, False to deny.
ConfirmFn = Callable[[str, dict], Awaitable[bool]]

# Who a confirmation is being asked for, when it is not the conversation whose
# hook is called: the titles of the subagents it passed through on the way up,
# outermost first, () for the conversation's own calls. A context variable
# rather than a third argument, so an embedder's two-argument hook keeps
# working and only a hook that wants to say where a request came from reads it.
_ORIGIN: ContextVar[tuple[str, ...]] = ContextVar("paimon_confirm_origin", default=())


def confirm_origin() -> tuple[str, ...]:
    """The subagents the confirmation now being asked comes from, outermost
    first; () when it is the asking conversation's own. For confirm hooks."""
    return _ORIGIN.get()


async def confirm_for(confirm: ConfirmFn, origin: str, name: str, args: dict) -> bool:
    """Ask ``confirm`` about a call that ``origin`` (a subagent's title) made.

    Set only for the duration of the call, in the asking task's context, so
    nothing else running concurrently sees it.
    """
    token = _ORIGIN.set((origin, *_ORIGIN.get()))
    try:
        return await confirm(name, args)
    finally:
        _ORIGIN.reset(token)

# A review callback judges a call auto mode will not let through on its own:
# None lets it run, a string is the refusal the model is shown.
ReviewFn = Callable[[str, dict], Awaitable[str | None]]


# Permission modes: read (reads inside cwd, everything else refused), auto
# (writes inside cwd too, and a reviewer model decides the rest), yolo (no
# check at all). In read and auto modes, shell commands recognized by
# safe_command() run as well unless the safe_commands toggle is off.
MODES = ("read", "auto", "yolo")

USER_DENIAL = "User denied this operation."
READ_DENIAL = ("Denied: read mode only allows reading inside the working directory and "
               "clearly read-only shell commands. Do not look for another way to do "
               "this; say what you would have done instead.")


# Any of these anywhere in the raw command string — even inside quotes, since
# "$" and backticks expand inside double quotes and the rest are rare enough
# quoted that fail-closed simplicity wins — means the segment view built below
# may not be what the shell actually runs (substitution, redirection,
# escaping, brace expansion, glob character classes that can match ".."), so
# the command is never auto-allowed. A false rejection only costs a review
# in auto mode, so this errs hard toward rejection. ";", "&" and "|"
# are not here: _split_segments handles them position-aware, so quoted forms
# like grep "a|b" stay allowed.
_SHELL_METACHARS = frozenset("$`<>()\\{[\n\r")


def _split_segments(command: str) -> list[tuple[str, str]] | None:
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


# Builtins in bash and in sh alike, so never looked up on PATH.
_BUILTINS = frozenset({"cd", "echo", "pwd"})
# Under Git Bash every other command above must be the one Git for Windows
# ships, the GNU tool this list was written for; these alone are not shipped
# and come from the user's PATH.
_NOT_FROM_GIT = frozenset({"rg"})
# Win32 device names. Opened by a native program (rg, git) they are devices
# in any directory and with any extension: CON reads the console paimon's
# own TUI is reading.
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"{port}{n}" for port in ("com", "lpt") for n in "0123456789¹²³"}
)


def _lookup_is_sound(name: str, shell: str) -> bool:
    """Whether ``name`` runs the program the analysis assumes it does.

    The lists above assume "cat" is the system's cat. A PATH entry that is
    relative (an empty one, in sh, means the current directory) lets a file
    the agent wrote into the working directory a moment ago run instead, so
    such an entry ahead of the command's real location fails the check.

    Under Git Bash the shell puts Git's own directories ahead of the Windows
    PATH (see find_git_bash), so a command found there is certain; anything
    else would be some Windows program of the same name with other semantics
    (System32's find.exe), and is refused, except for the few commands that
    are expected to come from PATH.
    """
    path = os.environ.get("PATH")
    if os.name == "nt":
        root = git_bash_root(shell)
        if root is None:
            return False  # not a shell whose lookup is modelled here
        for directory in ("mingw64", "clangarm64", "mingw32", "usr"):
            folder = root / directory / "bin"
            if (folder / f"{name}.exe").is_file() or (folder / name).is_file():
                return True
        if name not in _NOT_FROM_GIT:
            return False
        # Where on the Windows PATH it is found does not matter, provided no
        # entry is relative; MSYS drops the empty ones rather than reading
        # them as the current directory.
        return all(os.path.isabs(entry) for entry in (path or "").split(";") if entry)
    if path is None:
        return True  # the shell's built-in default, which is all absolute
    for entry in path.split(os.pathsep):
        if not os.path.isabs(entry):
            return False
        candidate = os.path.join(entry, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return True
    return True  # not found anywhere: the command fails, and nothing runs


def _windows_path_ok(tok: str) -> bool:
    """Whether Python's Windows path rules and Git Bash agree on ``tok``.

    _inside resolves a token the Win32 way and the shell's programs resolve
    it the MSYS way, and only some tokens mean the same thing to both:

    - a leading "/" is an MSYS path: /tmp and /etc are directories in Git's
      install or the user's temp directory, /c/x is drive C, //host/share is
      UNC, /dev and /proc are emulated, while Python reads each as a
      directory on the current drive. A drive path written "C:/x" is
      understood alike by both and stays allowed.
    - any other ":" is a drive-relative path ("C:x", resolved against
      a per-drive directory Python and MSYS do not share) or an alternate
      data stream ("file:stream").
    - Win32 strips trailing dots and spaces from a name and MSYS does not,
      so "x." is two different files.
    - a reserved device name is a device to a native program wherever it is.

    Backslashes never reach this: they are rejected with the metacharacters.
    """
    if tok.startswith("/"):
        return False
    drive = len(tok) > 2 and tok[0].isascii() and tok[0].isalpha() and tok[1:3] == ":/"
    rest = tok[3:] if drive else tok
    if ":" in rest:
        return False
    for segment in rest.split("/"):
        if segment in ("", ".", ".."):
            continue
        if segment[-1] in ". ":
            return False
        if segment.split(".")[0].rstrip(" ").lower() in _RESERVED_NAMES:
            return False
    return True


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


def _safe_cd(argv: list[str], base: Path, root: Path) -> Path | None:
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
    if os.name == "nt" and not _windows_path_ok(tok):
        return None
    target = _resolve(tok, base)
    # Resolved so later segments check against the directory the kernel will
    # actually be in (without "..", physical and logical agree).
    return target.resolve() if _inside(target, root) else None


def _safe_simple(argv: list[str], base: Path, root: Path, shell: str) -> Path | None:
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
    # bash imports exported functions, and a function is found before a
    # builtin or anything on PATH. Only one named like the command matters:
    # some systems export a few of their own (Fedora's "which"), and refusing
    # on any of them would turn the allowance off there altogether.
    if f"BASH_FUNC_{argv[0]}%%" in os.environ:
        return None
    if argv[0] not in _BUILTINS and not _lookup_is_sound(argv[0], shell):
        return None
    windows = os.name == "nt"
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
            # On Windows a drive letter needs no "/" ("-fC:x").
            if _flag_denied(tok, spec.deny_flags) or "/" in tok or (windows and ":" in tok):
                return None
        elif _hidden_glob(tok):
            return None
        elif spec.check_paths and windows and not _windows_path_ok(tok):
            return None
        elif spec.check_paths and not _inside(_resolve(tok, base), root):
            return None
    return base


def safe_command(command: str, cwd: Path) -> bool:
    """Conservatively decide whether a shell command is clearly read-only.

    A guardrail against agent mistakes, not a security boundary: anything not
    positively recognized returns False and the call is held like any other
    command: reviewed in auto mode, refused in read mode.

    Compound commands pass when every &&/||/;/|-separated segment passes on
    its own. A cd segment moves the base later relative paths resolve
    against, but containment is always checked against the original cwd.

    What runs the command decides what can be recognized:

    - POSIX (bash or sh): the analysis below, with paths resolved the POSIX
      way and each command looked up on PATH the way the shell will.
    - Windows with Git Bash: the same analysis, since bash parses the line,
      plus what MSYS changes. A command must be the one Git ships (System32's
      find.exe is not GNU find, and tree has no GNU counterpart there), and a
      path token must mean the same file to Python's Win32 resolution and to
      MSYS: no "/"-rooted MSYS path, drive-relative path, alternate data
      stream, trailing dot or space, or device name (see _windows_path_ok).
      MSYS also rewrites "/"-rooted arguments before a native program such as
      git or rg sees them; those are rejected for the same reason, except as
      git pathspecs, where git's own repository boundary applies.
    - Windows with cmd.exe (no Git Bash): nothing, as everything here models
      sh. cmd does not treat "'" as a quote, so "echo ' && del x && echo '"
      reads as one echo here and runs as three commands there, and it looks
      a command up in the current directory before PATH, so "cat" may be a
      cat.bat the agent wrote a moment ago.
    """
    shell = shell_executable()
    if shell is None:
        return False
    # bash runs BASH_ENV before the command, and that file can redefine "ls".
    if os.environ.get("BASH_ENV"):
        return False
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
        new_base = _safe_simple(argv, base, cwd, shell)
        if new_base is None:
            return False
        base = new_base
    return True


def gate(name: str, args: dict, mode: str, cwd: Path,
         registry: dict[str, Tool] | None = None,
         safe_commands: bool = True,
         ctx: ToolContext | None = None) -> str:
    """Decide what stands between a tool call and running it.

    "allow" is nothing. A call the mode does not let through on its own is
    held: "deny" in read mode, "review" in auto mode.

    ``safe_commands`` lets shell commands recognized as clearly read-only
    through (see safe_command); False holds every command.
    """
    tool = (REGISTRY if registry is None else registry).get(name)
    if mode == "yolo" or tool is None or tool.access == "none":
        return "allow"
    held = "review" if mode == "auto" else "deny"
    if tool.access == "execute":
        if safe_commands and safe_command(str(args.get("command") or ""), cwd):
            return "allow"
        return held
    if tool.access == "background":
        # No safe_command exception here. That list is about what a command
        # reads and writes, and a process that never ends on its own is not
        # harmless just because it only reads: "tail -f" is on the deny side of
        # it precisely because it runs until the timeout — and a background
        # command has no timeout to run into.
        return held
    # A missing/malformed path resolves to cwd itself; the tool then fails on its own.
    resolved = _resolve(str(args.get("path") or ""), cwd)
    inside = _inside(resolved, cwd)
    if tool.access == "read":
        # An overflow file this agent's own command produced is paimon's record
        # of something already let through, so reading it back is not a
        # new escalation. The exemption is per agent on purpose: the directory
        # itself is shared by every session and project on the machine, and a
        # blanket allowance would let one agent read another one's command
        # output — a different project's, or last week's — unchecked.
        own_output = ctx is not None and _real(resolved) in ctx.shell_outputs
        return "allow" if inside or own_output else held
    if tool.access == "write" and mode == "auto" and inside and not _in_git_dir(resolved, cwd):
        return "allow"
    return held


def _in_git_dir(path: Path, cwd: Path) -> bool:
    """Whether ``path`` (inside cwd) lies in a .git directory, or is one.

    Writing there is running code: a hook, or a config key such as
    core.fsmonitor, executes on the next git command — and git status and
    git log are let through as read-only. So these writes go to review like
    any other way of running something unreviewed.
    """
    real, base = _real(path), _real(cwd)
    if real is None or base is None:
        return True
    try:
        parts = real.relative_to(base).parts
    except ValueError:
        return False
    # Case-insensitive, and with the trailing dots and spaces Windows strips
    # from a name: ".GIT" and ".git." are the same directory there.
    return any(part.rstrip(". ").lower() == ".git" for part in parts)


async def authorize(name: str, args: dict, mode: str, cwd: Path,
                    confirm: ConfirmFn | None = None,
                    review: ReviewFn | None = None,
                    registry: dict[str, Tool] | None = None,
                    safe_commands: bool = True,
                    ctx: ToolContext | None = None) -> str | None:
    """Gate a tool call and put it to whoever decides: None when it may run,
    else the refusal the model is shown.

    This is the enforcement point. A held call is never let through for want
    of someone to ask: without a reviewer it goes to the confirm hook, and
    without that either it is refused, so a headless Agent cannot bypass the
    permission mode.
    """
    decision = gate(name, args, mode, cwd, registry, safe_commands=safe_commands, ctx=ctx)
    if decision == "allow":
        return None
    if decision == "deny":
        return READ_DENIAL
    if decision == "review" and review is not None:
        return await review(name, args)
    allowed = await confirm(name, args) if confirm else False
    return None if allowed else USER_DENIAL


async def run_tool(name: str, args: dict, cwd: Path, mode: str,
                   confirm: ConfirmFn | None = None,
                   registry: dict[str, Tool] | None = None,
                   safe_commands: bool = True,
                   ctx: ToolContext | None = None,
                   review: ReviewFn | None = None) -> ToolResult:
    """Authorize, then execute a tool call. A refusal is a "denied" result.

    ``registry`` narrows the available tools (an agent's own set); None means
    the full REGISTRY. ``ctx`` carries the calling agent's own state, so gating
    decisions that depend on what this agent did earlier stay scoped to it.
    """
    refusal = await authorize(name, args, mode, cwd, confirm, review, registry,
                              safe_commands=safe_commands, ctx=ctx)
    if refusal is not None:
        return ToolResult(refusal, "denied")
    return await execute_tool(name, args, cwd, mode=mode, registry=registry, ctx=ctx)
