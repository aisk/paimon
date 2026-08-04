"""Machine-facing subcommands: probe and provision without opening the UI.

These exist so another program (a shell script, a calling code agent) can
check whether Paimon is ready and log it in non-interactively. Nothing here
talks to the network: ``login`` validates shape and persists, the first real
turn is what proves the credentials.

Exit codes: ``status`` exits 0 when a model is configured and 1 when not;
``login`` exits 0 on success and 1 on any refusal, with the reason on stderr;
``log`` exits 0 on success and 1 when the session cannot be found.
"""

import argparse
import json
import os
import sys
from importlib import metadata
from pathlib import Path
from typing import Optional

from .config import UNSET, Config, config_path, validate_profile
from .llm import is_provider_available, split_model_string
from .session import SUMMARY_PREFIX, Session
from .tools import summarize_call


def resolve_session(prefix: str) -> Session:
    """The unique session of the current directory matching the id prefix."""
    matches = [session for session in Session.list(Path.cwd()) if session.id.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"no session matching '{prefix}' in this directory")
    raise ValueError(f"ambiguous session id '{prefix}' ({len(matches)} matches)")


def latest_session() -> Session:
    sessions = Session.list(Path.cwd())
    if not sessions:
        raise ValueError("no session in this directory")
    return sessions[0]


def _profile_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default=None, metavar="NAME",
                        help="use this named profile's configuration")


def _resolve_profile(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    try:
        return validate_profile(args.profile)
    except ValueError as exc:
        parser.error(str(exc))


def version() -> str:
    try:
        return metadata.version("paimon")
    except metadata.PackageNotFoundError:
        return "unknown"


def status(argv: list) -> int:
    """Report whether Paimon is ready to run, without creating anything."""
    parser = argparse.ArgumentParser(
        prog="paimon status",
        description="Show login state and configuration. Exits 0 when a model "
                    "is configured, 1 when login is still needed.",
    )
    parser.add_argument("--json", action="store_true", help="one JSON object on stdout")
    _profile_option(parser)
    args = parser.parse_args(argv)
    profile = _resolve_profile(parser, args)

    config = Config.load(profile)
    logged_in = bool(config.model)
    if args.json:
        # The api_key itself is deliberately absent: status output is meant
        # to be pasted into logs and other agents' contexts.
        print(json.dumps({
            "version": version(),
            "logged_in": logged_in,
            "model": config.model,
            "api_base": config.api_base,
            "api_key_set": bool(config.api_key),
            "safe_commands": config.safe_commands,
            "config_path": str(config_path(profile)),
            "sessions_here": len(Session.list(Path.cwd())),
        }, ensure_ascii=False))
    elif logged_in:
        key_note = "api key set" if config.api_key else "no api key stored"
        print(f"paimon {version()}")
        print(f"model: {config.model} ({key_note})")
        if config.api_base:
            print(f"api base: {config.api_base}")
        if not config.safe_commands:
            print("safe read-only commands: off (strict)")
        print(f"config: {config_path(profile)}")
        print(f"sessions here: {len(Session.list(Path.cwd()))}")
    else:
        print(f"paimon {version()}")
        print("not logged in — run 'paimon' for interactive setup, or "
              "'paimon login --model provider:name --api-key-env VAR'")
    return 0 if logged_in else 1


def _read_api_key(args: argparse.Namespace) -> Optional[str]:
    """The key named by the flags, or None when no key flag was passed.

    Raises ValueError with the reason when a flag was passed but yields
    nothing — an empty key silently stored would fail much later, mid-turn.
    """
    if args.api_key_env:
        key = os.environ.get(args.api_key_env, "").strip()
        if not key:
            raise ValueError(f"environment variable {args.api_key_env} is unset or empty")
        return key
    if args.api_key_stdin:
        key = sys.stdin.read().strip()
        if not key:
            raise ValueError("no api key arrived on stdin")
        return key
    return None


def login(argv: list) -> int:
    """Persist model settings from flags instead of the interactive flow.

    The key is never taken on the command line — argv is visible to every
    process on the machine. Fields not passed keep their current values, so
    ``login --model x:y`` switches just the model.
    """
    parser = argparse.ArgumentParser(
        prog="paimon login",
        description="Log in without the UI. Fields not passed keep their current values.",
    )
    parser.add_argument("--model", required=True, metavar="PROVIDER:NAME",
                        help="model to use, e.g. 'zai:glm-4.7'")
    parser.add_argument("--api-base", metavar="URL",
                        help="endpoint override for the provider (pass '' to clear a stored one)")
    key_source = parser.add_mutually_exclusive_group()
    key_source.add_argument("--api-key-env", metavar="VAR",
                            help="read the API key from this environment variable")
    key_source.add_argument("--api-key-stdin", action="store_true",
                            help="read the API key from stdin")
    _profile_option(parser)
    args = parser.parse_args(argv)
    profile = _resolve_profile(parser, args)

    try:
        provider, _ = split_model_string(args.model)
        if not is_provider_available(provider):
            raise ValueError(f"provider {provider!r} needs a dependency Paimon does not ship")
        api_key = _read_api_key(args)
    except ValueError as exc:
        print(f"paimon: {exc}", file=sys.stderr)
        return 1

    Config.load(profile).save(
        model=args.model,
        # An absent flag keeps the stored value; only an explicit '' clears it.
        api_base=args.api_base if args.api_base is not None else UNSET,
        api_key=api_key if api_key is not None else UNSET,
    )
    print(f"logged in: {args.model}")
    return 0


def _preview(text: Optional[str], limit: int = 60) -> str:
    line = " ".join((text or "").split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


def sessions(argv: list) -> int:
    """List the sessions of the current directory, newest first."""
    parser = argparse.ArgumentParser(
        prog="paimon sessions",
        description="List resumable sessions in this directory, newest first.",
    )
    parser.add_argument("--json", action="store_true", help="one JSON array on stdout")
    args = parser.parse_args(argv)

    found = Session.list(Path.cwd())
    if args.json:
        print(json.dumps([{
            "id": session.id,
            "created_at": session.created_at(),
            "preview": _preview(session.first_user_text()),
            "path": str(session.path),
        } for session in found], ensure_ascii=False))
        return 0
    if not found:
        print("no sessions in this directory", file=sys.stderr)
        return 0
    for session in found:
        created = session.created_at() or ""
        print(f"{session.id[:8]}  {created:25}  {_preview(session.first_user_text())}".rstrip())
    return 0


# Compact log lines are collapsed to one line and clipped at this width.
LOG_DETAIL_WIDTH = 120


def _chars(text: str) -> str:
    n = len(text)
    return f"{n / 1000:.1f}k chars" if n >= 1000 else f"{n} chars"


def _clip(text: object, full: bool) -> str:
    if full:
        return str(text)
    line = " ".join(str(text).split())
    return line if len(line) <= LOG_DETAIL_WIDTH else line[: LOG_DETAIL_WIDTH - 1] + "…"


def _is_turn_start(record: Optional[dict]) -> bool:
    """True for a message that opens a turn: a real user prompt, not a
    compaction summary (same test as Session.first_user_text)."""
    if not record or record.get("type") != "message":
        return False
    message = record.get("message")
    if not isinstance(message, dict) or message.get("kind") != "request":
        return False
    for part in message.get("parts") or []:
        content = part.get("content") if isinstance(part, dict) else None
        if (isinstance(part, dict) and part.get("part_kind") == "user-prompt"
                and isinstance(content, str) and not content.startswith(SUMMARY_PREFIX)):
            return True
    return False


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


def _render_record(seq: int, record: Optional[dict], full: bool) -> list[str]:
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
    return [f"[{seq}] {kind or '?'}"]


def log(argv: list) -> int:
    """Render a session's JSONL event log, whole or incrementally.

    Seq numbers are physical line numbers (1-based), stable because the log is
    append-only; corrupt lines keep their number so cursors never shift. A
    caller that saved ``log_end`` from a ``--output-format result`` run passes
    it to ``--after`` to see exactly what happened since.
    """
    parser = argparse.ArgumentParser(
        prog="paimon log",
        description="Show a session's event log. Each line is prefixed with its "
                    "seq; pass the highest seq already seen to --after to get "
                    "only what is new.",
    )
    parser.add_argument("session", nargs="?", metavar="ID",
                        help="session id prefix (default: the latest session here)")
    window = parser.add_mutually_exclusive_group()
    window.add_argument("--after", type=int, metavar="SEQ",
                        help="only records with seq greater than SEQ")
    window.add_argument("--turns", type=int, metavar="N", help="only the last N turns")
    window.add_argument("--tail", type=int, metavar="N", help="only the last N records")
    parser.add_argument("--json", action="store_true",
                        help="raw records, one JSON object per line with a seq field")
    parser.add_argument("--full", action="store_true",
                        help="full content instead of one clipped line per part")
    args = parser.parse_args(argv)
    if args.after is not None and args.after < 0:
        parser.error("--after must be >= 0")
    for flag, value in (("--turns", args.turns), ("--tail", args.tail)):
        if value is not None and value < 1:
            parser.error(f"{flag} must be >= 1")

    try:
        session = resolve_session(args.session) if args.session else latest_session()
    except ValueError as exc:
        print(f"paimon: {exc}", file=sys.stderr)
        return 1

    entries: list = []  # (seq, record | None for a corrupt line)
    try:
        with session.path.open(encoding="utf-8") as file:
            for seq, line in enumerate(file, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = None
                entries.append((seq, record if isinstance(record, dict) else None))
    except OSError as exc:
        print(f"paimon: cannot read {session.path}: {exc}", file=sys.stderr)
        return 1

    if args.after is not None:
        entries = [entry for entry in entries if entry[0] > args.after]
    elif args.turns is not None:
        starts = [i for i, (_, record) in enumerate(entries) if _is_turn_start(record)]
        entries = entries[starts[max(0, len(starts) - args.turns)]:] if starts else []
    elif args.tail is not None:
        entries = entries[-args.tail:]

    for seq, record in entries:
        if args.json:
            payload = {"seq": seq, "corrupt": True} if record is None else {"seq": seq, **record}
            print(json.dumps(payload, ensure_ascii=False, default=str))
        else:
            for line in _render_record(seq, record, args.full):
                print(line)
    return 0


# Where each calling agent discovers user-level skills.
SKILL_DIRS = {
    "claude": Path(".claude") / "skills" / "paimon",
    "codex": Path(".codex") / "skills" / "paimon",
}


def install_skill(argv: list) -> int:
    """Copy the bundled SKILL.md to where a calling agent will find it.

    A copy rather than a symlink: uv/pipx tool upgrades replace the package
    directory, and a skill that dangles until the next install-skill run is
    worse than one that lags a version behind.
    """
    parser = argparse.ArgumentParser(
        prog="paimon install-skill",
        description="Teach a calling code agent how to drive Paimon.",
    )
    parser.add_argument("--target", choices=sorted(SKILL_DIRS), default="claude",
                        help="which agent's skill directory to install into (default: claude)")
    parser.add_argument("--dest", metavar="DIR", default=None,
                        help="install into this directory instead of the --target default")
    args = parser.parse_args(argv)

    source = Path(__file__).parent / "skill" / "SKILL.md"
    directory = Path(args.dest).expanduser() if args.dest else Path.home() / SKILL_DIRS[args.target]
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "SKILL.md"
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"installed: {destination}")
    return 0


# Dispatched by cli.main before its own parser runs; each entry takes the
# remaining argv and returns the process exit code.
REGISTRY = {
    "status": status,
    "login": login,
    "sessions": sessions,
    "log": log,
    "install-skill": install_skill,
}
