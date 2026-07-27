"""Machine-facing subcommands: probe and provision without opening the UI.

These exist so another program (a shell script, a calling code agent) can
check whether Paimon is ready and log it in non-interactively. Nothing here
talks to the network: ``login`` validates shape and persists, the first real
turn is what proves the credentials.

Exit codes: ``status`` exits 0 when a model is configured and 1 when not;
``login`` exits 0 on success and 1 on any refusal, with the reason on stderr.
"""

import argparse
import json
import os
import sys
from importlib import metadata
from pathlib import Path
from typing import Optional

from .config import Config, activate_profile, config_path
from .llm import is_provider_available, split_model_string
from .session import Session


def _profile_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default=None, metavar="NAME",
                        help="use this named profile's configuration")


def _apply_profile(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    # Called even when the flag is absent, so repeated in-process invocations
    # never inherit a previous run's profile.
    try:
        activate_profile(args.profile)
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
    _apply_profile(parser, args)

    config = Config.load()
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
            "config_path": str(config_path()),
            "sessions_here": len(Session.list(Path.cwd())),
        }, ensure_ascii=False))
    elif logged_in:
        key_note = "api key set" if config.api_key else "no api key stored"
        print(f"paimon {version()}")
        print(f"model: {config.model} ({key_note})")
        if config.api_base:
            print(f"api base: {config.api_base}")
        print(f"config: {config_path()}")
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
                        help="endpoint override for the provider")
    key_source = parser.add_mutually_exclusive_group()
    key_source.add_argument("--api-key-env", metavar="VAR",
                            help="read the API key from this environment variable")
    key_source.add_argument("--api-key-stdin", action="store_true",
                            help="read the API key from stdin")
    _profile_option(parser)
    args = parser.parse_args(argv)
    _apply_profile(parser, args)

    try:
        provider, _ = split_model_string(args.model)
        if not is_provider_available(provider):
            raise ValueError(f"provider {provider!r} needs a dependency Paimon does not ship")
        api_key = _read_api_key(args)
    except ValueError as exc:
        print(f"paimon: {exc}", file=sys.stderr)
        return 1

    Config.load().save(model=args.model, api_base=args.api_base, api_key=api_key)
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
    "install-skill": install_skill,
}
