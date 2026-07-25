"""Command-line entry point: argument parsing and launch modes."""

import argparse
import shlex
import sys
from pathlib import Path

from . import headless as headless_mode
from .agent import Agent
from .app import PaimonApp
from .session import Session, SessionError, resume_hint


class CliError(Exception):
    """A message to report as ``paimon: ...`` before exiting with status 1."""


def _resolve_session(prefix: str) -> Session:
    matches = [session for session in Session.list(Path.cwd()) if session.id.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise CliError(f"no session matching '{prefix}' in this directory")
    raise CliError(f"ambiguous session id '{prefix}' ({len(matches)} matches)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paimon terminal code agent")
    parser.add_argument("-r", "--resume", nargs="?", const="", default=None, metavar="ID",
                        help="resume a session: with a session id prefix resume it directly, "
                             "without a value open a session picker")
    parser.add_argument("-p", "--print", nargs="?", const="", default=None, metavar="PROMPT",
                        dest="prompt",
                        help="run one turn without the UI and exit; with no value the prompt "
                             "is read from stdin")
    parser.add_argument("--output-format", choices=("text", "json"), default="text",
                        help="output for --print: text (default) or one JSON event per line")
    parser.add_argument("--mode", choices=("read", "edit", "yolo"), default="read",
                        help="permission mode: read (confirm writes, commands and reads outside cwd), "
                             "edit (auto-approve edits in cwd), yolo (no confirmation)")
    parser.add_argument("--web", action="store_true",
                        help="serve the app in a browser instead of the terminal")
    parser.add_argument("--port", type=int, default=8000,
                        help="port for --web (default: 8000)")
    # Set when the UI is launched with a stdin that is not a terminal, which
    # would otherwise be read as "run headless" (textual-serve pipes stdin).
    parser.add_argument("--tui", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ehe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.ehe:
        print("エヘッてなんだよ！！")
        return

    # Decided before touching stdin so serving the UI never consumes a pipe.
    if args.web:
        if args.prompt is not None:
            parser.error("--web and --print cannot be combined")
        from textual_serve.server import Server

        flags = ["--tui"]
        if args.resume is not None:
            flags += ["--resume"] if args.resume == "" else ["--resume", args.resume]
        if args.mode != "read":
            flags += ["--mode", args.mode]
        command = shlex.join([sys.executable, "-m", "paimon", *flags])
        Server(command, port=args.port).serve()
        return

    piped_stdin = not args.tui and not sys.stdin.isatty()
    headless = args.prompt is not None or piped_stdin
    if args.output_format != "text" and not headless:
        parser.error("--output-format only applies to --print")
    if headless and args.resume == "":
        parser.error("--resume needs a session id with --print (the picker needs a terminal)")

    try:
        resume_session = _resolve_session(args.resume) if args.resume else None
    except CliError as exc:
        if headless:
            sys.exit(headless_mode.fail(str(exc), args.output_format))
        print(f"paimon: {exc}", file=sys.stderr)
        sys.exit(1)

    if headless:
        piped = headless_mode.read_stdin() if piped_stdin else ""
        if not (args.prompt or "").strip() and not piped.strip():
            parser.error("nothing to do: pass a prompt to --print or pipe one on stdin")
        sys.exit(headless_mode.run(
            prompt=args.prompt or "", piped=piped, cwd=Path.cwd(), mode=args.mode,
            session=resume_session, output_format=args.output_format,
        ))

    try:
        agent = Agent.open(cwd=Path.cwd(), session=resume_session, mode=args.mode)
    except SessionError as exc:
        print(f"paimon: {exc}", file=sys.stderr)
        sys.exit(1)
    app = PaimonApp(agent, resumed=resume_session is not None, pick_session=args.resume == "")
    try:
        app.run()
    finally:
        # Also on the way out of a crash: that is when knowing how to get the
        # conversation back matters most. Skipped under --tui, where the
        # streams belong to textual-serve rather than to a user.
        if not args.tui and app.agent.history:
            print(f"resume: {resume_hint(app.agent.session.id)}", file=sys.stderr)


if __name__ == "__main__":
    main()
