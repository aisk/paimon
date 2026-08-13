"""Command-line entry point: argument parsing and launch modes."""

import argparse
import shlex
import sys
from pathlib import Path

from . import commands
from . import headless as headless_mode
from .agent import Agent
from .app import PaimonApp
from .config import Config
from .llm import split_model_string
from .session import SessionError, resume_hint


def main() -> None:
    # Subcommands are dispatched before the flag parser: they have their own
    # argument sets and none of the launch-mode logic below applies to them.
    argv = sys.argv[1:]
    if argv and argv[0] in commands.REGISTRY:
        sys.exit(commands.REGISTRY[argv[0]](argv[1:]))

    parser = argparse.ArgumentParser(
        description="Paimon terminal code agent",
        epilog="commands: status (login state and configuration), "
               "login (log in without the UI), sessions (list resumable sessions), "
               "log (inspect a session's event log), "
               "install-skill (teach a calling code agent how to drive paimon)",
    )
    parser.add_argument("-r", "--resume", nargs="?", const="", default=None, metavar="ID",
                        help="resume a session: with a session id prefix resume it directly, "
                             "without a value open a session picker")
    parser.add_argument("-c", "--continue", dest="continue_latest", action="store_true",
                        help="resume the most recent session in this directory")
    parser.add_argument("--model", default=None, metavar="PROVIDER:NAME",
                        help="model for this run only; the configured one is untouched")
    parser.add_argument("--profile", default=None, metavar="NAME",
                        help="use this named profile's configuration "
                             "(an independent config dir; log in with "
                             "'paimon login --profile NAME ...')")
    parser.add_argument("-p", "--print", nargs="?", const="", default=None, metavar="PROMPT",
                        dest="prompt",
                        help="run one turn without the UI and exit; with no value the prompt "
                             "is read from stdin")
    parser.add_argument("--output-format", choices=("text", "json", "result"), default="text",
                        help="output for --print: text (default), one JSON event per line, "
                             "or just the final result as one JSON object")
    parser.add_argument("--timeout", type=float, default=None, metavar="SECS",
                        help="with --print: give up after this many seconds (exit 124)")
    parser.add_argument("--max-tool-calls", type=int, default=None, metavar="N",
                        help="with --print: stop before the N+1th tool call (exit 4)")
    parser.add_argument("--append-system-prompt", default=None, metavar="TEXT",
                        help="with --print: add TEXT (e.g. a role definition) to the end of "
                             "the new session's system prompt; persisted with the session, "
                             "so not combinable with --continue/--resume")
    parser.add_argument("--mode", choices=("read", "edit", "yolo"), default="read",
                        help="permission mode: read (confirm writes, non-read-only commands and "
                             "reads outside cwd), edit (auto-approve edits in cwd), yolo (no confirmation)")
    parser.add_argument("--strict", action="store_true",
                        help="always ask before shell commands, even clearly read-only ones "
                             "(overrides the safe_commands config for this run)")
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

    if args.continue_latest and args.resume is not None:
        parser.error("--continue and --resume cannot be combined")
    # Loaded here, once: the profile is a property of this Config instance,
    # so everything downstream just carries the instance.
    try:
        config = Config.load(args.profile)
    except ValueError as exc:
        parser.error(str(exc))
    if args.strict:
        config.safe_commands = False  # session-only; save() never persists this key
    if args.model is not None:
        try:
            split_model_string(args.model)
        except ValueError as exc:
            parser.error(str(exc))

    # Decided before touching stdin so serving the UI never consumes a pipe.
    if args.web:
        if args.prompt is not None:
            parser.error("--web and --print cannot be combined")
        from textual_serve.server import Server

        flags = ["--tui"]
        if args.resume is not None:
            flags += ["--resume"] if args.resume == "" else ["--resume", args.resume]
        if args.continue_latest:
            flags += ["--continue"]
        if args.mode != "read":
            flags += ["--mode", args.mode]
        if args.strict:
            flags += ["--strict"]
        if args.model:
            flags += ["--model", args.model]
        if args.profile:
            flags += ["--profile", args.profile]
        command = shlex.join([sys.executable, "-m", "paimon", *flags])
        Server(command, port=args.port).serve()
        return

    piped_stdin = not args.tui and not sys.stdin.isatty()
    headless = args.prompt is not None or piped_stdin
    if args.output_format != "text" and not headless:
        parser.error("--output-format only applies to --print")
    for flag, value in (("--timeout", args.timeout), ("--max-tool-calls", args.max_tool_calls)):
        if value is not None:
            if not headless:
                parser.error(f"{flag} only applies to --print")
            if value < 0 or (flag == "--timeout" and value == 0):
                parser.error(f"{flag} must be positive")
    if args.append_system_prompt is not None:
        if not headless:
            parser.error("--append-system-prompt only applies to --print")
        if args.continue_latest or args.resume is not None:
            parser.error("--append-system-prompt cannot be combined with --continue/--resume: "
                         "the system prompt is persisted when the session is created")
        if not args.append_system_prompt.strip():
            parser.error("--append-system-prompt needs a non-empty value")
    if headless and args.resume == "":
        parser.error("--resume needs a session id with --print (the picker needs a terminal)")

    try:
        if args.continue_latest:
            resume_session = commands.latest_session()
        else:
            resume_session = commands.resolve_session(args.resume) if args.resume else None
    except ValueError as exc:
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
            session=resume_session, output_format=args.output_format, config=config,
            model=args.model, timeout=args.timeout, max_tool_calls=args.max_tool_calls,
            append_system_prompt=args.append_system_prompt,
        ))

    if args.model:
        config.model = args.model
    try:
        agent = Agent.open(cwd=Path.cwd(), session=resume_session, mode=args.mode, config=config)
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
        if not args.tui:
            for pane in app.sessions:
                if pane.agent.history:
                    print(f"resume: {resume_hint(pane.agent.session.id)}", file=sys.stderr)


if __name__ == "__main__":
    main()
