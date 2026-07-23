"""Command-line entry point: argument parsing and launch modes."""

import argparse
import shlex
import sys
from pathlib import Path

from .app import PaimonApp
from .session import Session


def _resolve_session(prefix: str) -> Session:
    matches = [session for session in Session.list(Path.cwd()) if session.id.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        print(f"paimon: no session matching '{prefix}' in this directory", file=sys.stderr)
    else:
        print(f"paimon: ambiguous session id '{prefix}' ({len(matches)} matches)", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paimon terminal code agent")
    parser.add_argument("-r", "--resume", nargs="?", const="", default=None, metavar="ID",
                        help="resume a session: with a session id prefix resume it directly, "
                             "without a value open a session picker")
    parser.add_argument("--mode", choices=("read", "edit", "yolo"), default="read",
                        help="permission mode: read (confirm writes, commands and reads outside cwd), "
                             "edit (auto-approve edits in cwd), yolo (no confirmation)")
    parser.add_argument("--web", action="store_true",
                        help="serve the app in a browser instead of the terminal")
    parser.add_argument("--port", type=int, default=8000,
                        help="port for --web (default: 8000)")
    parser.add_argument("--ehe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.ehe:
        print("エヘッてなんだよ！！")
        return
    if args.web:
        from textual_serve.server import Server

        flags = []
        if args.resume is not None:
            flags += ["--resume"] if args.resume == "" else ["--resume", args.resume]
        if args.mode != "read":
            flags += ["--mode", args.mode]
        command = shlex.join([sys.executable, "-m", "paimon", *flags])
        Server(command, port=args.port).serve()
        return
    resume_session = _resolve_session(args.resume) if args.resume else None
    PaimonApp(mode=args.mode, session=resume_session, pick_session=args.resume == "").run()


if __name__ == "__main__":
    main()
