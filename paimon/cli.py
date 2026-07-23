"""Command-line entry point: argument parsing and launch modes."""

import argparse
import shlex
import sys

from .app import PaimonApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Paimon terminal code agent")
    parser.add_argument("-c", "--continue", dest="continue_session", action="store_true",
                        help="continue the most recent session for this directory")
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

        flags = ["-c"] if args.continue_session else []
        if args.mode != "read":
            flags += ["--mode", args.mode]
        command = shlex.join([sys.executable, "-m", "paimon", *flags])
        Server(command, port=args.port).serve()
        return
    PaimonApp(continue_session=args.continue_session, mode=args.mode).run()


if __name__ == "__main__":
    main()
