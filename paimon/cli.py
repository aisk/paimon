"""Command-line entry point: argument parsing and launch modes."""

import argparse
import shlex
import sys

from .app import PaimonApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Paimon terminal code agent")
    parser.add_argument("-c", "--continue", dest="continue_session", action="store_true",
                        help="continue the most recent session for this directory")
    parser.add_argument("--yolo", action="store_true",
                        help="allow dangerous tool calls without confirmation")
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

        flags = [flag for flag, enabled in (("-c", args.continue_session), ("--yolo", args.yolo)) if enabled]
        command = shlex.join([sys.executable, "-m", "paimon", *flags])
        Server(command, port=args.port).serve()
        return
    PaimonApp(continue_session=args.continue_session, yolo=args.yolo).run()


if __name__ == "__main__":
    main()
