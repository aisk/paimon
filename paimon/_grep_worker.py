"""Disposable process for bounded, cancellable code searches."""

import json
import sys
from pathlib import Path

from .tools import _grep


def main() -> None:
    args, cwd, sandboxed = json.load(sys.stdin)
    sys.stdout.write(json.dumps(_grep(args, Path(cwd), sandboxed=sandboxed)))


if __name__ == "__main__":
    main()
