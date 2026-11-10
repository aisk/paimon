"""Disposable process for bounded, cancellable code searches."""

import json
import sys
from pathlib import Path

from .tools.grep import _grep
from .tools.result import ToolError


def main() -> None:
    args, cwd, sandboxed = json.load(sys.stdin)
    try:
        result = _grep(args, Path(cwd), sandboxed=sandboxed)
    except ToolError as exc:
        # An object rather than a string, so the failure crosses the process
        # boundary as one instead of being read back out of the text.
        result = {"error": exc.text}
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
