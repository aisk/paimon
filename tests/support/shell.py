"""Shell output generators and readers shared by tool tests."""

import re
from pathlib import Path


def filler(lines: int, payload: str, newline: bool = True) -> str:
    """A POSIX sh loop printing ``payload`` ``lines`` times (/bin/sh, not bash)."""
    fmt = "%s\\n" if newline else "%s"
    return f'i=0; while [ $i -lt {lines} ]; do printf "{fmt}" "{payload}"; i=$((i+1)); done'


def overflow_path(result: str) -> Path:
    match = re.search(r"full output: (\S+?)\]", result)
    assert match, f"no overflow path in result: {result[-300:]!r}"
    return Path(match.group(1))
