"""Side-by-side diff rendering for tool confirmations.

Prefers `delta` (https://github.com/dandavison/delta) when it is on PATH for
syntax-highlighted side-by-side output; otherwise falls back to a pure-difflib
two-column table so no external tool is required.
"""

import difflib
import shutil
import subprocess

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text


def render_diff(old: str, new: str, *, width: int) -> RenderableType:
    return _delta_diff(old, new, width) or _table_diff(old, new)


def _delta_diff(old: str, new: str, width: int) -> Text | None:
    if not shutil.which("delta"):
        return None
    unified = "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="old",
            tofile="new",
        )
    )
    try:
        proc = subprocess.run(
            ["delta", "--side-by-side", "--paging=never", f"--width={width}"],
            input=unified,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return Text.from_ansi(proc.stdout.rstrip("\n"))


def _table_diff(old: str, new: str) -> Table:
    a, b = old.splitlines(), new.splitlines()
    left: list[Text] = []
    right: list[Text] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            for line in a[i1:i2]:
                left.append(Text(line, style="dim"))
                right.append(Text(line, style="dim"))
            continue
        left.extend(Text(line, style="red") for line in a[i1:i2])
        right.extend(Text(line, style="green") for line in b[j1:j2])
        # pad the shorter side so this hunk stays row-aligned
        while len(left) < len(right):
            left.append(Text())
        while len(right) < len(left):
            right.append(Text())

    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    table.add_column(width=1)
    table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    for row_left, row_right in zip(left, right):
        table.add_row(row_left, Text("│", style="dim"), row_right)
    return table
