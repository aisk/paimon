"""Pictures in the terminal, through Kitty's graphics protocol.

Of the protocols terminals have for images, this is the one a full-screen app
can use without fighting its own renderer. The picture is sent once, out of
band, under a number; where it goes is then ordinary text, a placeholder
character per cell whose foreground colour is that number. Those cells scroll,
clip and get redrawn like any others, and the terminal paints the picture over
whichever of them are on screen.

Only PNG is sent: it is the one encoded format the protocol takes, and
anything else would have to be decoded here first.
"""

from __future__ import annotations

import base64
import itertools
import math
import os
import struct
import sys
from typing import Mapping

# The most cells a picture is given. A preview, not a viewer: enough to tell
# which screenshot it was.
MAX_COLUMNS = 40
MAX_ROWS = 12

# What a cell is taken to measure when the terminal does not say.
_CELL = (8, 16)

_PLACEHOLDER = "\U0010eeee"
# The marks that number a placeholder's row, from the protocol's table: row n
# of a picture is the placeholder followed by the nth of these. Cells after
# the first in a row need none, each being one column on from its neighbour.
_ROW_MARKS = (
    "̅̍̎̐̒̽̾̿͆͊"
    "͋͌͐͑͒͗͛ͣͤͥ"
    "ͦͧͨͩͪͫͬͭͮͯ"
)
assert MAX_ROWS <= len(_ROW_MARKS)

# The payload of one escape sequence, the most the protocol allows.
_CHUNK = 4096

# Numbers are per terminal window, where another program may have pictures of
# its own, so they start somewhere this process is unlikely to share.
_ids = itertools.count(os.getpid() << 8)


def supported(environ: Mapping[str, str]) -> bool:
    """Whether the terminal this runs in draws placeholder pictures.

    Told from the environment, since asking the terminal would mean reading
    its answer out of the keyboard's stream. Kitty and Ghostty are the ones
    that do; others that speak the protocol place pictures at the cursor only.
    A multiplexer in between passes none of it on unless it is set up to.
    """
    if environ.get("TMUX") or environ.get("ZELLIJ"):
        return False
    term = environ.get("TERM", "").lower()
    if term.startswith(("tmux", "screen")):
        return False
    program = environ.get("TERM_PROGRAM", "").lower()
    return bool(environ.get("KITTY_WINDOW_ID") or "kitty" in term or program == "kitty"
                or environ.get("GHOSTTY_RESOURCES_DIR") or "ghostty" in term or program == "ghostty")


def new_id() -> int:
    """A number to send a picture under. It has to fit a 24-bit colour, and
    zero is no picture at all."""
    return next(_ids) % 0xFFFFFF + 1


def colour(image_id: int) -> tuple[int, int, int]:
    """The foreground colour that marks a cell as part of picture ``image_id``."""
    return image_id >> 16 & 0xFF, image_id >> 8 & 0xFF, image_id & 0xFF


def png_size(data: bytes) -> tuple[int, int] | None:
    """A PNG's width and height in pixels, or None for anything else."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or data[12:16] != b"IHDR" or len(data) < 24:
        return None
    width, height = struct.unpack(">II", data[16:24])
    return (width, height) if width and height else None


def cell_size() -> tuple[int, int]:
    """A character cell's width and height in pixels."""
    try:
        import fcntl
        import termios

        rows, columns, width, height = struct.unpack(
            "HHHH", fcntl.ioctl(sys.__stdout__.fileno(), termios.TIOCGWINSZ, b"\0" * 8))
    except (ImportError, OSError, AttributeError, ValueError):
        return _CELL
    if not (rows and columns and width and height):
        return _CELL
    return max(1, width // columns), max(1, height // rows)


def fit(size: tuple[int, int], cell: tuple[int, int],
        columns: int = MAX_COLUMNS, rows: int = MAX_ROWS) -> tuple[int, int]:
    """The columns and rows a picture of ``size`` pixels takes: its own size
    when that is small enough, and otherwise the largest box of its shape
    within ``columns`` by ``rows``."""
    width, height = size
    scale = min(1.0, columns * cell[0] / width, rows * cell[1] / height)
    return (max(1, min(columns, math.ceil(width * scale / cell[0]))),
            max(1, min(rows, math.ceil(height * scale / cell[1]))))


def transmit(image_id: int, png: bytes, columns: int, rows: int) -> str:
    """The sequence that hands the terminal a picture to draw wherever its
    placeholders are, scaled into ``columns`` by ``rows`` cells."""
    payload = base64.standard_b64encode(png).decode("ascii")
    chunks = [payload[start:start + _CHUNK] for start in range(0, len(payload), _CHUNK)] or [""]
    # q=2: no reply, which would arrive as keystrokes.
    keys = f"a=T,U=1,f=100,i={image_id},c={columns},r={rows},q=2,"
    return "".join(
        f"\x1b_G{keys if index == 0 else ''}m={int(index < len(chunks) - 1)};{chunk}\x1b\\"
        for index, chunk in enumerate(chunks))


def delete(image_id: int) -> str:
    """The sequence that has the terminal forget a picture."""
    return f"\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\"


def placeholder(row: int, columns: int) -> str:
    """The text of one row of a picture ``columns`` wide."""
    return _PLACEHOLDER + _ROW_MARKS[row] + _PLACEHOLDER * (columns - 1)
