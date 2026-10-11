"""Tell the terminal what this process is doing, with OSC 7501.

The Program Status Protocol lets a program say through its own pty that it is
idle, working, waiting on the user, finished or failed, so a terminal or an
agent inbox built on one does not have to guess from the screen. A terminal
that does not know the sequence ignores it, so nothing is asked first.

Only the root record is reported: one terminal, one state, the way Herdr is
told. https://www.superlogical.com/rex/docs/build/program-status
"""

import base64
import unicodedata

# The stable machine name the terminal groups reports by.
APP = "paimon"

IDLE = "idle"
WORKING = "working"
DONE = "done"
BLOCKED = "blocked"
ERROR = "error"

PERMISSION = "permission"
QUESTION = "question"

# The protocol's limit on a decoded message. A report over it is discarded
# whole, state included, so the message is cut here instead.
_MAX_MSG = 2048


def _line(text: str) -> str:
    """``text`` as the one line of plain text a message may be."""
    text = "".join(" " if unicodedata.category(ch) == "Cc" else ch for ch in text)
    text = " ".join(text.split())
    # A lone surrogate, which a tool argument can carry, has no encoding.
    return text.encode("utf-8", errors="replace")[:_MAX_MSG].decode("utf-8", errors="ignore")


def encode(state: str, *, kind: str = "", msg: str = "") -> str:
    """The sequence that sets the root record.

    A report replaces its record completely, so each one carries everything.
    """
    pairs = [f"state={state}"]
    if kind:
        pairs.append(f"kind={kind}")
    pairs.append(f"app={APP}")
    if line := _line(msg):
        pairs.append("msg=" + base64.b64encode(line.encode("utf-8")).decode("ascii"))
    return f"\x1b]7501;{':'.join(pairs)}\x1b\\"


# Removes every record. Sent on the way out: a finished or failed record
# otherwise outlives the process that reported it.
CLEAR = "\x1b]7501;state=clear\x1b\\"
