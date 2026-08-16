"""Side-by-side diff rendering for tool confirmations and the transcript.

Pure difflib + Pygments (bundled with rich): SequenceMatcher aligns the two
sides, rich.syntax colors the code with the pygments style closest to the app
theme, and a second character-level pass emphasizes what changed inside
edited line pairs. No external tools involved.
"""

import difflib
from pathlib import Path
from typing import Optional

from rich.padding import Padding
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Span, Text

# textual theme name -> pygments style, for the themes with a close
# counterpart; anything else falls back to a stock dark or light style.
_SYNTAX_THEMES = {
    "textual-light": "default",
    "nord": "nord",
    "gruvbox": "gruvbox-dark",
    "dracula": "dracula",
    "monokai": "monokai",
    "solarized-dark": "solarized-dark",
    "solarized-light": "solarized-light",
    "atom-one-dark": "one-dark",
    "ansi-dark": "ansi_dark",
    "ansi-light": "ansi_light",
}

# (whole line, emphasized region) backgrounds per darkness, GitHub-flavored
_REMOVED_BG = {True: ("#3a1d1f", "#6f2b30"), False: ("#ffebe9", "#ffc1bc")}
_ADDED_BG = {True: ("#173a24", "#2b6f42"), False: ("#dafbe1", "#aceebb")}

# Below this SequenceMatcher ratio two paired lines are a rewrite rather than
# an edit, and character-level emphasis would just paint both wall to wall.
_REFINE_THRESHOLD = 0.5


def render_diff(old: str, new: str, *, path: str = "",
                start_line: Optional[int] = None,
                theme: str = "", dark: bool = True) -> Table:
    """``start_line`` is where the diffed region starts in the file; None
    means unknown, which drops the line numbers rather than showing wrong
    ones. ``theme`` is the textual theme name and ``dark`` its darkness,
    steering the syntax colors toward the app's."""
    a, b = old.splitlines(), new.splitlines()
    style = _SYNTAX_THEMES.get(theme, "monokai" if dark else "default")
    a_text = _highlight(old, path, style)
    b_text = _highlight(new, path, style)
    del_bg, del_emph = _REMOVED_BG[dark]
    add_bg, add_emph = _ADDED_BG[dark]
    start = start_line or 1

    left: list[tuple[Text, Text, str]] = []  # number, content, background
    right: list[tuple[Text, Text, str]] = []

    def num(n: int) -> Text:
        return Text(str(n), style="dim")

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                left.append((num(start + i1 + k), a_text[i1 + k], ""))
                right.append((num(start + j1 + k), b_text[j1 + k], ""))
            continue
        if tag == "replace":
            for x, y in zip(range(i1, i2), range(j1, j2)):
                _refine(a[x], b[y], a_text[x], b_text[y], del_emph, add_emph)
        left.extend((num(start + i), a_text[i], del_bg) for i in range(i1, i2))
        right.extend((num(start + j), b_text[j], add_bg) for j in range(j1, j2))
        # pad the shorter side so this hunk stays row-aligned
        while len(left) < len(right):
            left.append((Text(), Text(), ""))
        while len(right) < len(left):
            right.append((Text(), Text(), ""))

    table = Table.grid(padding=(0, 1), expand=True)
    numbered = start_line is not None
    if numbered:
        num_width = len(str(start + max(len(a), len(b), 1) - 1))
        table.add_column(width=num_width, justify="right")
    table.add_column(ratio=1)
    table.add_column(width=1)
    if numbered:
        table.add_column(width=num_width, justify="right")
    table.add_column(ratio=1)
    for (ln, lt, lbg), (rn, rt, rbg) in zip(left, right):
        cells = [_cell(lt, lbg), Text("│", style="dim"), _cell(rt, rbg)]
        if numbered:
            cells = [ln, cells[0], cells[1], rn, cells[2]]
        table.add_row(*cells)
    return table


def locate_line(path: str, old: str, new: str,
                cwd: Optional[Path] = None) -> Optional[int]:
    """1-based line where the edited region starts in the file, else None.

    ``old`` is searched first; when the edit already ran (session replay) the
    file holds ``new`` instead, at the same starting line since everything
    before the region is untouched.
    """
    p = Path(path)
    if cwd is not None and not p.is_absolute():
        p = cwd / p
    try:
        text = p.read_text(errors="replace")
    except OSError:
        return None
    for needle in (old, new):
        idx = text.find(needle) if needle else -1
        if idx != -1:
            return text.count("\n", 0, idx) + 1
    return None


def _highlight(code: str, path: str, theme: str) -> list[Text]:
    """Per-line syntax-highlighted text, plain when no lexer matches.

    Token backgrounds are stripped so the pygments style's page color does
    not paint over the app background or the added/removed line tints.
    """
    src = code.splitlines()
    lexer = Syntax.guess_lexer(path, code)
    highlighted = Syntax(code, lexer, theme=theme).highlight(code)
    # the page color hides in two places: the base style of the returned Text
    # and the bgcolor baked into every token span
    highlighted.style = ""
    highlighted.spans = [
        Span(s.start, s.end, _foreground(s.style)) for s in highlighted.spans
    ]
    lines = list(highlighted.split("\n"))[: len(src)]
    while len(lines) < len(src):
        lines.append(Text(src[len(lines)]))
    for line in lines:
        line.no_wrap = True
        line.overflow = "ellipsis"
    return lines


def _foreground(style: Style | str) -> Style | str:
    if isinstance(style, str):
        return style
    return Style(color=style.color, bold=style.bold, italic=style.italic,
                 underline=style.underline)


def _refine(a_line: str, b_line: str, a_text: Text, b_text: Text,
            del_emph: str, add_emph: str) -> None:
    """Emphasize the characters that differ within a paired changed line."""
    matcher = difflib.SequenceMatcher(None, a_line, b_line)
    if matcher.ratio() < _REFINE_THRESHOLD:
        return
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        a_text.stylize(f"on {del_emph}", i1, i2)
        b_text.stylize(f"on {add_emph}", j1, j2)


def _cell(text: Text, bg: str):
    """A cell whose background, when set, fills the full column width."""
    return Padding(text, 0, style=f"on {bg}") if bg else text
