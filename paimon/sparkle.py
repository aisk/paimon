"""The parts of the TUI that move: the welcome sky, the masthead and the spinner.

All of it is drawn in the active theme's colours (see ``theme.palette``) and
all of it holds still when the terminal asks for no animation
(``TEXTUAL_ANIMATIONS=none``). Nothing here ticks while it is off screen, and
the welcome sky settles by itself, so an idle Paimon costs an idle terminal
nothing.
"""

import math
import random
from importlib import metadata
from pathlib import Path

from rich.cells import cell_len, set_cell_size
from textual.content import Content
from textual.widgets import Static

from .theme import gradient, palette, ramp

_LOGO = (
    "██████╗  █████╗ ██╗███╗   ███╗ ██████╗ ███╗   ██╗",
    "██╔══██╗██╔══██╗██║████╗ ████║██╔═══██╗████╗  ██║",
    "██████╔╝███████║██║██╔████╔██║██║   ██║██╔██╗ ██║",
    "██╔═══╝ ██╔══██║██║██║╚██╔╝██║██║   ██║██║╚██╗██║",
    "██║     ██║  ██║██║██║ ╚═╝ ██║╚██████╔╝██║ ╚████║",
    "╚═╝     ╚═╝  ╚═╝╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝",
)
_LOGO_WIDTH = len(_LOGO[0])

TAGLINE = "a coding agent that lives in your terminal"
_TIPS = ("Enter send", "/ commands", "@ file", "! shell", "Shift+Tab mode", "Ctrl+T new pane")

# Dimmest to brightest. All narrow, except the dot the rest of the UI already
# leans on as a separator.
_STAR_GLYPHS = "·⋆✧✦"
_TWINKLE = "·⋆✧✦✧⋆"


def _version() -> str:
    try:
        # Without the local part a development build carries after a "+".
        return "v" + metadata.version("paimon").partition("+")[0]
    except metadata.PackageNotFoundError:
        return ""


def short_path(path: Path, limit: int = 48) -> str:
    """``path`` with the home directory as ``~``, elided in the middle to fit."""
    text = str(path)
    try:
        home = str(Path.home())
    except RuntimeError:
        home = ""
    if home and (text == home or text.startswith(home + ("\\" if "\\" in home else "/"))):
        text = "~" + text[len(home):]
    if cell_len(text) <= limit:
        return text
    keep = max(limit - 1, 2)
    return text[: keep // 3] + "…" + text[-(keep - keep // 3):]


def _runs(cells: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """``cells`` with neighbours of one style joined: most of a sky is empty."""
    runs: list[tuple[str, str]] = []
    for glyph, style in cells:
        if runs and runs[-1][1] == style:
            runs[-1] = (runs[-1][0] + glyph, style)
        else:
            runs.append((glyph, style))
    return runs


class _Animated(Static):
    """A widget redrawn on a timer, only while it can be seen."""

    INTERVAL = 0.1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._frame = 0
        self._timer = None

    @property
    def _still(self) -> bool:
        """Whether the terminal asked for nothing to move."""
        return getattr(self.app, "animation_level", "full") == "none"

    def on_mount(self) -> None:
        if not self._still:
            self._timer = self.set_interval(self.INTERVAL, self._tick, pause=True)

    def on_show(self) -> None:
        self._run(True)

    def on_hide(self) -> None:
        self._run(False)

    def _run(self, running: bool) -> None:
        if self._timer is None:
            return
        if running and self._wants_ticks():
            self._timer.resume()
        else:
            self._timer.pause()

    def _wants_ticks(self) -> bool:
        return True

    def _tick(self) -> None:
        self._frame += 1
        self.refresh()


class Twinkle(_Animated):
    """One star cycling through its sizes, where a spinner would go."""

    INTERVAL = 0.14

    def render(self) -> Content:
        colors = palette(self.app)
        step = self._frame % len(_TWINKLE)
        glyph = _TWINKLE[step]
        brightness = _STAR_GLYPHS.index(glyph) / (len(_STAR_GLYPHS) - 1)
        color = colors["background"].blend(colors["primary"], 0.45 + 0.55 * brightness)
        return Content.styled(glyph, f"{color.hex} bold")


class Shimmer(_Animated):
    """A line of text with a highlight passing over it, for work under way."""

    INTERVAL = 0.08
    _BAND = 7
    # Frames between one pass and the next, on top of the text's own length.
    _REST = 14

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._text = ""

    def update(self, content="", **kwargs) -> None:
        self._text = str(content)
        super().update(content, **kwargs)

    def render(self) -> Content:
        text = self._text
        colors = palette(self.app)
        base = colors["foreground"].blend(colors["background"], 0.42)
        if self._still or not text:
            return Content.styled(text, base.hex)
        glow = colors["primary"].blend(colors["foreground"], 0.35)
        center = self._frame % (len(text) + self._BAND + self._REST) - self._BAND / 2
        parts = []
        for index, char in enumerate(text):
            near = max(0.0, 1 - abs(index - center) / self._BAND)
            parts.append((char, base.blend(glow, near).hex))
        return Content.assemble(*parts)


class Masthead(_Animated):
    """The top row: the name, a rule fading out, and where Paimon is working.

    The rule carries a glint from end to end for as long as any agent has a
    turn under way, so work in a tab that is not on screen still shows.
    """

    INTERVAL = 0.07
    NAME = "Paimon"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cwd = ""
        self._working = False

    def set_state(self, cwd: Path, working: bool) -> None:
        cwd_text = short_path(cwd)
        if (cwd_text, working) == (self._cwd, self._working):
            return
        self._cwd, self._working = cwd_text, working
        self._run(self.is_mounted and self.display)
        self.refresh()

    def _wants_ticks(self) -> bool:
        return self._working

    def render(self) -> Content:
        width = self.size.width
        colors = palette(self.app)
        gold, sky, night = colors["primary"], colors["accent"], colors["background"]
        moving = self._working and not self._still
        star = _TWINKLE[self._frame // 2 % len(_TWINKLE)] if moving else "✦"
        left = Content.styled(f"{star} ", f"{gold.hex} bold").append(
            gradient(self.NAME, [gold, colors["mid"], sky], style="bold"))
        dim = colors["foreground"].blend(night, 0.5)
        right = Content.styled(self._cwd, dim.hex)
        rule = width - left.cell_length - right.cell_length - 2
        if rule < 4:
            # No room for both ends: the name is what stays.
            return left
        glint = self._frame * 2 % (rule + 24) - 6 if moving else None
        parts = []
        for index in range(rule):
            # Fading toward the far end, where the path takes over.
            color = ramp([gold, sky, night], 0.15 + 0.75 * index / rule).blend(night, 0.35)
            glyph = "─"
            if glint is not None:
                near = max(0.0, 1 - abs(index - glint) / 6)
                if near:
                    color = color.blend(colors["foreground"], near)
                    glyph = "━" if near > 0.6 else "─"
            parts.append((glyph, color.hex))
        return Content.assemble(left, " ", *parts, " ", right)


class Hero(_Animated):
    """What an empty conversation opens on: the name in a sky of stars.

    While it is all there is in the log it takes the whole of it, the name in
    the middle: the stars twinkle, a glint crosses the letters and now and
    then a comet crosses the sky. Half a minute of that and it comes to rest;
    once something is said it folds down to a still picture at the top of the
    log and scrolls away with it.
    """

    INTERVAL = 0.125
    _REST_AFTER = 30.0
    # Rows of sky kept over the name once folded, and at the least before.
    _SKY_ABOVE = 2
    # The name, a row of sky, the tagline, where, a blank row, the tips.
    _BLOCK = len(_LOGO) + 5
    # The moment a sky at rest is drawn at.
    _REST_TIME = 1.6
    _COMET_EVERY = 9.0

    def __init__(self, cwd: Path) -> None:
        super().__init__(classes="hero")
        self._where = "  ·  ".join(part for part in (_version(), short_path(cwd)) if part)
        self._stars: tuple[tuple[int, int], list[tuple[int, int, float, float, int]]] = ((0, 0), [])
        self._resting = False
        self._folded = False

    # ---- the clock ----------------------------------------------------------

    def _wants_ticks(self) -> bool:
        return not self._resting

    def _tick(self) -> None:
        if self._frame * self.INTERVAL >= self._REST_AFTER:
            self.rest()
        else:
            super()._tick()

    def rest(self) -> None:
        """Stop moving, for good."""
        if not self._resting:
            self._resting = True
            self._run(False)
            self.refresh()

    def fold(self) -> None:
        """The conversation has started: give the log back."""
        if not self._folded:
            self._folded = True
            self.rest()
            self.refresh(layout=True)

    # ---- drawing ------------------------------------------------------------

    def _wide(self, width: int) -> bool:
        return width >= _LOGO_WIDTH + 4

    def get_content_height(self, container, viewport, width: int) -> int:
        if not self._wide(width):
            return 3
        folded = self._SKY_ABOVE + self._BLOCK
        if self._folded:
            return folded
        # All the log has, less the row every block in it keeps under itself.
        return max(folded, container.height - 1)

    def on_resize(self) -> None:
        self.refresh(layout=True)

    def _sky(self, width: int, height: int, top: int) -> list[tuple[int, int, float, float, int]]:
        """Where the stars are in a sky this size: (row, column, phase, speed, hue)."""
        if self._stars[0] != (width, height):
            rng = random.Random(width * 1000 + height)
            left = (width - _LOGO_WIDTH) // 2
            stars = []
            for row in range(height):
                if top + len(_LOGO) < row < top + self._BLOCK:
                    # The lines under the name run wider than it does.
                    continue
                # Clear of the name, with room to breathe.
                beside = top <= row <= top + len(_LOGO)
                for column in range(width):
                    if beside and left - 3 <= column < left + _LOGO_WIDTH + 3:
                        continue
                    if rng.random() < 0.06:
                        stars.append((row, column, rng.uniform(0, math.tau),
                                      rng.uniform(0.6, 2.2), rng.randrange(3)))
            self._stars = ((width, height), stars)
        return self._stars[1]

    def render(self) -> Content:
        width, height = self.size
        colors = palette(self.app)
        gold, sky, night, text = (colors["primary"], colors["accent"],
                                  colors["background"], colors["foreground"])
        dim = text.blend(night, 0.5)
        name = [gold, colors["mid"], sky]
        if not self._wide(width):
            return Content("\n").join([
                Content.styled("✦ ", f"{gold.hex} bold").append(gradient("Paimon", name, style="bold")),
                Content.styled(TAGLINE, dim.hex),
                Content.styled("  ·  ".join(_TIPS[1:4]), dim.hex),
            ])

        height = max(height, self._SKY_ABOVE + self._BLOCK)
        top = max((height - self._BLOCK) // 2, self._SKY_ABOVE)
        moving = not (self._resting or self._still)
        time = self._frame * self.INTERVAL if moving else self._REST_TIME
        cells: list[list[tuple[str, str]]] = [[(" ", "")] * width for _ in range(height)]

        hues = (gold, sky, text)
        for row, column, phase, speed, hue in self._sky(width, height, top):
            brightness = 0.5 + 0.5 * math.sin(time * speed + phase)
            glyph = _STAR_GLYPHS[min(int(brightness * len(_STAR_GLYPHS)), len(_STAR_GLYPHS) - 1)]
            cells[row][column] = (glyph, night.blend(hues[hue], 0.15 + 0.85 * brightness).hex)

        if moving:
            # A comet falling across the sky over the name, a new line each time.
            passing, since = divmod(time, self._COMET_EVERY)
            start = random.Random(passing).randrange(max(top - 1, 1))
            head = int(since * 52) - 14
            for back in range(14):
                column = head - back
                row = start + column // 14
                if 0 <= column < width and 0 <= row < top:
                    color = night.blend(text if back < 2 else gold, 1 - back / 14)
                    cells[row][column] = ("✦" if back == 0 else "─", color.hex)

        left = (width - _LOGO_WIDTH) // 2
        # A band of light crossing the letters, leaning with the rows.
        glint = (time % 6.0) / 1.5 * (_LOGO_WIDTH + 24) - 12 if moving else None
        for index, line in enumerate(_LOGO):
            for column, char in enumerate(line):
                if char == " ":
                    continue
                color = ramp(name, column / (_LOGO_WIDTH - 1)).blend(night, 0.06 * index)
                if char != "█":
                    # The letters' own shadow, drawn back into the night.
                    color = color.blend(night, 0.62)
                if glint is not None:
                    near = max(0.0, 1 - abs(column + index - glint) / 7)
                    color = color.blend(colors["shine"], 0.75 * near)
                cells[top + index][left + column] = (char, color.hex)

        tips = "  ·  ".join(_TIPS)
        if cell_len(tips) > width:
            tips = "  ·  ".join(_TIPS[:4])
        where = set_cell_size(self._where, min(cell_len(self._where), width))
        captions = {
            len(_LOGO) + 1: [("✦  ", gold.hex), (TAGLINE, f"{text.hex} italic"), ("  ✦", sky.hex)],
            len(_LOGO) + 2: [(where, dim.hex)],
            len(_LOGO) + 4: [(tips, dim.hex)],
        }
        for offset, caption in captions.items():
            column = max((width - sum(cell_len(part) for part, _ in caption)) // 2, 0)
            for part, style in caption:
                for char in part:
                    if column < width:
                        cells[top + offset][column] = (char, style)
                    column += 1
        return Content("\n").join(Content.assemble(*_runs(row)) for row in cells)
