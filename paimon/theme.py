"""Paimon's own themes, and the colours the animated widgets are drawn in.

The stylesheet only ever names theme variables, so every Textual theme still
works. These two are what the app looks like until one is picked: a night sky
with gold stars in it, and the same palette by daylight.

The widgets in ``sparkle`` paint gradients, which a stylesheet cannot, so they
ask here for the active theme's colours as values and blend them themselves.
"""

from collections.abc import Sequence

from textual.app import App
from textual.color import Color, ColorParseError
from textual.content import Content
from textual.theme import Theme

DEFAULT_THEME = "paimon"

THEMES = (
    Theme(
        name="paimon",
        primary="#f3c969",
        secondary="#8f9df8",
        accent="#7fd6ff",
        foreground="#e8eaf6",
        background="#0b1020",
        surface="#131a33",
        panel="#1b2444",
        success="#86e3a8",
        warning="#ffb86b",
        error="#ff7a90",
        dark=True,
        variables={
            "block-cursor-background": "#f3c969",
            "block-cursor-foreground": "#0b1020",
            "input-selection-background": "#8f9df8 35%",
            "border": "#f3c969",
            "border-blurred": "#2a3560",
        },
    ),
    Theme(
        name="paimon-light",
        primary="#a8720b",
        secondary="#5b6bd6",
        accent="#177fb3",
        foreground="#272c47",
        background="#f8f5ec",
        surface="#f0ebdd",
        panel="#e6dfcc",
        success="#2c8f56",
        warning="#b96a12",
        error="#cf3f58",
        dark=False,
        variables={
            "block-cursor-background": "#a8720b",
            "block-cursor-foreground": "#f8f5ec",
            "input-selection-background": "#5b6bd6 25%",
            "border": "#a8720b",
            "border-blurred": "#cfc6ae",
        },
    ),
)

# What each permission mode is drawn in, wherever a mode is named: cool for
# the one that only reads, warm for the one that checks nothing.
MODE_COLORS = {"read": "accent", "auto": "success", "yolo": "warning"}

# Used when a theme's own value is not a colour that can be blended.
_FALLBACK = {
    "primary": "#f3c969",
    "secondary": "#8f9df8",
    "accent": "#7fd6ff",
    "foreground": "#e8eaf6",
    "background": "#0b1020",
    "surface": "#131a33",
    "success": "#86e3a8",
    "warning": "#ffb86b",
    "error": "#ff7a90",
}

_palettes: dict[str, dict[str, Color]] = {}


def palette(app: App) -> dict[str, Color]:
    """The active theme's base colours, as values to blend, and two made of them.

    Kept per theme name: resolving the variables builds a whole colour system,
    which is too much to do on every frame of an animation.
    """
    name = app.theme
    cached = _palettes.get(name)
    if cached is None:
        variables = app.get_css_variables()
        cached = {}
        for key, fallback in _FALLBACK.items():
            try:
                cached[key] = Color.parse(variables.get(key, fallback))
            except ColorParseError:
                cached[key] = Color.parse(fallback)
        # What the name passes through between gold and sky, and what a glint
        # is made of: white on a night sky, where on paper the text colour
        # would be mud in the first and a shadow in the second.
        dark = app.current_theme.dark
        cached["mid"] = cached["foreground" if dark else "secondary"]
        cached["shine"] = cached["foreground" if dark else "background"]
        _palettes[name] = cached
    return cached


def ramp(colors: Sequence[Color], position: float) -> Color:
    """The colour ``position`` (0..1) of the way along ``colors``."""
    if len(colors) == 1:
        return colors[0]
    position = min(max(position, 0.0), 1.0) * (len(colors) - 1)
    index = min(int(position), len(colors) - 2)
    return colors[index].blend(colors[index + 1], position - index)


def gradient(text: str, colors: Sequence[Color], *, style: str = "") -> Content:
    """``text`` coloured along ``colors`` from its first cell to its last."""
    span = max(len(text) - 1, 1)
    return Content.assemble(*(
        (char, f"{ramp(colors, index / span).hex} {style}".strip())
        for index, char in enumerate(text)
    ))


def gauge(fraction: float, width: int = 8) -> tuple[str, str]:
    """A bar ``width`` cells long: the part filled to ``fraction``, and the rest.

    Two strings for the caller to draw in two colours, as one weight of rule
    reads as a single track where two glyphs would read as two things.
    """
    filled = round(min(max(fraction, 0.0), 1.0) * width)
    return "━" * filled, "━" * (width - filled)
