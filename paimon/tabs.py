"""The strip listing the open panes, docked at the bottom of the screen.

Bottom rather than top: terminals and multiplexers put their own tab bars
along the top edge, and two rows of tabs stacked on each other are more
confusing than either alone. The bottom also sits next to the prompt, which is
where the eye already is when panes are switched.

Written by hand rather than with ``Tabs``/``TabbedContent``: that widget brings
an underline and a set of bindings this strip does not want, and it prefixes
every tab ID, which a mix of session and background-task panes would have to
undo.

Each tab draws its own frame instead of leaning on a CSS border: the line
separating the strip from the conversation has to run unbroken across the whole
strip and meet the current tab's frame in a junction glyph, and per-widget
borders never join up like that. So the box drawing lives here and the
stylesheet only supplies colours and sizes.
"""

from rich.cells import cell_len, set_cell_size
from textual.containers import Container
from textual.content import Content
from textual.message import Message
from textual.widgets import Static

# Textual has no spinner widget (LoadingIndicator fills its whole area), so a
# running tab cycles one narrow glyph on the strip's own interval.
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_INTERVAL = 0.12

# All narrow, so labels stay aligned on terminals that render ambiguous-width
# glyphs double-wide. Idle is a space rather than a bullet: the marker keeps a
# fixed slot so a tab never changes width when a turn starts or ends, and the
# frame already does the separating a bullet used to do.
_IDLE = " "
_ATTENTION = "!"

# A tab's interior, in cells. Tabs share the strip evenly between these bounds:
# the maximum keeps two panes from each taking half the terminal, the minimum
# (index plus marker, no title) is what lets all eight still fit.
_TAB_MAX = 30
_TAB_MIN = 6


def _fit(text: str, width: int) -> str:
    """`text` in exactly `width` cells, padded with spaces or elided with '…'."""
    if cell_len(text) > width:
        return set_cell_size(text, max(0, width - 1)) + "…"
    return set_cell_size(text, width)


def _cell(label: str, marker: str, width: int) -> str:
    """One tab's interior: ' <label> <marker> ', exactly `width` cells wide."""
    return f" {_fit(label, max(0, width - 4))} {marker} "


class PaneTab(Static):
    """One pane's entry in the strip."""

    def __init__(self, pane) -> None:
        super().__init__(classes="pane-tab", id=f"tab-{pane.id}")
        self.pane = pane

    def on_click(self) -> None:
        self.post_message(PaneTabs.Selected(self.pane))

    def draw(self, label: str, marker: str, active: bool, width: int) -> None:
        """Render this tab: the rule on top, the frame opening downwards."""
        cell = _cell(label, marker, width)
        if active:
            self.update(Content.from_markup(
                "[$primary]┬$bar┬[/]\n[$primary]│[/]$cell[$primary]│[/]\n[$primary]╰$bar╯[/]",
                bar="─" * width, cell=cell))
            return
        # The rule carries no markup style: $text-muted is an "auto" colour and
        # only resolves against the background when the stylesheet applies it.
        rule = "─" * (width + 2)
        self.update(Content(f"{rule}\n {cell} \n{' ' * (width + 2)}"))


class _StripFill(Static):
    """Carries the separator past the last tab, to the end of the strip."""

    def __init__(self) -> None:
        super().__init__(id="pane-tabs-fill")

    def on_resize(self) -> None:
        self.refresh()

    def render(self) -> Content:
        width, height = self.size.width, self.size.height
        if width <= 0 or height <= 0:
            return Content("")
        return Content(f"{'─' * width}\n\n")


class PaneTabs(Container):
    """Lists the panes and reports clicks on them.

    It renders state, it does not own it: the app calls ``sync`` whenever a
    pane is added, removed or changes state, and the strip re-reads the panes.
    """

    class Selected(Message):
        """A tab was clicked."""

        def __init__(self, pane) -> None:
            self.pane = pane
            super().__init__()

    def __init__(self) -> None:
        super().__init__(id="pane-tabs")
        self._panes: list = []
        self._current = None
        self._frame = 0
        self._timer = None
        self._fill = _StripFill()

    def compose(self):
        yield self._fill

    def on_mount(self) -> None:
        self._timer = self.set_interval(_SPINNER_INTERVAL, self._tick, pause=True)

    def sync(self, panes: list, current) -> None:
        """Mirror the pane list, then redraw every label.

        The strip is hidden while a single pane is open: one tab says nothing
        and would cost three rows of the conversation.
        """
        self.display = len(panes) > 1
        keep = set(panes)
        for tab in self.query(PaneTab):
            if tab.pane not in keep:
                tab.remove()
        known = {tab.pane for tab in self.query(PaneTab)}
        # Panes are only ever appended, so mounting in order keeps the strip
        # in the app's order without any reshuffling. The fill stays last.
        for pane in panes:
            if pane not in known:
                if self._fill.is_mounted:
                    self.mount(PaneTab(pane), before=self._fill)
                else:
                    self.mount(PaneTab(pane))
        self._panes = list(panes)
        self._current = current
        self._redraw()
        if self._timer is not None:
            if any(pane.is_running for pane in panes):
                self._timer.resume()
            else:
                self._timer.pause()

    def _tick(self) -> None:
        self._frame += 1
        self._redraw()

    def on_resize(self) -> None:
        self._redraw()

    def _tab_width(self) -> int:
        """How wide each tab gets: an even share of the strip, bounded.

        Tabs are clipped rather than scrolled, and the pane the user just
        opened is the last one, so tabs that refuse to shrink would hide the
        very tab that matters.
        """
        available = self.container_size.width
        if not available or not self._panes:
            return _TAB_MAX
        # Two of every tab's columns go to its frame, not to the interior.
        share = available // len(self._panes) - 2
        return max(_TAB_MIN, min(_TAB_MAX, share))

    def _redraw(self) -> None:
        width = self._tab_width()
        # Iterate the mounted tabs, not the pane list: mounting is
        # asynchronous, so a pane added this frame has no tab yet.
        for tab in self.query(PaneTab):
            pane = tab.pane
            if pane not in self._panes:
                continue
            index = self._panes.index(pane) + 1
            active = pane is self._current
            tab.set_class(active, "-active")
            tab.set_class(pane.needs_confirm, "-attention")
            tab.draw(f"{index} {pane.tab_title}", self._marker(pane), active, width)

    def _marker(self, pane) -> str:
        if pane.needs_confirm:
            return _ATTENTION
        if pane.is_running:
            return _SPINNER[self._frame % len(_SPINNER)]
        return _IDLE
