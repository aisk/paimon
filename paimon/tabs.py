"""The strip listing the open panes, docked at the top, left or right.

Written by hand rather than with ``Tabs``/``TabbedContent``: that widget's
underline and its left/right bindings are horizontal by construction, and it
prefixes every tab ID, which a mix of session and background-task panes would
have to undo. The strip here is a plain container of ``PaneTab`` widgets whose
orientation is a CSS class.
"""

from textual.containers import Container
from textual.content import Content
from textual.message import Message
from textual.widgets import Static

# Where the strip can sit. The first is the default.
DOCKS = ("top", "left", "right")

# Textual has no spinner widget (LoadingIndicator fills its whole area), so a
# running tab cycles one narrow glyph on the strip's own interval.
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_INTERVAL = 0.12

# All narrow, so labels stay aligned on terminals that render ambiguous-width
# glyphs double-wide.
_IDLE = "·"
_ATTENTION = "!"


class PaneTab(Static):
    """One pane's entry in the strip."""

    def __init__(self, pane) -> None:
        super().__init__(classes="pane-tab", id=f"tab-{pane.id}")
        self.pane = pane

    def on_click(self) -> None:
        self.post_message(PaneTabs.Selected(self.pane))


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

    def __init__(self, dock: str = DOCKS[0]) -> None:
        super().__init__(id="pane-tabs")
        self._panes: list = []
        self._current = None
        self._frame = 0
        self._timer = None
        self.set_dock(dock)

    def on_mount(self) -> None:
        self._timer = self.set_interval(_SPINNER_INTERVAL, self._tick, pause=True)

    def set_dock(self, dock: str) -> None:
        """Move the strip. Orientation and size are what the class selects."""
        if dock not in DOCKS:
            dock = DOCKS[0]
        self.set_classes([f"-{dock}"])

    def sync(self, panes: list, current) -> None:
        """Mirror the pane list, then redraw every label.

        The strip is hidden while a single pane is open: one tab says nothing
        and would cost a row (or a column) of the conversation.
        """
        self.display = len(panes) > 1
        keep = set(panes)
        for tab in self.query(PaneTab):
            if tab.pane not in keep:
                tab.remove()
        known = {tab.pane for tab in self.query(PaneTab)}
        # Panes are only ever appended, so mounting in order keeps the strip
        # in the app's order without any reshuffling.
        for pane in panes:
            if pane not in known:
                self.mount(PaneTab(pane))
        self._panes = list(panes)
        self._current = current
        self._redraw()
        if self._timer is not None:
            if any(pane.is_turn_running for pane in panes):
                self._timer.resume()
            else:
                self._timer.pause()

    def _tick(self) -> None:
        self._frame += 1
        self._redraw()

    def _redraw(self) -> None:
        # Iterate the mounted tabs, not the pane list: mounting is
        # asynchronous, so a pane added this frame has no tab yet.
        for tab in self.query(PaneTab):
            pane = tab.pane
            if pane not in self._panes:
                continue
            index = self._panes.index(pane) + 1
            tab.set_class(pane is self._current, "-active")
            tab.set_class(pane.needs_confirm, "-attention")
            tab.update(Content.from_markup(
                "$marker $index. $title",
                marker=self._marker(pane), index=str(index), title=pane.tab_title))

    def _marker(self, pane) -> str:
        if pane.needs_confirm:
            return _ATTENTION
        if pane.is_turn_running:
            return _SPINNER[self._frame % len(_SPINNER)]
        return _IDLE
