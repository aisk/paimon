"""One subagent's pane: its turn as it happens, and nothing to type into.

A window onto an agent a conversation started, not a conversation of its own.
The agent belongs to its parent, which is who hears how it ended; this pane
shows the work and asks the confirmations it raises. It goes when the agent
does: the answer lands in the parent's conversation, and the transcript stays
in the child's session log.
"""

import asyncio

from textual.app import ComposeResult

from .agent import Agent, Job, TurnEnd
from .pane import Pane
from .transcript import EventRenderer, Transcript
from .ui import BlockingPanel, ConfirmPanel


class AgentPane(Pane):
    """A running subagent, drawn into a tab of its own."""

    def __init__(self, owner: Agent, job_id: str, job: Job, *, id: str | None = None) -> None:
        super().__init__(id=id)
        # The id the agent that started it calls it by.
        self.job_id = job_id
        self.job = job
        # The agent that started it, and the one to tell when the user stops it.
        self._owner = owner
        child = job.agent
        self.cwd = child.cwd
        self.mode = child.mode
        self.transcript = Transcript(id="log")
        self._renderer = EventRenderer(self.transcript, child)
        # One confirmation on screen at a time: a second panel mounting over
        # the first would leave the first one's answer waiting forever.
        self._panel_lock = asyncio.Lock()
        self.needs_confirm = False
        # Set by close(): the agent is cancelled and the widgets go away, so
        # nothing it unwinds through must touch the DOM.
        self._pane_closing = False
        job.sink = self._on_event
        job.on_done = self._done
        # Asked here rather than in the parent's pane: this is where what the
        # agent wants to do can be read in the context of what it has done.
        child.confirm = self._confirm

    @property
    def is_running(self) -> bool:
        return self.job.running and not self.needs_confirm

    @property
    def is_busy(self) -> bool:
        return self.job.running

    @property
    def tab_title(self) -> str:
        return f"{self.job_id} {' '.join(self.job.label.split()) or 'agent'}"

    def compose(self) -> ComposeResult:
        yield self.transcript

    def on_mount(self) -> None:
        self._focus_input()

    def _focus_input(self) -> None:
        # A waiting confirmation, or else the log, which is what makes the
        # arrow keys scroll it. Never from a pane the user is not looking at:
        # focusability ignores display, so it would take the keyboard away
        # from whoever is typing.
        if self.is_current:
            panels = self.query(BlockingPanel)
            self._request_focus(panels.last() if panels else self.transcript)

    def notice(self, renderable) -> None:
        self.transcript.add(renderable)

    def close(self) -> None:
        """Closing the tab is what stopping the agent means.

        Its parent is told, since it is still waiting for an answer. A no-op
        for an agent that is already over, which is how this pane usually goes.
        """
        if self._pane_closing:
            return
        self._pane_closing = True
        self.job.sink = None
        if self.job.running:
            self._owner.stop_job(self.job_id, by_user=True)

    def shutdown(self) -> None:
        # The parent's pane closes its agent, and the children go with it.
        self._pane_closing = True
        self.job.sink = None

    async def _on_event(self, ev) -> None:
        """Render one event of the agent's turn. The job's sink."""
        if self._pane_closing or not self.app.is_running:
            return
        await self._renderer.handle(ev)
        if isinstance(ev, TurnEnd):
            await self._renderer.close()

    def _done(self) -> None:
        """The agent is over, however it ended, and so is this tab.

        Also how the pane leaves when the model stops the agent or its parent
        closes: every one of those ends the same task.
        """
        if not self._pane_closing:
            self.app.call_later(self.app._drop_pane, self)

    async def _confirm(self, tool_name: str, args: dict) -> bool:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        panel = ConfirmPanel(tool_name, args, future, cwd=self.cwd)
        async with self._panel_lock:
            # Removal below is asynchronous, so the previous panel may still
            # be mounted; under the lock it cannot be a live one.
            await self.query(BlockingPanel).remove()
            await self.mount(panel)
            # A panel in a background pane must not grab the keyboard: the
            # user's next keystroke would answer a question they never saw.
            self._request_focus(panel)
            self.needs_confirm = True
            self._notify_state()
            try:
                return await future == "allow"
            finally:
                self.needs_confirm = False
                focused = self.app.focused if self.app.screen_stack else None
                restore = focused is panel or (focused is not None and panel in focused.ancestors)
                if restore and not self._pane_closing:
                    self._request_focus(self.transcript)
                await panel.remove()
                if not self._pane_closing:
                    self._notify_state()
