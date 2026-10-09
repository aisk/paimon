"""The driver of one conversation: an agent, and the turns it is asked to run.

The driver owns the coroutine that runs its agent's turns and pulls prompts
from its own inbox, so delivery and completion are facts about an
``asyncio.Task`` rather than about whether a widget got around to posting
something.

Deliberately free of Textual, so delivery, cancellation and cleanup can be
tested against a plain sink instead of a driven terminal.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum

from .agent import UserInput

# Inbox sentinel for a turn with no user input: the agent runs just to react
# to what its jobs did. Distinct from None, which tells the driver to stop.
WAKE = object()


class Outcome(str, Enum):
    """How one turn ended."""

    SUCCESS = "success"
    # Cancelled on purpose. Not a failure, but not a finished turn either: the
    # caller must not treat what follows as an answer.
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass(frozen=True)
class Result:
    """The outcome of the last turn, and why."""

    outcome: Outcome
    error: str = ""

    @property
    def finished(self) -> bool:
        """Whether the turn ran to its own end, rather than being stopped."""
        return self.outcome is Outcome.SUCCESS


@dataclass(frozen=True)
class TurnOver:
    """One turn is over, however it ended. Always the last event of a turn.

    Emitted by the driver rather than by the turn itself: a turn that was
    interrupted is a cancelled task, and nothing awaited from inside its
    ``finally`` would survive to reach a renderer.
    """

    result: Result


class TurnDriver:
    """One agent, and the turns it is asked to run.

    Turns are serialized by the driver rather than by a lock: it takes one
    prompt off the inbox at a time, so two ``Agent.run`` generators can never
    share the agent's history and overwrite each other's tool results.
    """

    def __init__(self, agent) -> None:
        self.agent = agent
        self.result: Result | None = None
        self.killed = False
        # Called, with no arguments, whenever the state changes.
        self.on_change = None
        self._task: asyncio.Task | None = None
        # Called with each event of the running turn, awaited. The pane's
        # renderer; None while nobody is showing this agent, which then runs
        # unwatched rather than having to stop.
        self.sink = None
        # Depth of pending permission confirmations. Counted rather than a
        # flag: the tab badge has to clear the moment the answer is in, and
        # removing the panel that asked is asynchronous.
        self.blocked = 0
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._turn: asyncio.Task | None = None

    # ---- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Begin driving. Called once, from inside the running loop."""
        self._task = asyncio.ensure_future(self._run())

    async def _run(self) -> None:
        try:
            await self._drive()
        finally:
            # Not when it was cancelled: cancel() has already said so, and the
            # unwinding happens late enough that on the way out there may be
            # nothing left to tell.
            if not self.killed:
                self._notify()

    async def _drive(self) -> None:
        while True:
            prompt = await self._inbox.get()
            if prompt is None:
                return
            self._turn = asyncio.ensure_future(self._run_turn(prompt))
            self._notify()
            # Waited on rather than awaited: interrupting a turn cancels that
            # task, and awaiting a cancelled task raises into the driver too,
            # which would end the agent instead of the turn.
            await asyncio.wait({self._turn})
            self._turn = None
            await self._emit(TurnOver(self.result or Result(Outcome.INTERRUPTED)))
            self._notify()

    async def _emit(self, event) -> None:
        if self.sink is not None:
            await self.sink(event)

    async def _run_turn(self, prompt) -> None:
        try:
            user_input = None if prompt is WAKE else prompt
            if user_input is not None:
                # The prompt is an event like any other, so every turn
                # renders through the one sink. A wake-up has no prompt to
                # show: its first event is a job notice, or nothing at all.
                await self._emit(UserInput(user_input))
            async for event in self.agent.run(user_input):
                await self._emit(event)
            self.result = Result(Outcome.SUCCESS)
        except asyncio.CancelledError:
            self.result = Result(Outcome.INTERRUPTED)
            raise
        except Exception as exc:  # noqa: BLE001 — reported, not raised at the UI
            self.result = Result(Outcome.FAILED, error=str(exc))

    def submit(self, text: str) -> bool:
        self._inbox.put_nowait(text)
        self._notify()
        return True

    def wake(self) -> bool:
        """Queue a turn with no user input, so the agent reacts to what its
        jobs did (their notices are the only new input). Refused while busy
        or killed: a busy agent already reports at its next step, and one
        wake-up in the inbox is enough — is_busy covers a queued one too.
        """
        if self.killed or self.is_busy:
            return False
        self._inbox.put_nowait(WAKE)
        self._notify()
        return True

    def interrupt(self) -> None:
        turn = self._turn
        if turn is not None and not turn.done():
            turn.cancel()

    def cancel(self) -> None:
        """Stop driving for good. Idempotent: several paths lead here."""
        if self.killed:
            return
        self.killed = True
        self.interrupt()
        # Anything still queued was accepted on the understanding that this
        # agent would get to it, which is no longer true.
        while not self._inbox.empty():
            self._inbox.get_nowait()
        if self._task is not None:
            self._task.cancel()
        self._notify()

    def shutdown(self) -> None:
        """The process is going: whatever the driver is in the middle of,
        there will be no loop left to finish it."""
        self.interrupt()
        if self._task is not None:
            self._task.cancel()

    # ---- observation --------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        return self._turn is not None or not self._inbox.empty()

    @property
    def is_running(self) -> bool:
        """Whether a turn is in flight and not stuck on a confirmation."""
        return self.is_busy and self.blocked == 0

    def mark_blocked(self, blocked: bool) -> None:
        """Report that a confirmation is (or is no longer) waiting on the user."""
        self.blocked += 1 if blocked else -1
        self._notify()

    def _notify(self) -> None:
        if self.on_change is not None:
            self.on_change()
