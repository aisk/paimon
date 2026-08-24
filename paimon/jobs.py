"""One Job per thing the app is running: an agent, or a background command.

A job owns the coroutine that drives it. That is the whole point of the class:
before it, the supervisor asked a pane whether its Textual worker happened to
be running, and could only hand an agent its next instruction once a UI message
had made the round trip. Here the driver pulls from its own inbox, so delivery
and completion are facts about an ``asyncio.Task`` rather than about whether a
widget got around to posting something.

The two kinds share their lifecycle (start, interrupt one unit of work, cancel
the lot, wait for it to stop, report a state) and share nothing about storage:
an agent's durable record is its ``history``, already persisted and already the
one source of truth for replay, while a command's is the byte buffer nobody
else keeps. ``read`` hides which of the two a cursor points into.

Deliberately free of Textual, so delivery, cancellation and cleanup can be
tested against a plain sink instead of a driven terminal.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

from .agent import UserInput
from .tools import tail_text

# Inbox sentinel for a turn with no user input: the agent runs just to react
# to what its jobs did. Distinct from None, which tells the driver to stop.
WAKE = object()


class State(str, Enum):
    """What a job is doing, as reported back to the model that asked."""

    RUNNING = "running"
    # An agent between turns: done with the last one, ready for another.
    IDLE = "idle"
    # An agent blocked on a permission prompt in its own tab.
    NEEDS_CONFIRM = "needs_confirm"
    # A command that exited by itself, having succeeded.
    DONE = "done"
    # A turn that raised, or a command that exited non-zero.
    FAILED = "failed"
    KILLED = "killed"
    # An id that names no job: a stale one from a resumed session, or one
    # belonging to somebody else. Returned rather than raised — the model is
    # certain to try it, and an error would end its turn over nothing.
    UNKNOWN = "unknown"


class Outcome(str, Enum):
    """How one unit of work ended: one turn, or the command's whole run."""

    SUCCESS = "success"
    # Cancelled on purpose. Not a failure, but not a finished turn either: the
    # caller must not treat what follows as an answer.
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass(frozen=True)
class Result:
    """The outcome of the last unit of work, and why."""

    outcome: Outcome
    error: str = ""
    exit_code: Optional[int] = None

    @property
    def finished(self) -> bool:
        """Whether work ran to its own end, rather than being stopped."""
        return self.outcome is Outcome.SUCCESS


@dataclass(frozen=True)
class TurnOver:
    """One turn is over, however it ended. Always the last event of a turn.

    Emitted by the driver rather than by the turn itself: a turn that was
    interrupted is a cancelled task, and nothing awaited from inside its
    ``finally`` would survive to reach a renderer.
    """

    result: Result


@dataclass(frozen=True)
class View:
    """What one caller sees of a job's output."""

    job_id: str
    state: State
    text: str
    exit_code: Optional[int] = None
    # False when only the part since the caller's last read is included.
    complete: bool = False


def _blocks(messages: list[ModelMessage]) -> list[str]:
    """The assistant text of each message, one entry per message.

    Entries stay aligned with message indexes, empty ones included, so a read
    cursor means the same thing before and after a job is retired and its
    messages are thrown away.
    """
    rendered = []
    for message in messages:
        parts = []
        if isinstance(message, ModelResponse):
            parts = [part.content.strip() for part in message.parts
                     if isinstance(part, TextPart) and part.content.strip()]
        rendered.append("\n\n".join(parts))
    return rendered


def _join(blocks: list[str]) -> str:
    return "\n\n".join(block for block in blocks if block)


def render_text(messages: list[ModelMessage]) -> str:
    """The assistant text of these messages, and nothing else.

    Never the raw messages: an agent exists to keep its thinking and its tool
    output out of the caller's context, and handing those back would spend the
    tokens the split was meant to save.
    """
    return _join(_blocks(messages))


class Job:
    """Something the app is running, and the coroutine that runs it."""

    kind = ""

    def __init__(self, job_id: str, *, parent=None) -> None:
        self.job_id = job_id
        # The Agent that started this. None for a pane the user opened: it
        # belongs to nobody, so no agent can read or stop it.
        self.parent = parent
        self.result: Optional[Result] = None
        self.killed = False
        # The state last reported to the parent; see Supervisor.status_summary.
        self.reported = State.RUNNING
        # Each called with this job whenever its state changes: the pane that
        # shows it, and whoever else needs to react (the supervisor's wake-up
        # check). A list so no listener has to know about the others.
        self.on_change: list = []
        self._task: Optional[asyncio.Task] = None
        # Per caller, how much of the output it has already read. Keyed by the
        # caller object itself, so the same job read twice gets only what is
        # new and a cursor can never be confused with somebody else's.
        self._cursors: dict = {}
        # Created on first use: an asyncio primitive binds to the loop that
        # first blocks on it, and a job may be built before that loop exists.
        self._changed: Optional[asyncio.Event] = None

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
        raise NotImplementedError

    def submit(self, text: str) -> bool:
        """Give the job more work. False when this kind takes none."""
        return False

    def interrupt(self) -> None:
        """Stop the unit of work in flight. The job itself stays."""

    def cancel(self) -> None:
        """End the job. Idempotent: several paths lead here for one job.

        What it produced stays readable — whoever started it may still be about
        to ask — so this marks and releases rather than forgetting.
        """
        if self.killed:
            return
        self.killed = True
        self._retire()
        if self._task is not None:
            self._task.cancel()
        self._notify()

    def _retire(self) -> None:
        """Give up what this job holds, while it can still be read off."""

    def shutdown(self) -> None:
        """The process is going. Release what the OS would not release for us.

        Subclasses extend and then call this: whatever the driver is in the
        middle of, there will be no loop left to finish it.
        """
        if self._task is not None:
            self._task.cancel()

    # ---- observation --------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        """Whether work is in flight or already accepted for this job."""
        return False

    @property
    def state(self) -> State:
        return State.KILLED if self.killed else self._state()

    def _state(self) -> State:
        raise NotImplementedError

    @property
    def exit_code(self) -> Optional[int]:
        """The command's exit status, or None for anything that has no such thing."""
        return None

    def read(self, caller, mode: str = "new") -> View:
        raise NotImplementedError

    async def wait(self, timeout: float) -> State:
        """Block until the job stops running, or the timeout runs out.

        Returns RUNNING when it is still going: a caller that cannot get its
        turn back is a deadlock, and the job may be waiting on a permission
        prompt in a tab the user is not looking at.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            state = self.state
            if state is not State.RUNNING:
                return state
            remaining = deadline - loop.time()
            if remaining <= 0:
                return State.RUNNING
            if self._changed is None:
                self._changed = asyncio.Event()
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), remaining)
            except asyncio.TimeoutError:
                return self.state

    def _notify(self) -> None:
        if self._changed is not None:
            self._changed.set()
        for listener in list(self.on_change):
            listener(self)


class AgentJob(Job):
    """One agent, and the turns it is asked to run.

    Turns are serialized by the driver rather than by a lock: it takes one
    prompt off the inbox at a time, so two ``Agent.run`` generators can never
    share the agent's history and overwrite each other's tool results.
    """

    kind = "agent"

    def __init__(self, job_id: str, agent, *, parent=None) -> None:
        super().__init__(job_id, parent=parent)
        self.agent = agent
        # Called with each event of the running turn, awaited. The pane's
        # renderer; None while nobody is showing this agent, which is a job
        # running unwatched rather than a job that has to stop.
        self.sink = None
        # Depth of pending permission confirmations. Counted rather than a
        # flag: the tab badge has to clear the moment the answer is in, and
        # removing the panel that asked is asynchronous.
        self.blocked = 0
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._turn: Optional[asyncio.Task] = None
        # The assistant text of each message, kept when the agent goes. Text
        # and not the messages: what a caller can read back is the rendered
        # text either way, and a killed agent otherwise holds its whole
        # history for as long as the app runs.
        self._final: Optional[list[str]] = None

    # ---- lifecycle ----------------------------------------------------------

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
                # The prompt is an event like any other, so a turn the
                # supervisor started renders in its pane the same way one the
                # user typed does. A wake-up has no prompt to show: its first
                # event is the agent's own status line, or nothing at all.
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
        jobs did (the status line is the only new input). Refused while busy
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

    def _retire(self) -> None:
        # Snapshotted before the turn is cancelled, past which nothing a
        # caller was waiting to read can arrive.
        self._final = _blocks(self.agent.history)
        self.interrupt()
        # Anything still queued was accepted on the understanding that this
        # agent would get to it, which is no longer true.
        while not self._inbox.empty():
            self._inbox.get_nowait()

    def shutdown(self) -> None:
        self.interrupt()
        super().shutdown()

    # ---- observation --------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        return self._turn is not None or not self._inbox.empty()

    def _state(self) -> State:
        if self.blocked > 0:
            return State.NEEDS_CONFIRM
        if self.is_busy:
            return State.RUNNING
        if self.result is not None and self.result.outcome is Outcome.FAILED:
            return State.FAILED
        return State.IDLE

    def mark_blocked(self, blocked: bool) -> None:
        """Report that this agent is (or is no longer) awaiting a confirmation."""
        self.blocked += 1 if blocked else -1
        self._notify()

    def read(self, caller, mode: str = "new") -> View:
        blocks = self._final if self._final is not None else _blocks(self.agent.history)
        start = 0 if mode == "all" else min(self._cursors.get(caller, 0), len(blocks))
        self._cursors[caller] = len(blocks)
        return View(self.job_id, self.state, _join(blocks[start:]), complete=start == 0)


class CommandJob(Job):
    """One background command: a process, and the bytes it has written."""

    kind = "command"

    def __init__(self, job_id: str, command, description: str, *, parent=None) -> None:
        super().__init__(job_id, parent=parent)
        self.command = command  # a tools.BackgroundCommand
        self.description = description

    # ---- lifecycle ----------------------------------------------------------

    async def _drive(self) -> None:
        code = await self.command.wait()
        self.result = Result(Outcome.SUCCESS if code == 0 else Outcome.FAILED, exit_code=code)

    def _retire(self) -> None:
        # The buffer belongs to the command object, which this job keeps
        # holding, so a late read still works after the process is gone.
        self.command.kill()

    def shutdown(self) -> None:
        self.command.terminate_now()
        super().shutdown()

    # ---- observation --------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        return self.command.running

    @property
    def exit_code(self) -> Optional[int]:
        # Read off the command rather than off the result, which the driver
        # only writes once it has been round the loop again.
        return self.command.exit_code

    def _state(self) -> State:
        if self.command.killed:
            return State.KILLED
        if self.command.running:
            return State.RUNNING
        return State.DONE if self.command.exit_code == 0 else State.FAILED

    @property
    def status_text(self) -> str:
        """How this command is doing, for the tab and the status bar."""
        if self.command.killed:
            return "stopped"
        if self.command.running:
            return "running"
        return f"exited (code {self.command.exit_code})"

    def read(self, caller, mode: str = "new") -> View:
        start = 0 if mode == "all" else self._cursors.get(caller, 0)
        data, cursor, dropped = self.command.output.since(start)
        self._cursors[caller] = cursor
        return View(self.job_id, self.state, tail_text(data, dropped),
                    exit_code=self.command.exit_code, complete=start == 0)
