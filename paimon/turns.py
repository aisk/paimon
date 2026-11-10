"""The driver of one conversation: an agent, and everything that writes its history.

The driver owns the coroutine that runs its agent's turns and pulls work
from its own inbox, so delivery and completion are facts about an
``asyncio.Task`` rather than about whether a widget got around to posting
something.

Everything that changes the agent's history goes through it: turns, wake-up
turns, manual compactions and the "!" runs the user records. That is what
keeps a compaction from replacing a history a turn is still appending to.

Deliberately free of Textual, so delivery, cancellation and cleanup can be
tested against a plain sink instead of a driven terminal.
"""

import asyncio
import contextlib
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .events import UserInput
from .compaction import CompactionResult
from .session import shell_message

# Inbox sentinel for a turn with no user input: the agent runs just to react
# to what its jobs did. Distinct from None, which tells the driver to stop.
WAKE = object()

# Inbox sentinel for a compaction the user asked for.
COMPACT = object()

# What a wake-up turn leaves instead of a TurnOver when it found nothing to
# react to: the agent never ran, so there is no turn to report.
_IDLE = object()


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


@dataclass(frozen=True)
class CompactionStarted:
    """A compaction the user asked for has started. Its first event."""


@dataclass(frozen=True)
class CompactionOver:
    """A compaction the user asked for is over, however it ended.

    ``compacted`` is the agent's CompactionResult when it succeeded, and None
    when it succeeded with nothing to compact or did not succeed at all.
    """

    result: Result
    compacted: CompactionResult | None = None


class TurnDriver:
    """One agent, and the turns it is asked to run.

    Turns are serialized by the driver rather than by a lock: it takes one
    item off the inbox at a time — a prompt, a wake-up or a compaction — so
    no two of them can ever share the agent's history and overwrite each
    other's writes.

    The driver also decides when the agent runs on its own: it wakes it when
    one of its jobs reports while nothing else is under way, and after a turn
    that finished with a report still waiting.
    """

    def __init__(self, agent) -> None:
        self.agent = agent
        # The outcome of the last turn. A compaction leaves it alone: what the
        # conversation last said is what a turn said.
        self.result: Result | None = None
        self.killed = False
        # Called, with no arguments, whenever the state changes.
        self.on_change = None
        # Called, with no arguments, whenever the agent's jobs move, after the
        # driver has woken the agent if that was due. For whoever shows them.
        self.on_jobs_changed: Callable[[], None] | None = None
        # Returns the prompts the user typed while a turn was running, taking
        # them: the agent's loop asks between steps, so they join the turn
        # already under way. None where nobody can type meanwhile.
        self.take_queued: Callable[[], list[str]] | None = None
        self._task: asyncio.Task | None = None
        # Called with each event of the running item, awaited. The pane's
        # renderer; None while nobody is showing this agent, which then runs
        # unwatched rather than having to stop.
        self.sink = None
        # Depth of pending permission confirmations. Counted rather than a
        # flag: the tab badge has to clear the moment the answer is in, and
        # removing the panel that asked is asynchronous.
        self.blocked = 0
        self._inbox: asyncio.Queue = asyncio.Queue()
        # The task of the item in flight, a turn or a compaction.
        self._item: asyncio.Task | None = None
        # The event that closes the item in flight, set by the item itself.
        self._over = None
        # "!" runs that finished while an item was in flight, as (command,
        # output): a turn takes them at its next step, and whatever is left
        # when the item ends is recorded before anything else runs.
        self._held: list[tuple[str, str]] = []
        agent.pending = self._take_pending
        agent.jobs.on_changed = self._jobs_changed

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
            item = await self._inbox.get()
            if item is None:
                return
            self._over = None
            work = self._compact() if item is COMPACT else self._run_turn(item)
            self._item = asyncio.ensure_future(work)
            self._notify()
            # Waited on rather than awaited: interrupting an item cancels that
            # task, and awaiting a cancelled task raises into the driver too,
            # which would end the agent instead of the item.
            await asyncio.wait({self._item})
            if self._over is None:
                # Cancelled before it took its first step.
                if item is COMPACT:
                    self._over = CompactionOver(Result(Outcome.INTERRUPTED))
                else:
                    self.result = Result(Outcome.INTERRUPTED)
                    self._over = TurnOver(self.result)
            try:
                await self._close(None if self._over is _IDLE else self._over)
            except Exception as exc:  # noqa: BLE001 — the driver outlives any one item
                # _close contains what it calls into; this is the backstop for
                # a listener that raises, which must not end the driver and
                # leave every later prompt waiting on an inbox nobody reads.
                self._item = None
                self.result = Result(Outcome.FAILED, error=_describe(exc))

    async def _close(self, over) -> None:
        """The item in flight is over: settle what it left behind, report it,
        and wake the agent if a report is waiting for it.

        ``over`` is None for a wake-up that found nothing to do, which reports
        nothing: it must neither overwrite the last turn's result nor set off
        what the end of a turn does on screen.

        Nothing here may end the driver. What goes wrong is reported as a
        failed outcome, the one thing every viewer of a driver reads; the
        item itself has already left its record in the session log.
        """
        self._item = None
        # Before the event, whose handler may submit the next prompt: a "!"
        # run from during this item belongs to the history it happened in.
        if (problem := self._record_held()) is not None:
            over = self._failed(over, problem)
        if over is None:
            self._release_queued()
        else:
            try:
                await self._emit(over)
            except Exception as exc:  # noqa: BLE001 — a renderer bug must not end the driver
                over = self._failed(over, f"its end could not be shown ({_describe(exc)})")
                if isinstance(over, TurnOver):
                    # Once more, as a failure: whatever the sink choked on,
                    # this gives it a chance to close the turn and hand back
                    # what was queued behind it. If it fails again, the
                    # result is left to say so.
                    with contextlib.suppress(Exception):
                        await self._emit(over)
        # A job that ended during the last step has not been reported: the
        # loop only looks between steps. Refused when a queued prompt has just
        # been submitted, which reports it anyway. Not after a turn the user
        # stopped or that failed, nor a compaction the user stopped: the
        # notice waits for whatever runs next.
        if over is None:
            wakes = True
        elif isinstance(over, CompactionOver):
            wakes = over.result.outcome is not Outcome.INTERRUPTED
        else:
            wakes = over.result.finished
        if wakes and self.agent.jobs.notices:
            self.wake()
        self._notify()

    def _record_held(self) -> str | None:
        """Record the "!" runs held while the item was in flight: None, or
        what went wrong.

        A run that cannot be recorded stays held, with the ones after it, so
        it is not lost: the next turn takes it at its first step, and the end
        of the next item tries again.
        """
        runs, self._held = self._held, []
        for index, (command, output) in enumerate(runs):
            try:
                self.agent.record_shell(command, output)
            except Exception as exc:  # noqa: BLE001 — reported through the outcome
                self._held = runs[index:] + self._held
                return f'a "!" run could not be recorded yet ({_describe(exc)})'
        return None

    def _failed(self, over, problem: str):
        """``over`` with ``problem`` reported, and the result saying it.

        A turn's own report becomes a failure, so the pane shows it where it
        shows any other. A compaction's stays as it went, which it did; only
        the result says what went wrong after it. None (an idle wake-up)
        stays None.
        """
        error = problem
        if isinstance(over, TurnOver) and over.result.outcome is Outcome.FAILED:
            error = f"{over.result.error}; {problem}"
        self.result = Result(Outcome.FAILED, error=error)
        return TurnOver(self.result) if isinstance(over, TurnOver) else over

    def _release_queued(self) -> None:
        """Send what was typed while an idle wake-up sat in the inbox.

        A turn hands the queue back through its TurnOver; an idle wake-up has
        none, so the queue is taken here and sent the way the pane sends it
        after a finished turn: as one prompt.
        """
        texts = self.take_queued() if self.take_queued is not None else []
        if texts and not self.killed:
            self.submit("\n\n".join(texts))

    async def _emit(self, event) -> None:
        if self.sink is not None:
            await self.sink(event)

    async def _run_turn(self, prompt) -> None:
        user_input = None if prompt is WAKE else prompt
        # What stays when the turn is cancelled.
        result = Result(Outcome.INTERRUPTED)
        idle = False
        try:
            if user_input is not None:
                # The prompt is an event like any other, so every turn
                # renders through the one sink. A wake-up has no prompt to
                # show: its first event is a job notice, or nothing at all.
                await self._emit(UserInput(user_input))
            shown = False
            # Closed here even when the turn is cancelled while the sink has
            # an event: left to the garbage collector, the turn would hold the
            # agent past its end and the next one would be refused.
            async with contextlib.aclosing(self.agent.run(user_input)) as events:
                async for event in events:
                    shown = True
                    await self._emit(event)
            # A wake-up whose news was gone by the time it ran yields nothing
            # and persists nothing: the turn never happened.
            idle = user_input is None and not shown
            result = Result(Outcome.SUCCESS)
        except Exception as exc:  # noqa: BLE001 — reported, not raised at the UI
            result = Result(Outcome.FAILED, error=str(exc))
        finally:
            if idle:
                self._over = _IDLE
            else:
                self.result = result
                self._over = TurnOver(result)

    async def _compact(self) -> None:
        compacted = None
        # What stays when it is cancelled, which Agent.compact_now guarantees
        # leaves the history and the log as they were.
        result = Result(Outcome.INTERRUPTED)
        try:
            await self._emit(CompactionStarted())
            compacted = await self.agent.compact_now()
            result = Result(Outcome.SUCCESS)
        except Exception as exc:  # noqa: BLE001 — the session is still usable
            result = Result(Outcome.FAILED, error=str(exc))
        finally:
            self._over = CompactionOver(result, compacted)

    def submit(self, text: str) -> None:
        self._inbox.put_nowait(text)
        self._notify()

    def wake(self) -> bool:
        """Queue a turn with no user input, so the agent reacts to what its
        jobs did (their notices are the only new input). Refused while busy
        or killed: a busy agent already reports at its next step, or once the
        item in flight is over, and one wake-up in the inbox is enough —
        is_busy covers a queued one too.
        """
        if self.killed or self.is_busy:
            return False
        self._inbox.put_nowait(WAKE)
        self._notify()
        return True

    def compact(self) -> bool:
        """Queue a compaction of the agent's context. Refused while busy or
        killed: compacting under a turn would replace the history it is
        appending to, so it is asked for between turns."""
        if self.killed or self.is_busy:
            return False
        self._inbox.put_nowait(COMPACT)
        self._notify()
        return True

    def record_shell(self, command: str, output: str) -> None:
        """Let the agent in on a "!" run the user made.

        Recorded at once when nothing is in flight. Otherwise held: a turn
        takes it at its next step, since a message inserted between a tool
        call and its result is the one shape providers reject, and a
        compaction would summarize the history it was appended to away. What
        is still held when the item ends is recorded then.

        Synchronous on purpose: nothing can start between the check and the
        append.
        """
        if self.killed:
            return
        if self._item is None:
            self.agent.record_shell(command, output)
        else:
            self._held.append((command, output))

    def interrupt(self) -> None:
        item = self._item
        if item is not None and not item.done():
            item.cancel()

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
        self._held.clear()
        if self._task is not None:
            self._task.cancel()
        self._notify()

    def shutdown(self) -> None:
        """The process is going: whatever the driver is in the middle of,
        there will be no loop left to finish it.

        Killed like cancel(), so nothing on the way out calls back into a
        pane that is going away, but without telling anyone: there is nobody
        left to tell.
        """
        self.killed = True
        self.interrupt()
        if self._task is not None:
            self._task.cancel()

    # ---- the agent's hooks --------------------------------------------------

    def _take_pending(self) -> list:
        """What the running turn takes at its next step: held "!" runs first,
        since they happened before anything still queued was typed, then the
        queued prompts."""
        runs, self._held = self._held, []
        texts = self.take_queued() if self.take_queued is not None else []
        return [shell_message(*run) for run in runs] + texts

    def _jobs_changed(self) -> None:
        """One of the agent's jobs started, ended or reported.

        A report that arrives while nothing is under way starts a turn for
        it; one that arrives mid-turn is picked up between steps, and one
        that arrives during a compaction once it is over. wake() refuses both.
        """
        if self.agent.jobs.notices:
            self.wake()
        if self.on_jobs_changed is not None:
            self.on_jobs_changed()

    # ---- observation --------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        """Whether an item is in flight, or one is queued behind it."""
        return self._item is not None or not self._inbox.empty()

    @property
    def is_running(self) -> bool:
        """Whether an item is in flight and not stuck on a confirmation."""
        return self.is_busy and self.blocked == 0

    def mark_blocked(self, blocked: bool) -> None:
        """Report that a confirmation is (or is no longer) waiting on the user."""
        self.blocked += 1 if blocked else -1
        self._notify()

    def _notify(self) -> None:
        if self.on_change is not None:
            self.on_change()


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
