"""The agents and background commands one agent starts, and what became of them.

An ``Agent`` owns one ``Jobs``: the table of what it started, by the id the
model quotes back, the notices waiting to be shown to the model, the caps, and
the job tools that act on them. The turn loop stays in the agent; this module
only starts, watches and stops.

The dependency runs one way. The agent hands in what opening a child takes, as
callables, and reads back the notices; nothing here reaches into the agent.
"""

import asyncio
import contextlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable
from uuid import uuid4

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

from .errors import PaimonError
from .events import AgentEvent, UserInput
from .tools.background import BackgroundCommand, start_background, tail_text
from .tools.result import ToolError

if TYPE_CHECKING:
    from .agent import Agent

# Agents and running background commands one agent may hold at a time: each
# holds a model context or a process. A bound on how wide the work fans out,
# not on how long it goes on, since a finished agent frees its slot.
MAX_JOBS = 8

# Of those, how many may be agents. Tighter than the cap they share with
# commands: every agent running is another model context being paid for at the
# same time, and a model left to fan out freely burns through a quota fast.
MAX_AGENTS = 4

# Chars of a child agent's answer delivered to its parent. The answer lands in
# the parent's context without the parent having asked for it, so it is kept
# to what a report needs; the child's session log has the rest.
ANSWER_LIMIT = 8_000


class PaneLimitError(PaimonError):
    """Every pane is taken, so nothing more can be put on screen.

    Raised by an ``open_job`` hook that has no room left. Defined here rather
    than with the UI that raises it, so the job manager can tell "no room to
    show it" from any other failure without importing the UI.
    """


@dataclass
class Job:
    """An agent or a background command another agent started."""

    kind: str  # "agent" or "command"
    label: str = ""
    agent: "Agent | None" = None
    command: BackgroundCommand | None = None
    # What ends with the job: the child's one turn, or the wait for the
    # command to exit. Cancelling it is how a job is stopped without a notice.
    # None only until the job has been put on screen, which comes first.
    task: asyncio.Task | None = None
    # How much of the command's output read_job has already handed over.
    cursor: int = 0
    # Set by whoever shows an agent. The sink is awaited with every event of
    # its turn, so drawing them is what paces it; on_done is called once when
    # its task is over, however it ended.
    sink: Callable[[AgentEvent], Awaitable[None]] | None = None
    on_done: Callable[[], None] | None = None

    @property
    def running(self) -> bool:
        if self.command is not None:
            return self.command.running and not self.command.killed
        return self.task is None or not self.task.done()


def _answer(history: list[ModelMessage], session_id: str) -> str:
    """A child's final answer, clipped: its last assistant text and nothing else.

    Never its thinking or its tool output: an agent exists to keep those out
    of the caller's context, and handing them back would spend the tokens the
    split was meant to save.
    """
    text = ""
    for message in reversed(history):
        if isinstance(message, ModelResponse):
            text = "\n\n".join(part.content.strip() for part in message.parts
                               if isinstance(part, TextPart) and part.content.strip())
            if text:
                break
    if not text:
        return "(it ended without an answer)"
    if len(text) > ANSWER_LIMIT:
        text = (f"{text[:ANSWER_LIMIT]}\n... (clipped, {len(text) - ANSWER_LIMIT} more chars; "
                f"`paimon log {session_id[:8]} --tail 1 --full` prints all of it)")
    return text


# Opens a child of the owning agent, on the model named or, for None, on the
# owner's own: a new Agent with its session open and its lock taken, ready for
# its one turn. The owner decides everything else about it (cwd, mode, config,
# the narrowed toolset, the parent session, the confirm fallback).
OpenChild = Callable[[str | None], "Agent"]


class Jobs(Mapping[str, Job]):
    """The agents and background commands one agent started.

    Reads as a mapping from the id the model quotes back to the ``Job``; only
    the manager itself adds or removes one.
    """

    def __init__(self, cwd: Path, *, open_child: OpenChild,
                 check_model: Callable[[str], object]) -> None:
        # Where background commands run.
        self._cwd = cwd
        self._open_child = open_child
        # Raises when a model named for a child cannot be built: an unknown
        # provider, no credential.
        self._check_model = check_model
        # The agents and background commands this agent started, by the id the
        # model quotes back. An agent leaves when it ends; a command stays, so
        # what it printed can still be read.
        self._table: dict[str, Job] = {}
        # What became of the jobs, waiting to be shown to the model: taken
        # at the top of a turn and between its steps, like queued user input.
        self.notices: list[str] = []
        # Set by the UI. Called whenever a job starts or ends or a notice
        # arrives, so it can redraw and wake an idle conversation.
        self.on_changed: Callable[[], None] | None = None
        # Set by the UI: puts a job on screen before it gets under way,
        # awaited with (job id, job). None where there is no tab to show it
        # in: an agent then runs unseen, and run_background refuses.
        self.open_job: Callable[[str, Job], Awaitable[None]] | None = None
        # The tool budget of the owner's turn in flight, which the agents it
        # starts inherit: an unattended run bounded for one agent is bounded
        # for all. Set by the owner as each turn starts.
        self.tool_budget: int | None = None
        self._closed = False
        # Events run_until_idle waits on, set whenever the jobs move.
        self._waiters: set[asyncio.Event] = set()

    # ---- the mapping --------------------------------------------------------

    def __getitem__(self, job_id: str) -> Job:
        return self._table[job_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._table)

    def __len__(self) -> int:
        return len(self._table)

    def __repr__(self) -> str:
        return f"Jobs({self._table!r})"

    # ---- what the owner asks of it ------------------------------------------

    def close(self) -> None:
        """Stop every job and drop every notice; nothing is reported after."""
        self._closed = True
        for job in self._table.values():
            if not job.task.done():
                job.task.cancel()
            if job.command is not None:
                job.command.kill()
        self._table.clear()
        self.notices.clear()

    def set_mode(self, mode: str) -> None:
        # The agents this one started follow it: tightening the mode while a
        # child is at work must not leave the child running under the old one.
        for job in self._table.values():
            if job.agent is not None:
                job.agent.mode = mode

    def take_notices(self) -> list[str]:
        """The waiting notices, taken: each one is handed over once."""
        notices, self.notices = self.notices, []
        return notices

    @property
    def agents_running(self) -> bool:
        """Whether any agent this one started has yet to report."""
        return any(job.kind == "agent" for job in self._table.values())

    @contextlib.contextmanager
    def watch(self) -> Iterator[asyncio.Event]:
        """An event set whenever the jobs move, for as long as the block runs."""
        moved = asyncio.Event()
        self._waiters.add(moved)
        try:
            yield moved
        finally:
            self._waiters.discard(moved)

    # ---- the job tools ------------------------------------------------------

    async def call(self, name: str, args: dict) -> str:
        """Run one job tool call and render its result for the model.

        Fails the call by raising ToolError, the way a registry tool does.
        """
        if name == "spawn_agent":
            prompt = str(args.get("prompt") or "").strip()
            if not prompt:
                raise ToolError("Error: prompt is required.")
            self._check_room("agent")
            model = str(args.get("model") or "").strip() or None
            if model:
                # Built here only to be thrown away: the child would not find
                # out before its first request, a turn too late to pick again.
                try:
                    self._check_model(model)
                except Exception as exc:  # noqa: BLE001 — unknown provider, no credential
                    raise ToolError(f"Error: cannot start an agent on {model!r}: {exc} "
                                    "Call list_models for the ones available.") from None
            try:
                job_id = await self._spawn(prompt, model)
            except Exception as exc:  # noqa: BLE001 — a session or a tab that would not open
                raise ToolError(f"Error: could not start an agent: {exc}") from None
            return (f"Started agent {job_id}; it is running now. Its answer is delivered "
                    "to you when it finishes, so carry on or end your turn.")

        if name == "run_background":
            command = str(args.get("command") or "").strip()
            if not command:
                raise ToolError("Error: command is required.")
            if self.open_job is None:
                raise ToolError("Error: background commands only work in the interactive UI; "
                                "use shell instead.")
            self._check_room("command")
            try:
                job_id = await self._start_command(
                    command, str(args.get("description") or "").strip())
            except Exception as exc:  # noqa: BLE001 — a command or a tab that would not start
                raise ToolError(f"Error: could not start the command: {exc}") from None
            return (f"Started background command {job_id}; it keeps running after this "
                    "turn and you are told when it exits. Read its output with read_job "
                    "and stop it with stop_job.")

        job_id = str(args.get("job_id") or "").strip()
        job = self._table.get(job_id)
        if job is None:
            raise ToolError(f"Error: no agent or background command {job_id}. Agents leave "
                            "when they finish, and ids from an earlier run are gone.")

        if name == "read_job":
            command = job.command
            if command is None:
                raise ToolError(f"Error: {job_id} is an agent; its answer is delivered to "
                                "you when it finishes.")
            start = 0 if args.get("mode") == "all" else job.cursor
            data, job.cursor, dropped = command.output.since(start)
            if command.killed:
                state = "stopped"
            elif command.running:
                state = "running"
            else:
                state = f"exited, code {command.exit_code}"
            text = tail_text(data, dropped)
            if not text.strip():
                return f"{job_id} [{state}]: no output{' since your last read' if start else ''}."
            return f"{job_id} [{state}]:\n{text}"

        if name == "stop_job":
            if not self.stop(job_id):
                raise ToolError(f"Error: nothing to stop under {job_id}; it has already ended.")
            if job.kind == "agent":
                return f"Stopped agent {job_id}. It reports nothing back."
            return f"Stopped {job_id}. What it printed is still readable with read_job."

        raise ToolError(f"Error: unknown tool {name!r}")

    def _check_room(self, kind: str) -> None:
        """Fail a job tool call whose job would exceed a cap."""
        live = [job for job in self._table.values() if job.running]
        agents = sum(1 for job in live if job.kind == "agent")
        if kind == "agent" and agents >= MAX_AGENTS:
            raise ToolError(f"Error: {agents} agents are already running (limit {MAX_AGENTS}); "
                            "wait for one to finish or stop one before starting another.")
        if len(live) >= MAX_JOBS:
            raise ToolError(f"Error: {len(live)} agents and commands are already running "
                            f"(limit {MAX_JOBS}); stop one before starting another.")

    def _new_job_id(self) -> str:
        """An id no job of this agent is using.

        One space for agents and commands both: they are quoted back to the
        model side by side, and an id that named one of each would be read as
        whichever the model happened to expect.
        """
        while True:
            job_id = uuid4().hex[:4]
            if job_id not in self._table:
                return job_id

    def _notify(self, notice: str | None = None) -> None:
        """Queue ``notice`` for the model, and tell the UI the jobs moved."""
        if self._closed:
            return
        if notice:
            self.notices.append(notice)
        for waiter in self._waiters:
            waiter.set()
        if self.on_changed is not None:
            self.on_changed()

    # ---- agents -------------------------------------------------------------

    async def _spawn(self, prompt: str, model: str | None) -> str:
        """Start a child agent on ``prompt`` as a task of its own; returns its id."""
        job_id = self._new_job_id()
        child = self._open_child(model)
        job = Job("agent", label=prompt, agent=child)
        # On screen before its first step, so nothing it does goes unseen.
        if self.open_job is not None:
            try:
                await self.open_job(job_id, job)
            except PaneLimitError:
                # Every tab is taken. That limits what the screen can show,
                # not what the agent can do: the child runs unseen, as it does
                # where there are no tabs at all, and its confirmations go
                # through the fallback its owner opened it with.
                pass
            except BaseException:
                child.close()
                raise
        job.task = asyncio.ensure_future(self._run_child(job, prompt))
        self._table[job_id] = job
        # A callback rather than a finally in the task: one cancelled before
        # its first step never enters the coroutine, and its session lock
        # would be held for the life of the process.
        job.task.add_done_callback(lambda _: self._child_done(job_id, job))
        self._notify()
        return job_id

    async def _run_child(self, job: Job, prompt: str) -> str:
        """Drive a child through its one turn; returns what it answered."""
        child = job.agent

        async def emit(event: AgentEvent) -> None:
            # Shown where somebody is showing the child, dropped otherwise:
            # the answer is all the parent gets either way.
            if job.sink is not None:
                await job.sink(event)

        # The prompt is an event like any other, as in a turn the user starts.
        await emit(UserInput(prompt))
        # Closed on the way out even when stopped mid-event, so the child's
        # turn records how it ended before _child_done closes its session.
        async with contextlib.aclosing(child.run(prompt, max_tool_calls=self.tool_budget)) as events:
            async for event in events:
                await emit(event)
        return _answer(child.history, child.session.id)

    def _child_done(self, job_id: str, job: Job) -> None:
        """A child's task is over, however it ended: release it and report."""
        job.agent.close()
        if job.on_done is not None:
            job.on_done()
        if self._table.get(job_id) is job:
            del self._table[job_id]
        notice = None
        # Cancelled is how a child is stopped, and whoever stopped it has
        # already said what there is to say.
        if not job.task.cancelled():
            exc = job.task.exception()
            if exc is None:
                notice = f"agent {job_id} finished:\n{job.task.result()}"
            else:
                notice = f"agent {job_id} failed: {type(exc).__name__}: {exc}"
        self._notify(notice)

    # ---- commands -----------------------------------------------------------

    async def _start_command(self, command: str, description: str) -> str:
        """Start a background command and put it on screen; returns its id."""
        job_id = self._new_job_id()
        running = await start_background(command, self._cwd)
        job = Job("command", label=description or command, command=running)
        try:
            await self.open_job(job_id, job)
        except BaseException:
            # Nothing would ever kill it: no tab holds it and no job names it,
            # and it is in its own process group, so it would outlive the app.
            running.kill()
            raise
        job.task = asyncio.ensure_future(self._watch_command(job_id, running))
        self._table[job_id] = job
        self._notify()
        return job_id

    async def _watch_command(self, job_id: str, command: BackgroundCommand) -> None:
        """Tell the model when a command ends by itself, or its tab was closed."""
        code = await command.wait()
        self._notify(f"command {job_id} was stopped" if command.killed
                     else f"command {job_id} exited (code {code})")

    # ---- stopping -----------------------------------------------------------

    def stop(self, job_id: str, *, by_user: bool = False) -> bool:
        """End one job. False when the id names nothing still running.

        The model is not told about a stop it ordered itself, since the tool
        result already says so; one the user ordered is news to it.
        """
        job = self._table.get(job_id)
        if job is None or not job.running:
            return False
        if by_user:
            self.notices.append(f"{job.kind} {job_id} was stopped by the user")
        job.task.cancel()
        if job.command is not None:
            job.command.kill()
            self._notify()
        return True
