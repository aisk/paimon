"""The pool of agents and background tasks this process is running.

One ``Supervisor`` per app. It owns the agent registry, each agent's inbox and
each caller's read cursor, the table of background commands, and it is the only
thing that decides when a queued prompt becomes a turn. Deliberately free of
Textual: what runs an agent is a ``runner`` (in the UI, a ``SessionPane``), and
everything here — delivery, concurrency, cleanup — can be tested against a
plain object instead of a driven terminal.

A runner is anything with:

``agent``            the ``Agent`` it drives (for its history and its session)
``is_busy``          whether a turn is running or about to start
``needs_confirm``    whether it is blocked on a permission confirmation
``turn_failed``      whether the last finished turn ended in an error
``start_turn(text)`` begin a turn; only ever called when ``is_busy`` is False

The supervisor never calls ``start_turn`` on a busy runner: two ``Agent.run``
generators on one agent would share its history and overwrite each other's
tool results in the log.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

from .tools import DEFAULT_WAIT_TIMEOUT, MAX_WAIT_TIMEOUT, start_background, tail_text


class AgentState(str, Enum):
    """What an agent is doing, as reported back to the model that asked."""

    RUNNING = "running"
    IDLE = "idle"
    NEEDS_CONFIRM = "needs_confirm"
    FAILED = "failed"
    KILLED = "killed"
    # An id that names no agent: a stale one from a resumed session, or one
    # belonging to somebody else. Returned rather than raised — the model is
    # certain to try it, and an error would end its turn over nothing.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AgentView:
    """What one caller sees of an agent's output."""

    agent_id: str
    state: AgentState
    text: str
    # False when only the part since the caller's last read is included.
    complete: bool = False


class TaskState(str, Enum):
    """What a background command is doing.

    Its own enum rather than more of AgentState: a task has no history, no
    turn and nothing to confirm, but it does have an exit code, and the states
    the two share ("running") are the only ones they share.
    """

    RUNNING = "running"
    EXITED = "exited"
    KILLED = "killed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TaskView:
    """What one caller sees of a background task."""

    task_id: str
    state: TaskState
    text: str
    exit_code: Optional[int] = None
    # False when only the part since the caller's last read is included.
    complete: bool = False


@dataclass(frozen=True)
class Delivery:
    """The outcome of sending a prompt to an agent."""

    agent_id: str
    state: AgentState
    accepted: bool = True
    # The agent was busy, so the prompt waits in its inbox for the next turn.
    queued: bool = False


class SupervisorError(RuntimeError):
    """An agent could not be started. Reported to the model, not raised at it."""


# What a state change is called in the status line injected into the parent's
# next turn. RUNNING is absent on purpose: "still running" is not news.
_NEWS = {
    AgentState.IDLE: "finished",
    AgentState.NEEDS_CONFIRM: "needs confirmation",
    AgentState.FAILED: "failed",
    AgentState.KILLED: "killed",
}

# The same for tasks. RUNNING is absent for the same reason.
_TASK_NEWS = {
    TaskState.EXITED: "exited",
    TaskState.KILLED: "was stopped",
}


def render_text(messages: list[ModelMessage]) -> str:
    """The assistant text of these messages, and nothing else.

    Never the raw messages: an agent exists to keep its thinking and its tool
    output out of the caller's context, and handing those back would spend the
    tokens the split was meant to save.
    """
    blocks = []
    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, TextPart) and part.content.strip():
                    blocks.append(part.content.strip())
    return "\n\n".join(blocks)


@dataclass
class _Record:
    agent_id: str
    parent: object
    runner: object
    prompt: str
    inbox: list = field(default_factory=list)
    # Per caller, how much of the agent's history it has already read.
    cursors: dict = field(default_factory=dict)
    changed: Optional[asyncio.Event] = None
    killed: bool = False
    # History kept when the runner goes away, so a late read still works.
    final: list = field(default_factory=list)
    # The state last reported to the parent; see Supervisor.status_summary.
    reported: AgentState = AgentState.RUNNING

    @property
    def messages(self) -> list:
        return self.final if self.runner is None else self.runner.agent.history


@dataclass
class _TaskRecord:
    task_id: str
    parent: object
    runner: object
    description: str
    command: object  # a tools.BackgroundCommand
    # Per caller, how many bytes of the output it has already read.
    cursors: dict = field(default_factory=dict)
    reported: TaskState = TaskState.RUNNING


class Supervisor:
    """Starts agents and background commands, and reports on both."""

    def __init__(self, *, launch, close, limit: int,
                 launch_task=None, close_task=None) -> None:
        # launch(agent_id, parent, model) -> runner, awaited; close(runner) -> None.
        # launch_task(task_id, command, description) -> runner, awaited;
        # close_task(runner) -> None. All four belong to the UI: only it can
        # put a pane on screen.
        self._launch = launch
        self._close = close
        self._launch_task = launch_task
        self._close_task = close_task
        self._limit = limit
        self._records: dict[str, _Record] = {}
        self._tasks: dict[str, _TaskRecord] = {}

    # ---- lifecycle ----------------------------------------------------------

    async def spawn(self, prompt: str, *, parent, model: Optional[str] = None) -> str:
        """Start an agent on ``prompt`` and return its id. Raises SupervisorError."""
        if not prompt.strip():
            raise SupervisorError("a prompt is required")
        self._check_room()
        agent_id = self._new_id()
        runner = await self._launch(agent_id, parent, model)
        record = _Record(agent_id, parent, runner, prompt, changed=asyncio.Event())
        self._records[agent_id] = record
        runner.start_turn(prompt)
        return agent_id

    def kill(self, agent_id: str) -> bool:
        """Stop an agent, keeping its transcript readable. False if unknown."""
        record = self._records.get(agent_id)
        if record is None or record.killed:
            return False
        self._retire(record, close=True)
        return True

    def kill_children(self, parent) -> list[str]:
        """Stop every agent and task ``parent`` started, and forget them.

        Forgotten, not just killed: the caller is losing the conversation those
        ids were mentioned in, so nothing can ask about them afterwards.
        """
        killed = []
        for agent_id, record in list(self._records.items()):
            if record.parent is not parent:
                continue
            if not record.killed:
                self._retire(record, close=True)
            del self._records[agent_id]
            killed.append(agent_id)
        for task_id, task in list(self._tasks.items()):
            if task.parent is not parent:
                continue
            self._retire_task(task, close=True)
            del self._tasks[task_id]
            killed.append(task_id)
        return killed

    def released(self, runner) -> None:
        """The UI closed a runner's pane on its own: that agent or task is over.

        The record stays: whoever started it can still read what it wrote, and
        gets ``killed`` rather than a puzzling ``unknown``.
        """
        for record in self._records.values():
            if record.runner is runner:
                self._retire(record, close=False)
                return
        for task in self._tasks.values():
            if task.runner is runner:
                self._retire_task(task, close=False)
                return

    def _retire(self, record: _Record, *, close: bool) -> None:
        # The transcript is snapshotted before the runner goes: after that
        # there is no agent left to read a history off.
        record.final = list(record.messages)
        record.killed = True
        record.inbox.clear()
        runner, record.runner = record.runner, None
        if close and runner is not None:
            self._close(runner)
        self._wake(record)

    # ---- background tasks ---------------------------------------------------

    async def start_task(self, command: str, description: str, *, parent, cwd) -> str:
        """Start a background command and open a pane for it. Returns its id."""
        if not command.strip():
            raise SupervisorError("a command is required")
        if self._launch_task is None:
            raise SupervisorError("background tasks are not available here")
        self._check_room()
        task_id = self._new_id()
        running = await start_background(command, cwd)
        try:
            runner = await self._launch_task(task_id, running, description)
        except BaseException:
            # Nothing would ever kill it: no pane holds it and no record names
            # it, and it is in its own process group, so it would outlive the
            # app itself.
            running.kill()
            raise
        self._tasks[task_id] = _TaskRecord(task_id, parent, runner, description, running)
        return task_id

    def kill_task(self, task_id: str, *, caller=None) -> bool:
        """Stop a task, keeping its output readable. False if unknown."""
        task = self._tasks.get(task_id) if caller is None else self._find_task(task_id, caller)
        if task is None or task.runner is None:
            return False
        self._retire_task(task, close=True)
        return True

    def read_task(self, task_id: str, *, caller, mode: str = "new") -> TaskView:
        task = self._find_task(task_id, caller)
        if task is None:
            return TaskView(task_id, TaskState.UNKNOWN, "")
        start = 0 if mode == "all" else task.cursors.get(caller, 0)
        data, cursor, dropped = task.command.output.since(start)
        task.cursors[caller] = cursor
        return TaskView(task_id, self._task_state(task), tail_text(data, dropped),
                        exit_code=task.command.exit_code, complete=start == 0)

    def task_states(self) -> dict[str, TaskState]:
        return {task_id: self._task_state(task) for task_id, task in self._tasks.items()}

    def _retire_task(self, task: _TaskRecord, *, close: bool) -> None:
        # The output buffer belongs to the command object, which the record
        # keeps holding, so a late read still works after the pane is gone.
        task.command.kill()
        runner, task.runner = task.runner, None
        if close and runner is not None and self._close_task is not None:
            self._close_task(runner)

    def _find_task(self, task_id: str, caller) -> Optional[_TaskRecord]:
        task = self._tasks.get(task_id)
        return task if task is not None and task.parent is caller else None

    @staticmethod
    def _task_state(task: _TaskRecord) -> TaskState:
        if task.command.killed:
            return TaskState.KILLED
        return TaskState.RUNNING if task.command.running else TaskState.EXITED

    # ---- delivery -----------------------------------------------------------

    async def send(self, agent_id: str, prompt: str, *, caller) -> Delivery:
        """Give an agent another instruction, queueing it if it is busy.

        The queue lives here rather than in the runner's own input: that one
        hands its contents back to a text box when a turn is interrupted, which
        for an agent nobody is watching means the message is silently lost
        after the caller was told it was delivered.
        """
        record = self._find(agent_id, caller)
        if record is None:
            return Delivery(agent_id, AgentState.UNKNOWN, accepted=False)
        if record.killed:
            return Delivery(agent_id, AgentState.KILLED, accepted=False)
        if record.runner.is_busy:
            # One queued prompt stays one turn: two separate instructions
            # merged into a single turn is not what either of them asked for.
            record.inbox.append(prompt)
            return Delivery(agent_id, self._state(record), queued=True)
        record.runner.start_turn(prompt)
        return Delivery(agent_id, AgentState.RUNNING)

    def pump(self) -> None:
        """Hand a queued prompt to every agent that has gone idle.

        Called whenever a runner changes state; also wakes anything waiting.
        """
        for record in self._records.values():
            if not record.killed and record.inbox and not record.runner.is_busy:
                record.runner.start_turn(record.inbox.pop(0))
            self._wake(record)

    # ---- reading ------------------------------------------------------------

    def read(self, agent_id: str, *, caller, mode: str = "new") -> AgentView:
        record = self._find(agent_id, caller)
        if record is None:
            return AgentView(agent_id, AgentState.UNKNOWN, "")
        messages = record.messages
        # Keyed by the caller itself, which the record already holds: the same
        # agent read twice gets only what is new, and a cursor can never be
        # confused with somebody else's.
        start = 0 if mode == "all" else min(record.cursors.get(caller, 0), len(messages))
        record.cursors[caller] = len(messages)
        return AgentView(agent_id, self._state(record), render_text(messages[start:]),
                         complete=start == 0)

    async def wait(self, agent_id: str, *, caller, timeout: float) -> AgentState:
        """Block until the agent stops running, or the timeout runs out.

        Returns RUNNING when it is still going: a caller that cannot get its
        turn back is a deadlock, and the agent may be waiting on a permission
        prompt in a tab the user is not looking at.
        """
        record = self._find(agent_id, caller)
        if record is None:
            return AgentState.UNKNOWN
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            state = self._state(record)
            if state is not AgentState.RUNNING:
                return state
            remaining = deadline - loop.time()
            if remaining <= 0:
                return AgentState.RUNNING
            if record.changed is None:
                record.changed = asyncio.Event()
            record.changed.clear()
            try:
                await asyncio.wait_for(record.changed.wait(), remaining)
            except asyncio.TimeoutError:
                return self._state(record)

    def states(self) -> dict[str, AgentState]:
        return {agent_id: self._state(record) for agent_id, record in self._records.items()}

    def children(self, parent) -> list[str]:
        return [agent_id for agent_id, record in self._records.items() if record.parent is parent]

    def status_summary(self, parent) -> Optional[str]:
        """One line about what ``parent`` started and what became of it.

        Each report advances a per-agent cursor, so "a1f2 finished" is stated
        once and not repeated at the top of every later turn.
        """
        news = []
        for record in self._records.values():
            if record.parent is not parent:
                continue
            state = self._state(record)
            if state is not record.reported and state in _NEWS:
                news.append(f"{record.agent_id} {_NEWS[state]}")
            record.reported = state
        for task in self._tasks.values():
            if task.parent is not parent:
                continue
            state = self._task_state(task)
            if state is not task.reported and state in _TASK_NEWS:
                detail = (f" (code {task.command.exit_code})"
                          if state is TaskState.EXITED else "")
                news.append(f"{task.task_id} {_TASK_NEWS[state]}{detail}")
            task.reported = state
        return " · ".join(news) or None

    # ---- the tool calls -----------------------------------------------------

    async def handle(self, name: str, args: dict, *, caller) -> str:
        """Run one agent tool call and render its result for the model."""
        agent_id = str(args.get("agent_id") or "").strip()
        if name == "spawn_agent":
            model = args.get("model")
            try:
                agent_id = await self.spawn(str(args.get("prompt") or ""), parent=caller,
                                            model=str(model) if model else None)
            except SupervisorError as exc:
                return f"Error: {exc}"
            except Exception as exc:  # noqa: BLE001 — a session or a pane that would not open
                return f"Error: could not start an agent: {exc}"
            return (f"Started agent {agent_id}; it is running now. Use read_agent to "
                    "collect its output and wait_for_agent to wait for it.")

        if name == "send_to_agent":
            prompt = str(args.get("prompt") or "").strip()
            if not prompt:
                return "Error: prompt is required."
            delivery = await self.send(agent_id, prompt, caller=caller)
            if not delivery.accepted:
                return self._gone(agent_id, delivery.state)
            if delivery.queued:
                return f"Queued for agent {agent_id}; it runs after its current turn."
            return f"Delivered to agent {agent_id}; it is working on it now."

        if name == "read_agent":
            mode = "all" if args.get("mode") == "all" else "new"
            view = self.read(agent_id, caller=caller, mode=mode)
            if view.state is AgentState.UNKNOWN:
                return self._gone(agent_id, view.state)
            header = f"agent {agent_id} [{view.state.value}]"
            if not view.text:
                extra = "" if view.complete else " since your last read"
                return f"{header}: no output{extra}."
            return f"{header}:\n{view.text}"

        if name == "wait_for_agent":
            timeout = _timeout(args.get("timeout"))
            state = await self.wait(agent_id, caller=caller, timeout=timeout)
            if state is AgentState.UNKNOWN:
                return self._gone(agent_id, state)
            if state is AgentState.RUNNING:
                return (f"agent {agent_id} is still running after {timeout:g}s. "
                        "Wait again, or read what it has written so far.")
            if state is AgentState.NEEDS_CONFIRM:
                return (f"agent {agent_id} is waiting for the user to confirm a tool call "
                        "in its own tab. It cannot continue until they answer, so tell them.")
            return f"agent {agent_id} is {state.value}. Read its output with read_agent."

        task_id = str(args.get("task_id") or "").strip()
        if name == "run_background":
            command = str(args.get("command") or "").strip()
            if not command:
                return "Error: command is required."
            try:
                task_id = await self.start_task(
                    command, str(args.get("description") or "").strip(),
                    parent=caller, cwd=caller.cwd)
            except SupervisorError as exc:
                return f"Error: {exc}"
            except Exception as exc:  # noqa: BLE001 — a command or a pane that would not start
                return f"Error: could not start the command: {exc}"
            return (f"Started background task {task_id}; it keeps running after this turn. "
                    "Read its output with read_task and stop it with kill_task.")

        if name == "read_task":
            mode = "all" if args.get("mode") == "all" else "new"
            view = self.read_task(task_id, caller=caller, mode=mode)
            if view.state is TaskState.UNKNOWN:
                return (f"Error: no background task {task_id}. Ids from an earlier run are "
                        "gone; start the command again if it still needs running.")
            if view.state is TaskState.EXITED:
                header = f"task {task_id} [exited, code {view.exit_code}]"
            else:
                header = f"task {task_id} [{view.state.value}]"
            if not view.text.strip():
                extra = "" if view.complete else " since your last read"
                return f"{header}: no output{extra}."
            return f"{header}:\n{view.text}"

        if name == "kill_task":
            if not self.kill_task(task_id, caller=caller):
                return (f"Error: no background task {task_id} to stop; it may already have "
                        "been stopped.")
            return f"Stopped background task {task_id}. Its output is still readable."

        return f"Error: unknown tool {name!r}"

    @staticmethod
    def _gone(agent_id: str, state: AgentState) -> str:
        if state is AgentState.KILLED:
            return f"Error: agent {agent_id} was stopped."
        return (f"Error: no agent {agent_id}. Ids from an earlier run are gone; "
                "start a new agent if the work still needs doing.")

    # ---- internals ----------------------------------------------------------

    def _new_id(self) -> str:
        # One id space for agents and tasks: they are quoted back to the model
        # side by side, and an id that names one of each would be read as
        # whichever the model happened to expect.
        while True:
            new_id = uuid4().hex[:4]
            if new_id not in self._records and new_id not in self._tasks:
                return new_id

    def _check_room(self) -> None:
        """Refuse before starting anything when every pane is taken."""
        live = (sum(1 for record in self._records.values() if not record.killed)
                + sum(1 for task in self._tasks.values() if task.runner is not None))
        if live >= self._limit:
            raise SupervisorError(
                f"{live} agents and tasks are already running (limit {self._limit}); "
                "stop one before starting another")

    def _find(self, agent_id: str, caller) -> Optional[_Record]:
        """The record ``caller`` is allowed to see, or None.

        An agent belongs to whoever started it: everyone else gets the same
        answer as for an id that never existed.
        """
        record = self._records.get(agent_id)
        return record if record is not None and record.parent is caller else None

    def _state(self, record: _Record) -> AgentState:
        if record.killed:
            return AgentState.KILLED
        runner = record.runner
        if runner.needs_confirm:
            return AgentState.NEEDS_CONFIRM
        if runner.is_busy or record.inbox:
            return AgentState.RUNNING
        if runner.turn_failed:
            return AgentState.FAILED
        return AgentState.IDLE

    @staticmethod
    def _wake(record: _Record) -> None:
        if record.changed is not None:
            record.changed.set()


def _timeout(value: object) -> float:
    """The requested wait, clamped to something that always comes back."""
    try:
        timeout = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_WAIT_TIMEOUT
    if timeout <= 0:
        return DEFAULT_WAIT_TIMEOUT
    return min(timeout, MAX_WAIT_TIMEOUT)
