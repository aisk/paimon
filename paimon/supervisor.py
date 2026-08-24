"""The pool of jobs this process is running, and the tools that reach it.

One ``Supervisor`` per app. It owns the job table, hands out the ids the model
quotes back, decides who is allowed to see what, and renders each job tool's
result. What a job *is* and how it runs lives in ``jobs.py``; what puts one on
screen is a pair of callbacks the UI supplies, because only the UI can open a
pane.

Ownership is the rule everything else follows from: a job belongs to whoever
started it, and an id belonging to somebody else gets the same answer as an id
that never existed. Panes the user opened have no owner at all, so they are in
the table for the pane budget and invisible to every tool.

Deliberately free of Textual, so delivery, cleanup and the permission boundary
can be tested against plain jobs instead of a driven terminal.
"""

from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from .errors import PaimonError
from .jobs import Job, State
from .tools import DEFAULT_WAIT_TIMEOUT, MAX_WAIT_TIMEOUT, start_background


class SupervisorError(PaimonError):
    """A job could not be started. Reported to the model, not raised at it."""


@dataclass(frozen=True)
class Delivery:
    """The outcome of sending a prompt to a job."""

    state: State
    # False when the id named nothing the caller may send to, or named a job
    # that takes no instructions at all.
    accepted: bool = True
    # The agent was busy, so the prompt waits in its inbox for the next turn.
    queued: bool = False


# What a state change is called in the status line injected into the parent's
# next turn. RUNNING is absent on purpose: "still running" is not news.
_NEWS = {
    State.IDLE: "finished",
    State.NEEDS_CONFIRM: "needs confirmation",
    State.DONE: "exited",
    State.FAILED: "failed",
    State.KILLED: "was stopped",
}


class Supervisor:
    """Starts agents and background commands, and reports on both."""

    def __init__(self, *, launch, close, limit: int, launch_command=None) -> None:
        # launch(job_id, parent, model, agent) -> Job and launch_command(
        # job_id, command, description) -> Job are awaited and must have put a
        # pane on screen by the time they return; close(job) takes that pane
        # down again. All three belong to the UI. ``agent`` is an agent type
        # name the launcher resolves; the supervisor never interprets it.
        self._launch = launch
        self._launch_command = launch_command
        self._close = close
        self._limit = limit
        self._jobs: dict[str, Job] = {}

    # ---- the table ----------------------------------------------------------

    def register(self, job: Job) -> None:
        """Take a job the UI made on its own into the table.

        A pane the user opened is one of these: it has no owner, so no tool can
        reach it, but it is a live agent holding a pane and has to count
        against the budget like any other.
        """
        self._jobs[job.job_id] = job

    def new_id(self) -> str:
        """An id no live job is using.

        One space for agents and commands both: they are quoted back to the
        model side by side, and an id that named one of each would be read as
        whichever the model happened to expect. It is also what lets read,
        wait and stop each be a single tool.
        """
        while True:
            new_id = uuid4().hex[:4]
            if new_id not in self._jobs:
                return new_id

    def jobs(self) -> dict[str, Job]:
        return dict(self._jobs)

    def states(self) -> dict[str, State]:
        return {job_id: job.state for job_id, job in self._jobs.items()}

    def children(self, parent) -> list[str]:
        return [job_id for job_id, job in self._jobs.items() if job.parent is parent]

    # ---- lifecycle ----------------------------------------------------------

    async def spawn(self, prompt: str, *, parent, model: Optional[str] = None,
                    agent: Optional[str] = None) -> str:
        """Start an agent on ``prompt`` and return its id. Raises SupervisorError."""
        if not prompt.strip():
            raise SupervisorError("a prompt is required")
        self._check_room()
        job_id = self.new_id()
        job = await self._launch(job_id, parent, model, agent)
        self._jobs[job_id] = job
        job.submit(prompt)
        return job_id

    async def start_command(self, command: str, description: str, *, parent, cwd) -> str:
        """Start a background command and open a pane for it. Returns its id."""
        if not command.strip():
            raise SupervisorError("a command is required")
        if self._launch_command is None:
            raise SupervisorError("background commands are not available here")
        self._check_room()
        job_id = self.new_id()
        running = await start_background(command, cwd)
        try:
            job = await self._launch_command(job_id, running, description)
        except BaseException:
            # Nothing would ever kill it: no pane holds it and no job names it,
            # and it is in its own process group, so it would outlive the app.
            running.kill()
            raise
        self._jobs[job_id] = job
        return job_id

    def stop(self, job_id: str, *, caller=None) -> bool:
        """End a job and close its pane, keeping what it produced readable.

        False when the id names nothing the caller may stop.
        """
        job = self._jobs.get(job_id) if caller is None else self._find(job_id, caller)
        if job is None or job.killed:
            return False
        job.cancel()
        self._close(job)
        return True

    def released(self, job: Job) -> None:
        """The UI closed a job's pane on its own: that job is over.

        An owned job stays in the table — whoever started it can still read
        what it wrote, and gets ``killed`` rather than a puzzling ``unknown``.
        An unowned one is forgotten outright, since nothing can ask about it.
        """
        job.cancel()
        if job.parent is None:
            self._jobs.pop(job.job_id, None)

    def kill_children(self, parent) -> list[str]:
        """Stop every job ``parent`` started, and forget them.

        Forgotten, not just stopped: the caller is losing the conversation
        those ids were mentioned in, so nothing can ask about them afterwards.
        """
        stopped = []
        for job_id, job in list(self._jobs.items()):
            if job.parent is not parent:
                continue
            job.cancel()
            self._close(job)
            del self._jobs[job_id]
            stopped.append(job_id)
        return stopped

    # ---- delivery -----------------------------------------------------------

    def send(self, job_id: str, prompt: str, *, caller) -> Delivery:
        """Give a job more work.

        The queue lives in the job rather than in the pane's own input: that
        one hands its contents back to a text box when a turn is interrupted,
        which for an agent nobody is watching means the message is silently
        lost after the caller was told it was delivered.
        """
        job = self._find(job_id, caller)
        if job is None:
            return Delivery(State.UNKNOWN, accepted=False)
        if job.killed:
            return Delivery(State.KILLED, accepted=False)
        queued = job.is_busy
        if not job.submit(prompt):
            # A background command. Not an error the caller can retry into
            # success, so it is refused rather than queued nowhere.
            return Delivery(job.state, accepted=False)
        return Delivery(job.state, queued=queued)

    # ---- reading ------------------------------------------------------------

    def read(self, job_id: str, *, caller, mode: str = "new"):
        job = self._find(job_id, caller)
        if job is None:
            return None
        return job.read(caller, mode)

    async def wait(self, job_id: str, *, caller, timeout: float) -> State:
        job = self._find(job_id, caller)
        if job is None:
            return State.UNKNOWN
        return await job.wait(timeout)

    def status_summary(self, parent) -> Optional[str]:
        """One line about what ``parent`` started and what became of it.

        Each report advances a per-job cursor, so "a1f2 finished" is stated
        once and not repeated at the top of every later turn.
        """
        news = []
        for job in self._jobs.values():
            if job.parent is not parent:
                continue
            state = job.state
            if state is not job.reported and state in _NEWS:
                if job.kind == "command" and state in (State.DONE, State.FAILED):
                    # For a command "failed" would drop the one detail that
                    # decides what to do about it.
                    news.append(f"{job.job_id} exited (code {job.exit_code})")
                else:
                    news.append(f"{job.job_id} {_NEWS[state]}")
            job.reported = state
        return " · ".join(news) or None

    # ---- the tool calls -----------------------------------------------------

    async def handle(self, name: str, args: dict, *, caller) -> str:
        """Run one job tool call and render its result for the model."""
        if name == "spawn_agent":
            model = args.get("model")
            agent = args.get("agent")
            try:
                job_id = await self.spawn(str(args.get("prompt") or ""), parent=caller,
                                          model=str(model) if model else None,
                                          agent=str(agent) if agent else None)
            except SupervisorError as exc:
                return f"Error: {exc}"
            except Exception as exc:  # noqa: BLE001 — a session or a pane that would not open
                return f"Error: could not start an agent: {exc}"
            return (f"Started agent {job_id}; it is running now. Use read_job to collect "
                    "its output and wait_for_job to wait for it.")

        if name == "run_background":
            command = str(args.get("command") or "").strip()
            if not command:
                return "Error: command is required."
            try:
                job_id = await self.start_command(
                    command, str(args.get("description") or "").strip(),
                    parent=caller, cwd=caller.cwd)
            except SupervisorError as exc:
                return f"Error: {exc}"
            except Exception as exc:  # noqa: BLE001 — a command or a pane that would not start
                return f"Error: could not start the command: {exc}"
            return (f"Started background command {job_id}; it keeps running after this turn. "
                    "Read its output with read_job and stop it with stop_job.")

        job_id = str(args.get("job_id") or "").strip()

        if name == "send_to_agent":
            prompt = str(args.get("prompt") or "").strip()
            if not prompt:
                return "Error: prompt is required."
            delivery = self.send(job_id, prompt, caller=caller)
            if not delivery.accepted:
                if delivery.state in (State.UNKNOWN, State.KILLED):
                    return self._gone(job_id, delivery.state)
                return (f"Error: {job_id} is a background command, which takes no "
                        "instructions. Read its output with read_job instead.")
            if delivery.queued:
                return f"Queued for agent {job_id}; it runs after its current turn."
            return f"Delivered to agent {job_id}; it is working on it now."

        if name == "read_job":
            mode = "all" if args.get("mode") == "all" else "new"
            view = self.read(job_id, caller=caller, mode=mode)
            if view is None:
                return self._gone(job_id, State.UNKNOWN)
            if view.state is State.DONE or (view.state is State.FAILED
                                            and view.exit_code is not None):
                header = f"{job_id} [exited, code {view.exit_code}]"
            else:
                header = f"{job_id} [{view.state.value}]"
            if not view.text.strip():
                extra = "" if view.complete else " since your last read"
                return f"{header}: no output{extra}."
            return f"{header}:\n{view.text}"

        if name == "wait_for_job":
            timeout = _timeout(args.get("timeout"))
            state = await self.wait(job_id, caller=caller, timeout=timeout)
            if state is State.UNKNOWN:
                return self._gone(job_id, state)
            if state is State.RUNNING:
                return (f"{job_id} is still running after {timeout:g}s. Wait again, or "
                        "read what it has produced so far.")
            if state is State.NEEDS_CONFIRM:
                return (f"agent {job_id} is waiting for the user to confirm a tool call "
                        "in its own tab. It cannot continue until they answer, so tell them.")
            return f"{job_id} is {state.value}. Read what it produced with read_job."

        if name == "stop_job":
            if not self.stop(job_id, caller=caller):
                return (f"Error: nothing to stop under {job_id}; it may already have "
                        "been stopped.")
            return f"Stopped {job_id}. What it produced is still readable."

        return f"Error: unknown tool {name!r}"

    @staticmethod
    def _gone(job_id: str, state: State) -> str:
        if state is State.KILLED:
            return f"Error: {job_id} was stopped."
        return (f"Error: no agent or background command {job_id}. Ids from an earlier run "
                "are gone; start it again if the work still needs doing.")

    # ---- internals ----------------------------------------------------------

    def _check_room(self) -> None:
        """Refuse before starting anything when every pane is taken."""
        live = sum(1 for job in self._jobs.values() if not job.killed)
        if live >= self._limit:
            raise SupervisorError(
                f"{live} agents and commands are already running (limit {self._limit}); "
                "stop one before starting another")

    def _find(self, job_id: str, caller) -> Optional[Job]:
        """The job ``caller`` is allowed to see, or None.

        A job belongs to whoever started it: everyone else gets the same answer
        as for an id that never existed.
        """
        job = self._jobs.get(job_id)
        return job if job is not None and job.parent is caller else None


def _timeout(value: object) -> float:
    """The requested wait, clamped to something that always comes back."""
    try:
        timeout = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_WAIT_TIMEOUT
    if timeout <= 0:
        return DEFAULT_WAIT_TIMEOUT
    return min(timeout, MAX_WAIT_TIMEOUT)
