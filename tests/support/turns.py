"""Controllable agents and commands without model requests or subprocesses."""

import asyncio
from pathlib import Path

from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart

from paimon.tools.background import _TaskOutput


async def settle(times: int = 8) -> None:
    """Let the driver and its turn task get as far as they can."""
    for _ in range(times):
        await asyncio.sleep(0)


class FakeAgent:
    """An Agent whose turns and compactions end when the test says so."""

    def __init__(self) -> None:
        self.history: list = []
        self.cwd = Path(".")
        self.prompts: list[str] = []
        self.events: list = []
        self.answer = "done"
        self.fail: str | None = None
        self.running: asyncio.Event | None = None
        # The hooks a TurnDriver takes, and what it reads back.
        self.notices: list[str] = []
        self.pending = None
        self.on_jobs_changed = None
        self.shell_runs: list[tuple[str, str]] = []
        self.compactions = 0

    async def run(self, prompt: str, *, expand: bool = True):
        self.prompts.append(prompt)
        # Taken as the turn starts, as Agent.run takes them. The fake has no
        # later steps, so one that lands mid-turn waits for the next turn.
        self.notices.clear()
        self.running = asyncio.Event()
        for event in self.events:
            yield event
        await self.running.wait()
        if self.fail:
            raise RuntimeError(self.fail)
        self.history.append(ModelResponse(parts=[
            ThinkingPart(content="secret reasoning"), TextPart(content=self.answer)]))

    async def compact_now(self):
        self.compactions += 1
        self.running = asyncio.Event()
        await self.running.wait()
        if self.fail:
            raise RuntimeError(self.fail)
        return None

    def record_shell(self, command: str, output: str) -> None:
        self.shell_runs.append((command, output))

    def report(self, notice: str) -> None:
        """A job of this agent's reported, the way Agent._notify tells it."""
        self.notices.append(notice)
        if self.on_jobs_changed is not None:
            self.on_jobs_changed()

    def finish(self) -> None:
        assert self.running is not None, "no turn is running"
        self.running.set()


class FakeCommand:
    """A tools.BackgroundCommand, minus the process."""

    def __init__(self, command: str = "sleep 30") -> None:
        self.command = command
        self.output = _TaskOutput()
        self.exit_code = None
        self.killed = False
        self._over = asyncio.Event()

    @property
    def running(self) -> bool:
        return self.exit_code is None

    async def wait(self):
        await self._over.wait()
        return self.exit_code

    def kill(self) -> None:
        if self.killed or not self.running:
            return
        self.killed = True
        self.exit_code = -15
        self._over.set()

    def terminate_now(self) -> None:
        self.kill()

    def exit(self, code: int = 0) -> None:
        self.exit_code = code
        self._over.set()
