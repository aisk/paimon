"""TUI setup and interactions shared across feature tests."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
)

from paimon.agent import Agent
from paimon.app import PaimonApp
from paimon.config import Config
from paimon.jobs import Outcome, Result
from paimon.pane import SessionPane
from paimon.session import (
    Session,
)
from paimon.tabs import PaneTab


class AppTestCase(unittest.IsolatedAsyncioTestCase):
    """Pilot-driven TUI tests against an isolated data dir and a stub model."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = patch.dict("os.environ", {"PAIMON_DATA_HOME": tmp.name,
                                        "PAIMON_CONFIG_HOME": tmp.name})
        env.start()
        self.addCleanup(env.stop)

    def make_app(self, *, session: Session | None = None, mode: str = "read",
                 config: Config | None = None, pick_session: bool = False) -> PaimonApp:
        agent = Agent.open(session=session, mode=mode, config=config or Config(model="test-model"))
        return PaimonApp(agent, resumed=session is not None, pick_session=pick_session)

    @staticmethod
    async def _open_confirm(app: PaimonApp, pilot, tool: str = "shell", args: dict | None = None) -> asyncio.Future:
        task = asyncio.ensure_future(app.pane._confirm(tool, args or {"command": "echo hi"}))
        await pilot.pause()
        return task

    @staticmethod
    def _old_session(content: str = "hello there") -> Session:
        session = Session.create(Path.cwd())
        session.append_system_prompt("sys")
        session.append_message(ModelRequest(parts=[UserPromptPart(content=content)]))
        return session

    @staticmethod
    def _tab_text(app, pane: SessionPane) -> str:
        """A tab's drawn text with its frame and padding taken back out."""
        tab = app.query_one(f"#tab-{pane.id}", PaneTab)
        lines = str(tab.render()).splitlines()
        return " ".join(line.strip("╭╮╰╯┬┴│─ ") for line in lines).strip()

    @staticmethod
    def _log_text(pane: SessionPane) -> str:
        # Not just direct children: a tool call and its result now nest one
        # level deeper, inside the step box that groups them with the
        # reasoning that led to the call.
        return " ".join(str(widget.render())
                         for widget in pane.query_one("#log").walk_children())

    @staticmethod
    async def _wait_for(pilot, condition, *, timeout: float = 5.0) -> None:
        """Wait for UI state, yielding time for timers and real subprocesses."""
        async def poll():
            while True:
                await pilot.pause()
                if condition():
                    return
                await asyncio.sleep(0.02)

        try:
            await asyncio.wait_for(poll(), timeout)
        except asyncio.TimeoutError as exc:
            raise AssertionError(f"UI condition not reached within {timeout}s") from exc


class _HeldTurn:
    """A stand-in for the task a real turn runs in."""

    def done(self) -> bool:
        return False

    def cancel(self) -> None:
        pass


def hold_turn(pane) -> None:
    """Make a pane look busy without a model behind it.

    The driver is parked on its inbox, so a stand-in for the turn task is all
    is_busy needs; submitting a prompt would really run one.
    """
    pane.job._turn = _HeldTurn()


async def end_turn(pane, outcome: Outcome = Outcome.SUCCESS, error: str = "") -> None:
    """Finish the held turn the way the driver would."""
    pane.job._turn = None
    await pane._end_turn(Result(outcome, error=error))
