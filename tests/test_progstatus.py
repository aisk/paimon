import asyncio
import base64
import unittest
from unittest.mock import patch

from textual.drivers.headless_driver import HeadlessDriver

from paimon import progstatus
from paimon.agent import Agent
from paimon.app import PaimonApp
from paimon.config import Config
from paimon.turns import Outcome, Result, TurnDriver

from .support.app import AppTestCase


def _fields(sequence: str) -> dict[str, str]:
    """The pairs of one report, with its message decoded."""
    assert sequence.startswith("\x1b]7501;") and sequence.endswith("\x1b\\"), sequence
    fields = dict(pair.split("=", 1) for pair in sequence[7:-2].split(":"))
    if "msg" in fields:
        fields["msg"] = base64.b64decode(fields["msg"]).decode("utf-8")
    return fields


class EncodeTest(unittest.TestCase):
    def test_state_alone(self) -> None:
        self.assertEqual(progstatus.encode(progstatus.WORKING), "\x1b]7501;state=working:app=paimon\x1b\\")

    def test_blocked_says_on_what(self) -> None:
        sequence = progstatus.encode(progstatus.BLOCKED, kind=progstatus.PERMISSION, msg="shell: rm -rf build")
        self.assertEqual(_fields(sequence), {"state": "blocked", "kind": "permission", "app": "paimon",
                                             "msg": "shell: rm -rf build"})

    def test_message_is_one_line_without_control_characters(self) -> None:
        sequence = progstatus.encode(progstatus.ERROR, msg="first\nsecond\there\x1b[31m \x07 done")
        self.assertEqual(_fields(sequence)["msg"], "first second here [31m done")

    def test_long_message_is_cut_on_a_character(self) -> None:
        sequence = progstatus.encode(progstatus.ERROR, msg="错" * 2000)
        message = _fields(sequence)["msg"]
        self.assertEqual(message, "错" * 682)
        self.assertLessEqual(len(sequence.encode("utf-8")), 4096)

    def test_lone_surrogate_is_replaced(self) -> None:
        sequence = progstatus.encode(progstatus.BLOCKED, kind=progstatus.PERMISSION, msg="echo \ud83d")
        self.assertEqual(_fields(sequence)["msg"], "echo ?")

    def test_only_the_protocol_alphabet_is_sent(self) -> None:
        sequence = progstatus.encode(progstatus.BLOCKED, kind=progstatus.QUESTION, msg="a:b=c;d ?")
        self.assertRegex(sequence[7:-2], r"^[a-z]+=[A-Za-z0-9_.,+/=-]*(:[a-z]+=[A-Za-z0-9_.,+/=-]*)*$")


class AppStatusTest(AppTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.written: list[str] = []
        write = patch.object(HeadlessDriver, "write", lambda _driver, data: self.written.append(data))
        write.start()
        self.addCleanup(write.stop)

    def _reports(self) -> list[dict[str, str]]:
        return [_fields(data) for data in self.written if data.startswith("\x1b]7501;")]

    @staticmethod
    def _app(program_status: bool = True) -> PaimonApp:
        agent = Agent.open(mode="read", config=Config(model="test-model"))
        return PaimonApp(agent, program_status=program_status)

    async def test_starts_idle_and_says_so_once(self) -> None:
        app = self._app()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._sync_panes()
            self.assertEqual(self._reports(), [{"state": "idle", "app": "paimon"}])

    async def test_nothing_is_written_unless_asked(self) -> None:
        async with self._app(program_status=False).run_test() as pilot:
            await pilot.pause()
            self.assertEqual(self._reports(), [])

    async def test_confirmation_blocks_on_a_permission(self) -> None:
        app = self._app()
        async with app.run_test() as pilot:
            task = await self._open_confirm(app, pilot)
            self.assertEqual(self._reports()[-1], {"state": "blocked", "kind": "permission",
                                                   "app": "paimon", "msg": "shell: echo hi"})
            await self._wait_for_panel(app, pilot)
            await pilot.press("enter")
            await task
            await pilot.pause()
            self.assertEqual(self._reports()[-1]["state"], "idle")

    async def test_question_blocks_on_an_answer(self) -> None:
        app = self._app()
        async with app.run_test() as pilot:
            pane = app.panes[0]
            task = asyncio.ensure_future(pane._ask("Which branch?", ["main", "dev"]))
            await self._wait_for(pilot, lambda: pane.needs_confirm)
            self.assertEqual(self._reports()[-1], {"state": "blocked", "kind": "question",
                                                   "app": "paimon", "msg": "Which branch?"})
            await self._wait_for_panel(app, pilot)
            await pilot.press("escape")
            await task

    async def _end_turn(self, app: PaimonApp, pilot, result: Result) -> None:
        """Have the first pane's turn look started, then over with ``result``."""
        pane = app.panes[0]
        with patch.object(type(pane.driver), "is_busy", True):
            app._sync_panes()
            self.assertEqual(self._reports()[-1]["state"], "working")
        pane.driver.result = result
        app._sync_panes()
        await pilot.pause()

    async def test_finished_turn_is_done(self) -> None:
        app = self._app()
        async with app.run_test() as pilot:
            await pilot.pause()
            await self._end_turn(app, pilot, Result(Outcome.SUCCESS))
            self.assertEqual(self._reports()[-1], {"state": "done", "app": "paimon"})

    async def test_failed_turn_is_an_error_with_its_reason(self) -> None:
        app = self._app()
        async with app.run_test() as pilot:
            await pilot.pause()
            await self._end_turn(app, pilot, Result(Outcome.FAILED, error="rate limited"))
            self.assertEqual(self._reports()[-1], {"state": "error", "app": "paimon", "msg": "rate limited"})

    async def test_interrupted_turn_is_idle(self) -> None:
        app = self._app()
        async with app.run_test() as pilot:
            await pilot.pause()
            await self._end_turn(app, pilot, Result(Outcome.SUCCESS))
            await self._end_turn(app, pilot, Result(Outcome.INTERRUPTED))
            self.assertEqual(self._reports()[-1], {"state": "idle", "app": "paimon"})

    async def test_result_of_a_session_swapped_out_is_forgotten(self) -> None:
        app = self._app()
        async with app.run_test() as pilot:
            await pilot.pause()
            await self._end_turn(app, pilot, Result(Outcome.FAILED, error="rate limited"))
            app.action_new_session()
            await pilot.pause()
            self.assertEqual(self._reports()[-1], {"state": "idle", "app": "paimon"})

    async def test_tab_closed_mid_turn_does_not_repeat_an_older_result(self) -> None:
        app = self._app()
        async with app.run_test() as pilot:
            await pilot.pause()
            await self._end_turn(app, pilot, Result(Outcome.SUCCESS))
            await pilot.press("ctrl+t")
            await pilot.pause()
            second = app.panes[1]
            # Busy on this driver alone: the first pane's turn stays over.
            second.driver.__class__ = type("Busy", (TurnDriver,), {"is_busy": True})
            app._sync_panes()
            self.assertEqual(self._reports()[-1]["state"], "working")
            await app._drop_pane(second)
            await pilot.pause()
            self.assertEqual(self._reports()[-1], {"state": "idle", "app": "paimon"})
