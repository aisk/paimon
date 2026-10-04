import asyncio
from unittest.mock import patch

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from textual.widgets.markdown import MarkdownBlock

from paimon.app import PaimonApp
from paimon.config import Config
from paimon.session import Session
from paimon.ui import (
    PromptInput,
    RecapMessage,
)
from tests.support.app import AppTestCase


class RecapTest(AppTestCase):
    """A turn that did some work, then silence: Paimon says where things stand.

    The waits are real, the way the task-pane tests do it: nothing in the app
    fakes a clock, so neither does this.
    """

    RECAP = "读了 missing.txt；下一步把结果写回去"

    @staticmethod
    def _model(*, tool: bool = True, recap: str = RECAP, boom: bool = False) -> FunctionModel:
        """One tool call for the turn, a separate answer for the recap.

        A turn streams and a recap does not, so the two land on different
        functions without either having to look at what it was asked.
        """
        requests = 0

        async def stream(messages, info):
            nonlocal requests
            requests += 1
            if tool and requests == 1:
                yield {0: DeltaToolCall(name="read_file", json_args='{"path": "missing.txt"}',
                                        tool_call_id="call-1")}
            else:
                yield "done"

        def answer(messages, info) -> ModelResponse:
            if boom:
                raise ModelHTTPError(401, "stub")
            return ModelResponse(parts=[TextPart(content=recap)])

        return FunctionModel(answer, stream_function=stream)

    @staticmethod
    def _config(idle: float = 0.05, enabled: bool = True) -> Config:
        return Config(model="test-model", recap_idle_seconds=idle, recap_enabled=enabled)

    @staticmethod
    async def _stays_away(pilot, condition, rounds: int = 20) -> None:
        for _ in range(rounds):
            await pilot.pause()
            await asyncio.sleep(0.02)
            if condition():
                raise AssertionError("a recap turned up where none was due")

    @staticmethod
    def _recap_text(app: PaimonApp) -> str:
        # A recap is a Markdown container: its words live in the blocks it
        # mounted, not in the container's own render.
        return " ".join(
            str(block.render())
            for recap in app.query(RecapMessage)
            for block in recap.query(MarkdownBlock)
        )

    async def _finish_a_turn(self, app: PaimonApp, pilot) -> None:
        app.pane.handle_submit(PromptInput.Submitted("go"))
        await self._wait_for(pilot, lambda: not app.pane.is_busy)

    async def _armed(self, app: PaimonApp, pilot):
        """The recap timer, once the finished turn has got round to arming it.

        The job stops being busy a few awaits before the pane closes the turn
        out, which is where the timer is set.
        """
        await self._wait_for(pilot, lambda: app.pane._recap_timer is not None)
        return app.pane._recap_timer

    async def test_a_recap_follows_a_tool_turn_that_goes_quiet(self) -> None:
        app = self.make_app(config=self._config())
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                await self._finish_a_turn(app, pilot)
                await self._wait_for(pilot, lambda: bool(app.query(RecapMessage)))

                self.assertIn(self.RECAP, self._recap_text(app))
                self.assertIn("While you were away", self._recap_text(app))

    async def test_the_recap_is_not_part_of_the_conversation(self) -> None:
        app = self.make_app(config=self._config())
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                await self._finish_a_turn(app, pilot)
                agent = app.pane.agent
                messages = len(agent.history)
                lines = agent.session.path.read_text().count("\n")

                await self._wait_for(pilot, lambda: bool(app.query(RecapMessage)))

                self.assertEqual(len(agent.history), messages)
                self.assertEqual(agent.session.path.read_text().count("\n"), lines)
                self.assertNotIn(self.RECAP, agent.session.path.read_text())

    async def test_a_turn_that_only_answered_gets_no_recap(self) -> None:
        app = self.make_app(config=self._config())
        with patch("paimon.agent.build_model", return_value=self._model(tool=False)):
            async with app.run_test() as pilot:
                await self._finish_a_turn(app, pilot)

                await self._stays_away(pilot, lambda: bool(app.query(RecapMessage)))

    async def test_typing_restarts_the_countdown(self) -> None:
        # A wait long enough that the assertions are about the timer rather
        # than about how fast the machine running them is.
        app = self.make_app(config=self._config(idle=30))
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                await self._finish_a_turn(app, pilot)
                armed = await self._armed(app, pilot)
                self.assertIsNotNone(armed)

                await pilot.press("h")
                await pilot.pause()

                self.assertIsNotNone(app.pane._recap_timer)
                self.assertIsNot(app.pane._recap_timer, armed)
                self.assertFalse(app.query(RecapMessage))

    async def test_a_new_turn_drops_a_pending_recap(self) -> None:
        app = self.make_app(config=self._config(idle=30))
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                await self._finish_a_turn(app, pilot)
                await self._armed(app, pilot)

                app.pane.handle_submit(PromptInput.Submitted("more"))
                await pilot.pause()

                self.assertIsNone(app.pane._recap_timer)
                await self._wait_for(pilot, lambda: not app.pane.is_busy)

    async def test_a_failed_recap_says_nothing(self) -> None:
        app = self.make_app(config=self._config())
        with patch("paimon.agent.build_model", return_value=self._model(boom=True)):
            async with app.run_test() as pilot:
                await self._finish_a_turn(app, pilot)
                # Nothing at all is added, not even a note about the failure:
                # the recap was never asked for, so it does not get to complain.
                shown = len(app.query_one("#log").children)

                await self._stays_away(pilot, lambda: bool(app.query(RecapMessage)))
                self.assertEqual(len(app.query_one("#log").children), shown)

    async def test_only_the_pane_on_screen_recaps(self) -> None:
        app = self.make_app(config=self._config())
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                first = app.pane
                first.handle_submit(PromptInput.Submitted("go"))
                await app.action_new_pane()
                await self._wait_for(pilot, lambda: not first.is_busy)

                self.assertIsNot(app.pane, first)
                self.assertIsNone(first._recap_timer)
                await self._stays_away(pilot, lambda: bool(app.query(RecapMessage)))

    async def test_it_can_be_turned_off(self) -> None:
        app = self.make_app(config=self._config(enabled=False))
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                await self._finish_a_turn(app, pilot)

                self.assertIsNone(app.pane._recap_timer)
                await self._stays_away(pilot, lambda: bool(app.query(RecapMessage)))

    async def _session_with_work(self, app: PaimonApp, pilot) -> Session:
        """A finished tool turn, left behind in a pane that never recapped it."""
        await self._finish_a_turn(app, pilot)
        return app.pane.agent.session

    async def test_a_resumed_session_recaps_without_waiting(self) -> None:
        with patch("paimon.agent.build_model", return_value=self._model()):
            first = self.make_app(config=self._config(enabled=False))
            async with first.run_test() as pilot:
                session = await self._session_with_work(first, pilot)

            # An idle wait far longer than the test: what shows up came from
            # the resume, not from the timer.
            app = self.make_app(session=session, config=self._config(idle=30))
            async with app.run_test() as pilot:
                await self._wait_for(pilot, lambda: bool(app.query(RecapMessage)))

                self.assertIn(self.RECAP, self._recap_text(app))

    async def test_a_resumed_session_that_only_talked_gets_no_recap(self) -> None:
        with patch("paimon.agent.build_model", return_value=self._model(tool=False)):
            first = self.make_app(config=self._config(enabled=False))
            async with first.run_test() as pilot:
                session = await self._session_with_work(first, pilot)

            app = self.make_app(session=session, config=self._config(idle=30))
            async with app.run_test() as pilot:
                await self._stays_away(pilot, lambda: bool(app.query(RecapMessage)))
