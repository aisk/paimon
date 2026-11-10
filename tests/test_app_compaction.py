"""A manual compaction in the TUI: one item of the driver's, like a turn.

The summary request takes a while, and while it is out nothing else may write
the history: the compaction replaces it with one built from what it was when
the request went out, and anything appended meanwhile would be dropped from
the context and from what the log replays.
"""

import asyncio
from unittest.mock import patch

from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart

from paimon import compaction
from paimon.session import SHELL, SUMMARY, synthetic_kind
from paimon.ui import PromptInput
from tests.support.agent import stub_model
from tests.support.app import AppTestCase


class ManualCompactionTest(AppTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.gate = asyncio.Event()
        self.asked = asyncio.Event()

        async def compact(messages, **kwargs):
            # Snapshotted before the wait, as the real one hands its walk of
            # the history to a thread before the request goes out.
            kept = list(messages[-1:])
            self.asked.set()
            await self.gate.wait()
            return compaction.CompactionResult("checkpoint", kept, 100, 0)

        self.compact = compact

    def _patches(self):
        return (patch("paimon.agent.build_model", return_value=stub_model()),
                patch("paimon.agent.compaction.compact", side_effect=self.compact))

    async def _start(self, app, pilot):
        app.action_compact()
        await self._wait_for(pilot, self.asked.is_set)
        self.assertIn("Compacting context", self._status(app))

    @staticmethod
    def _status(app) -> str:
        status = app.pane.query_one("#response-status")
        return str(status.query_one(".status-label").render()) if status.display else ""

    def assert_replays_to(self, pane) -> None:
        """The agent's context is what its log replays, checkpoint included."""
        history, replayed = pane.agent.history, pane.agent.session.messages()
        self.assertEqual(synthetic_kind(replayed[0]), SUMMARY)
        self.assertEqual(history, replayed)

    @staticmethod
    def _prompts(history) -> list[str]:
        return [part.content for message in history if isinstance(message, ModelRequest)
                for part in message.parts if isinstance(part, UserPromptPart)]

    async def test_a_prompt_sent_during_it_runs_after_it_and_is_kept(self) -> None:
        app = self.make_app(session=self._old_session("earlier"))
        build, compact = self._patches()
        with build, compact:
            async with app.run_test() as pilot:
                pane = app.pane
                await self._start(app, pilot)

                pane.handle_submit(PromptInput.Submitted("go"))
                await pilot.pause()
                self.assertEqual(pane._queue, ["go"], "queued behind it, not run beside it")
                self.assertNotIn("go", self._prompts(pane.agent.history))

                self.gate.set()
                await self._wait_for(pilot, lambda: not pane.is_busy
                                     and "go" in self._prompts(pane.agent.history))
                await self._wait_for(pilot, lambda: not pane.is_busy)

                history = pane.agent.history
                self.assertEqual(synthetic_kind(history[0]), SUMMARY)
                self.assertEqual(self._prompts(history)[-1], "go")
                self.assertIsInstance(history[-1], ModelResponse, "and it was answered")
                self.assert_replays_to(pane)
                self.assertIn("Context compacted", self._log_text(pane))

    async def test_a_shell_run_that_ends_during_it_is_recorded_after_it(self) -> None:
        app = self.make_app(session=self._old_session("earlier"))
        build, compact = self._patches()
        with build, compact:
            async with app.run_test() as pilot:
                pane = app.pane
                await self._start(app, pilot)

                pane.run_user_command("echo hi")
                await self._wait_for(pilot, lambda: pane._shell_worker is None)
                self.assertFalse(any(synthetic_kind(m) == SHELL for m in pane.agent.history),
                                 "held while the compaction is out")

                self.gate.set()
                await self._wait_for(pilot, lambda: not pane.is_busy)
                history = pane.agent.history
                self.assertEqual(synthetic_kind(history[0]), SUMMARY)
                self.assertEqual(synthetic_kind(history[-1]), SHELL, "and it survived it")
                self.assert_replays_to(pane)

    async def test_escape_stops_it_and_leaves_the_session_as_it_was(self) -> None:
        app = self.make_app(session=self._old_session("earlier"))
        build, compact = self._patches()
        with build, compact:
            async with app.run_test() as pilot:
                pane = app.pane
                history = list(pane.agent.history)
                log = pane.agent.session.path.read_bytes()
                await self._start(app, pilot)

                await pilot.press("escape")
                await self._wait_for(pilot, lambda: not pane.is_busy)
                self.assertIn("Compaction stopped", self._log_text(pane))
                self.assertEqual(self._status(app), "")
                self.assertEqual(pane.agent.history, history)
                self.assertEqual(pane.agent.session.path.read_bytes(), log)

                # And the pane is usable: the next prompt runs.
                pane.handle_submit(PromptInput.Submitted("go"))
                await self._wait_for(pilot, lambda: "go" in self._prompts(pane.agent.history))
                await self._wait_for(pilot, lambda: not pane.is_busy)

    async def test_a_failure_is_shown_and_the_queue_still_goes_out(self) -> None:
        async def broken(messages, **kwargs):
            self.asked.set()
            await self.gate.wait()
            raise compaction.CompactionError("the summary came back empty")

        app = self.make_app(session=self._old_session("earlier"))
        with patch("paimon.agent.build_model", return_value=stub_model()), \
                patch("paimon.agent.compaction.compact", side_effect=broken):
            async with app.run_test() as pilot:
                pane = app.pane
                await self._start(app, pilot)
                pane.handle_submit(PromptInput.Submitted("go"))
                self.gate.set()
                await self._wait_for(pilot, lambda: "go" in self._prompts(pane.agent.history))
                await self._wait_for(pilot, lambda: not pane.is_busy)
                self.assertIn("Compaction failed", self._log_text(pane))
                self.assertIn("the summary came back empty", self._log_text(pane))

    async def test_busy_refuses_it(self) -> None:
        from tests.support.app import hold_turn

        app = self.make_app()
        async with app.run_test() as pilot:
            hold_turn(app.pane)
            app.action_compact()
            await pilot.pause()
            self.assertIn("compact the context after this turn", self._log_text(app.pane))
