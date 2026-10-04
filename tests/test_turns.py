import unittest

from paimon.agent import UserInput
from paimon.turns import Outcome, TurnDriver, TurnOver
from tests.support.turns import FakeAgent, settle


class TurnDriverTest(unittest.IsolatedAsyncioTestCase):
    def make(self) -> TurnDriver:
        self.agent = FakeAgent()
        job = TurnDriver(self.agent)
        job.start()
        return job

    async def finish(self, job: TurnDriver) -> None:
        await settle()
        self.agent.finish()
        await settle()

    async def test_wake_runs_a_turn_with_no_user_input(self) -> None:
        job = self.make()
        seen: list = []

        async def sink(event) -> None:
            seen.append(event)

        job.sink = sink
        self.assertTrue(job.wake())
        await self.finish(job)

        self.assertEqual(self.agent.prompts, [None], "the agent runs on None")
        self.assertNotIn(UserInput, [type(event) for event in seen],
                         "a wake-up is not rendered as something the user typed")

    async def test_wake_is_refused_while_busy_or_killed(self) -> None:
        job = self.make()
        job.submit("work")
        await settle()
        self.assertFalse(job.wake(), "busy: the agent reports at its next step")
        self.agent.finish()
        await settle()

        self.assertTrue(job.wake(), "idle again")
        self.assertFalse(job.wake(), "one queued wake-up is enough")
        await self.finish(job)

        job.cancel()
        self.assertFalse(job.wake())

    async def test_a_change_reaches_the_listener(self) -> None:
        job = self.make()
        seen: list = []
        job.on_change = lambda: seen.append(job.is_busy)

        job.submit("go")
        self.assertEqual(seen, [True])

    async def test_one_turn_at_a_time_in_the_order_submitted(self) -> None:
        job = self.make()
        job.submit("first")
        job.submit("second")
        await settle()
        self.assertEqual(self.agent.prompts, ["first"],
                         "the second prompt waits rather than sharing the history")

        self.agent.finish()
        await settle()
        self.assertEqual(self.agent.prompts, ["first", "second"])

    async def test_a_queued_prompt_keeps_the_job_busy(self) -> None:
        job = self.make()
        job.submit("first")
        job.submit("second")
        await settle()
        self.agent.finish()
        # The first turn is over but the second has not been picked up yet.
        self.assertTrue(job.is_busy)
        self.assertTrue(job.is_running)

    async def test_submitting_marks_busy_before_the_driver_wakes(self) -> None:
        """The window a Textual worker's PENDING state used to leave open."""
        job = self.make()
        job.submit("go")
        self.assertTrue(job.is_busy, "busy the moment the prompt is accepted")

    async def test_an_interrupt_stops_the_turn_and_keeps_the_agent(self) -> None:
        job = self.make()
        job.submit("first")
        await settle()
        job.interrupt()
        await settle()

        self.assertIs(job.result.outcome, Outcome.INTERRUPTED)
        self.assertFalse(job.is_busy, "interrupted, and ready for another")

        job.submit("second")
        await settle()
        self.assertEqual(self.agent.prompts, ["first", "second"])

    async def test_a_turn_that_raises_is_reported_not_propagated(self) -> None:
        job = self.make()
        self.agent.fail = "the model said no"
        job.submit("go")
        await self.finish(job)

        self.assertIs(job.result.outcome, Outcome.FAILED)
        self.assertEqual(job.result.error, "the model said no")
        self.assertFalse(job.result.finished)

    async def test_cancel_ends_the_driver(self) -> None:
        job = self.make()
        job.submit("go")
        await self.finish(job)
        job.cancel()

        self.assertTrue(job.killed)
        job.submit("more")
        await settle()
        self.assertEqual(self.agent.prompts, ["go"], "a cancelled driver runs nothing else")

    async def test_the_sink_sees_the_prompt_and_the_end_of_the_turn(self) -> None:
        job = self.make()
        seen: list = []

        async def sink(event) -> None:
            seen.append(event)

        job.sink = sink
        job.submit("go")
        await self.finish(job)

        self.assertEqual(getattr(seen[0], "text", None), "go",
                         "the prompt is rendered through the sink like any event")
        self.assertIsInstance(seen[-1], TurnOver)
        self.assertIs(seen[-1].result.outcome, Outcome.SUCCESS)

    async def test_an_interrupted_turn_still_reaches_the_sink(self) -> None:
        """The renderer has to be told, and a cancelled turn cannot tell it."""
        job = self.make()
        seen: list = []

        async def sink(event) -> None:
            seen.append(event)

        job.sink = sink
        job.submit("go")
        await settle()
        job.interrupt()
        await settle()

        self.assertIsInstance(seen[-1], TurnOver)
        self.assertIs(seen[-1].result.outcome, Outcome.INTERRUPTED)

    async def test_a_pending_confirmation_is_not_running(self) -> None:
        job = self.make()
        job.submit("go")
        await settle()
        job.mark_blocked(True)
        self.assertFalse(job.is_running, "stuck on the user, so no spinner")
        job.mark_blocked(False)
        self.assertTrue(job.is_running)
