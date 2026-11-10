import unittest

from paimon.agent import UserInput
from paimon.turns import CompactionOver, CompactionStarted, Outcome, TurnDriver, TurnOver
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

    # ---- waking on its own --------------------------------------------------

    async def test_a_notice_that_arrives_while_idle_wakes_the_agent(self) -> None:
        job = self.make()
        self.agent.report("agent a1f2 finished:\nfound it")
        self.assertTrue(job.is_busy, "the wake-up is queued the moment the notice lands")
        await self.finish(job)
        self.assertEqual(self.agent.prompts, [None])

    async def test_a_notice_mid_turn_waits_for_the_turn_and_then_wakes_it(self) -> None:
        job = self.make()
        job.submit("go")
        await settle()
        self.agent.report("agent a1f2 finished:\nfound it")
        await settle()
        self.assertEqual(self.agent.prompts, ["go"], "picked up between steps, not as a turn")

        # The fake takes notices only as a turn starts, so this one is still
        # waiting once the turn is over, as one from its last step would be.
        await self.finish(job)
        self.assertEqual(self.agent.prompts, ["go", None])

    async def test_a_turn_that_did_not_finish_does_not_wake(self) -> None:
        for stop in ("interrupt", "fail"):
            with self.subTest(stop):
                job = self.make()
                job.submit("go")
                await settle()
                self.agent.notices.append("agent a1f2 finished:\nfound it")
                if stop == "interrupt":
                    job.interrupt()
                else:
                    self.agent.fail = "boom"
                    self.agent.finish()
                await settle()
                self.assertFalse(job.is_busy, "the notice waits for whatever runs next")
                self.assertEqual(self.agent.prompts, ["go"])
                job.cancel()

    async def test_a_queued_prompt_wins_over_the_wake_up(self) -> None:
        job = self.make()

        async def sink(event) -> None:
            # What the pane does with a queue at the end of a finished turn.
            if isinstance(event, TurnOver):
                job.submit("queued meanwhile")

        job.sink = sink
        job.submit("go")
        await settle()
        self.agent.notices.append("agent a1f2 finished:\nfound it")
        await self.finish(job)
        await settle()
        self.assertEqual(self.agent.prompts, ["go", "queued meanwhile"],
                         "the next turn reports the notice anyway")

    async def test_a_notice_during_a_compaction_waits_for_it(self) -> None:
        job = self.make()
        self.assertTrue(job.compact())
        await settle()
        self.agent.report("agent a1f2 finished:\nfound it")
        await settle()
        self.assertEqual(self.agent.prompts, [], "nothing runs under a compaction")

        self.agent.finish()
        await settle()
        self.assertEqual(self.agent.prompts, [None], "reported once it is over")

    async def test_a_cancelled_driver_does_not_wake(self) -> None:
        job = self.make()
        job.cancel()
        self.agent.report("agent a1f2 finished:\nfound it")
        self.assertFalse(job.is_busy)

    # ---- compaction as an item ----------------------------------------------

    async def test_a_compaction_is_an_item_like_a_turn(self) -> None:
        job = self.make()
        seen: list = []

        async def sink(event) -> None:
            seen.append(event)

        job.sink = sink
        self.assertTrue(job.compact())
        job.submit("go")
        await settle()
        self.assertTrue(job.is_busy)
        self.assertEqual(self.agent.compactions, 1)
        self.assertEqual(self.agent.prompts, [], "the prompt waits for the compaction")

        await self.finish(job)
        self.assertEqual(self.agent.prompts, ["go"])
        self.assertIsInstance(seen[0], CompactionStarted)
        over = seen[1]
        self.assertIsInstance(over, CompactionOver)
        self.assertIs(over.result.outcome, Outcome.SUCCESS)
        self.assertIsNone(over.compacted, "the fake had nothing to compact")
        self.assertIsNone(job.result, "a compaction is not the last turn")

    async def test_a_compaction_is_refused_while_busy(self) -> None:
        job = self.make()
        job.submit("go")
        self.assertFalse(job.compact())

    async def test_an_interrupted_compaction_says_so_and_frees_the_driver(self) -> None:
        job = self.make()
        seen: list = []

        async def sink(event) -> None:
            seen.append(event)

        job.sink = sink
        job.compact()
        await settle()
        job.interrupt()
        await settle()
        self.assertIs(seen[-1].result.outcome, Outcome.INTERRUPTED)
        self.assertFalse(job.is_busy)

    async def test_a_failed_compaction_is_reported_not_propagated(self) -> None:
        job = self.make()
        seen: list = []

        async def sink(event) -> None:
            seen.append(event)

        job.sink = sink
        self.agent.fail = "summary request failed"
        job.compact()
        await self.finish(job)
        self.assertIs(seen[-1].result.outcome, Outcome.FAILED)
        self.assertEqual(seen[-1].result.error, "summary request failed")
        self.assertFalse(job.is_busy)

    # ---- "!" runs -----------------------------------------------------------

    async def test_a_shell_run_while_idle_is_recorded_at_once(self) -> None:
        job = self.make()
        job.record_shell("ls", "a.txt")
        self.assertEqual(self.agent.shell_runs, [("ls", "a.txt")])

    async def test_a_shell_run_during_an_item_is_held_until_it_is_over(self) -> None:
        for item in ("turn", "compaction"):
            with self.subTest(item):
                job = self.make()
                job.submit("go") if item == "turn" else job.compact()
                await settle()
                job.record_shell("ls", "a.txt")
                self.assertEqual(self.agent.shell_runs, [])
                await self.finish(job)
                self.assertEqual(self.agent.shell_runs, [("ls", "a.txt")])
                job.cancel()

    async def test_a_running_turn_takes_held_runs_before_queued_prompts(self) -> None:
        job = self.make()
        job.take_queued = lambda: ["and now this"]
        job.submit("go")
        await settle()
        job.record_shell("ls", "a.txt")
        pending = self.agent.pending()
        self.assertEqual(len(pending), 2)
        self.assertEqual(pending[1], "and now this")
        self.assertEqual(self.agent.pending(), ["and now this"], "a run is taken once")
        await self.finish(job)
        self.assertEqual(self.agent.shell_runs, [], "the turn took it, so nothing is left over")
