import asyncio
import unittest

from paimon.agent import UserInput
from paimon.jobs import AgentJob, CommandJob, Outcome, State, TurnOver
from tests.support.jobs import FakeAgent, FakeCommand, settle


class AgentJobTest(unittest.IsolatedAsyncioTestCase):
    def make(self) -> AgentJob:
        self.agent = FakeAgent()
        job = AgentJob("a1f2", self.agent)
        job.start()
        return job

    async def finish(self, job: AgentJob) -> None:
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

    async def test_every_change_listener_is_notified(self) -> None:
        job = self.make()
        first: list = []
        second: list = []
        job.on_change.append(first.append)
        job.on_change.append(second.append)

        job.submit("go")
        self.assertEqual(first, [job])
        self.assertEqual(second, [job], "one listener does not shadow another")

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
        self.assertIs(job.state, State.RUNNING)

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
        self.assertIs(job.state, State.IDLE, "interrupted, not failed and not gone")

        job.submit("second")
        await settle()
        self.assertEqual(self.agent.prompts, ["first", "second"])

    async def test_a_turn_that_raises_is_reported_not_propagated(self) -> None:
        job = self.make()
        self.agent.fail = "the model said no"
        job.submit("go")
        await self.finish(job)

        self.assertIs(job.state, State.FAILED)
        self.assertEqual(job.result.error, "the model said no")
        self.assertFalse(job.result.finished)

    async def test_cancel_ends_the_driver_and_keeps_the_text(self) -> None:
        job = self.make()
        job.submit("go")
        await self.finish(job)
        job.cancel()

        self.assertIs(job.state, State.KILLED)
        self.assertEqual(job.read(object(), mode="all").text, "done")
        job.submit("more")
        await settle()
        self.assertEqual(self.agent.prompts, ["go"], "a killed job runs nothing else")

    async def test_a_killed_job_holds_text_rather_than_messages(self) -> None:
        """Otherwise a long-dead agent keeps its whole history for the session."""
        job = self.make()
        job.submit("go")
        await self.finish(job)
        job.cancel()

        self.assertEqual(job._final, ["done"])
        self.assertTrue(all(isinstance(block, str) for block in job._final))

    async def test_reading_is_per_caller_and_incremental(self) -> None:
        job = self.make()
        one, two = object(), object()
        job.submit("go")
        await self.finish(job)

        self.assertEqual(job.read(one).text, "done")
        self.assertEqual(job.read(one).text, "", "nothing new since that caller's last read")
        self.assertEqual(job.read(two).text, "done", "another caller starts from the beginning")

    async def test_a_cursor_survives_the_job_being_killed(self) -> None:
        job = self.make()
        caller = object()
        job.submit("first")
        await self.finish(job)
        self.assertEqual(job.read(caller).text, "done")

        self.agent.answer = "and then this"
        job.submit("second")
        await self.finish(job)
        job.cancel()
        self.assertEqual(job.read(caller).text, "and then this",
                         "what was already read is not sent again")

    async def test_reasoning_never_comes_back(self) -> None:
        job = self.make()
        job.submit("go")
        await self.finish(job)
        self.assertNotIn("secret reasoning", job.read(object(), mode="all").text)

    async def test_the_sink_sees_the_prompt_and_the_end_of_the_turn(self) -> None:
        job = self.make()
        seen: list = []

        async def sink(event) -> None:
            seen.append(event)

        job.sink = sink
        job.submit("go")
        await self.finish(job)

        self.assertEqual(getattr(seen[0], "text", None), "go",
                         "a turn the supervisor started still renders its prompt")
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

    async def test_waiting_returns_when_the_turn_ends(self) -> None:
        job = self.make()
        job.submit("go")
        await settle()

        async def finish_soon() -> None:
            await settle(2)
            self.agent.finish()

        asyncio.ensure_future(finish_soon())
        self.assertIs(await job.wait(5), State.IDLE)

    async def test_a_wait_always_comes_back(self) -> None:
        job = self.make()
        job.submit("go")
        await settle()
        self.assertIs(await job.wait(0.01), State.RUNNING)

    async def test_a_pending_confirmation_ends_the_wait_early(self) -> None:
        job = self.make()
        job.submit("go")
        await settle()

        async def block_soon() -> None:
            await settle(2)
            job.mark_blocked(True)

        asyncio.ensure_future(block_soon())
        self.assertIs(await job.wait(5), State.NEEDS_CONFIRM)
        job.mark_blocked(False)
        self.assertIs(job.state, State.RUNNING)


class CommandJobTest(unittest.IsolatedAsyncioTestCase):
    def make(self) -> tuple[CommandJob, FakeCommand]:
        command = FakeCommand("npm run dev")
        job = CommandJob("b3c4", command, "dev server")
        job.start()
        return job, command

    async def test_a_running_command_is_read_incrementally(self) -> None:
        job, command = self.make()
        caller = object()
        command.output.append(b"listening on 3000\n")

        self.assertIs(job.state, State.RUNNING)
        self.assertIn("listening on 3000", job.read(caller).text)
        self.assertEqual(job.read(caller).text, "", "a second read only gets what is new")

    async def test_an_exit_code_decides_done_from_failed(self) -> None:
        job, command = self.make()
        command.exit(0)
        await settle()
        self.assertIs(job.state, State.DONE)
        self.assertIs(job.result.outcome, Outcome.SUCCESS)

        other, failing = self.make()
        failing.exit(2)
        await settle()
        self.assertIs(other.state, State.FAILED)
        self.assertEqual(other.exit_code, 2)

    async def test_it_takes_no_instructions(self) -> None:
        job, _ = self.make()
        self.assertFalse(job.submit("please stop"))

    async def test_cancel_kills_the_group_and_keeps_the_output(self) -> None:
        job, command = self.make()
        command.output.append(b"partial\n")
        job.cancel()
        await settle()

        self.assertTrue(command.killed)
        self.assertIs(job.state, State.KILLED)
        self.assertIn("partial", job.read(object(), mode="all").text)

    async def test_waiting_returns_when_the_command_exits(self) -> None:
        job, command = self.make()

        async def exit_soon() -> None:
            await settle(2)
            command.exit(0)

        asyncio.ensure_future(exit_soon())
        self.assertIs(await job.wait(5), State.DONE)
