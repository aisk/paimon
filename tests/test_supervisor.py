"""The job pool, driven without a terminal.

Everything here runs against real jobs over a fake agent and a fake command:
ownership, delivery, cursors and cleanup are where concurrency bugs live, and
reproducing those by clicking around a TUI is not a debugging strategy.
"""

import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from paimon.jobs import AgentJob, CommandJob, State
from paimon.supervisor import Supervisor, SupervisorError

from test_jobs import FakeAgent, FakeCommand, settle


class FakeCaller:
    """Stands in for the Agent that starts things.

    The pool only ever reads its identity — cursors and ownership are keyed on
    it — and, for a background command, where it works.
    """

    def __init__(self, cwd: Path = Path(".")) -> None:
        self.cwd = cwd


class SupervisorTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.jobs: list[AgentJob] = []
        self.commands: list[CommandJob] = []
        self.closed: list = []
        self.parent = FakeCaller()

    def make(self, limit: int = 4, launch=None) -> Supervisor:
        async def default_launch(job_id, parent, model, agent):
            job = AgentJob(job_id, FakeAgent(), parent=parent)
            job.model = model  # only so a test can see what was asked for
            job.agent_type = agent  # likewise
            self.jobs.append(job)
            job.start()
            return job

        async def launch_command(job_id, command, description):
            job = CommandJob(job_id, command, description, parent=self.parent)
            self.commands.append(job)
            job.start()
            return job

        return Supervisor(launch=launch or default_launch, close=self.closed.append,
                          limit=limit, launch_command=launch_command)

    async def finish(self, index: int = 0, answer: str = "done") -> None:
        """Let the agent's current turn reach its end."""
        agent = self.jobs[index].agent
        await settle()
        agent.answer = answer
        agent.finish()
        await settle()

    async def start_command(self, supervisor, command: str = "npm run dev",
                         description: str = "dev server", parent=None) -> tuple:
        """A command backed by a fake process, so nothing is ever started."""
        running = FakeCommand(command)
        with patch("paimon.supervisor.start_background",
                   new=AsyncMock(return_value=running)):
            job_id = await supervisor.start_command(
                command, description, parent=parent or self.parent, cwd=Path("."))
        return job_id, running


class SpawnTest(SupervisorTestCase):
    async def test_spawn_starts_a_turn_and_reports_running(self) -> None:
        supervisor = self.make()
        job_id = await supervisor.spawn("look at the parser", parent=self.parent, model="x:y")
        await settle()

        job = self.jobs[0]
        self.assertEqual(job.agent.prompts, ["look at the parser"])
        self.assertEqual(job.model, "x:y")
        self.assertEqual(supervisor.states(), {job_id: State.RUNNING})

    async def test_the_limit_refuses_instead_of_queueing(self) -> None:
        supervisor = self.make(limit=2)
        await supervisor.spawn("one", parent=self.parent)
        await supervisor.spawn("two", parent=self.parent)

        with self.assertRaises(SupervisorError):
            await supervisor.spawn("three", parent=self.parent)
        self.assertEqual(len(self.jobs), 2)

    async def test_a_stopped_agent_frees_its_slot(self) -> None:
        supervisor = self.make(limit=1)
        first = await supervisor.spawn("one", parent=self.parent)
        supervisor.stop(first)
        await supervisor.spawn("two", parent=self.parent)
        self.assertEqual(len(self.jobs), 2)

    async def test_a_user_opened_pane_counts_against_the_budget(self) -> None:
        """It holds a pane and an agent like any other, but nothing may read it."""
        supervisor = self.make(limit=2)
        mine = AgentJob(supervisor.new_id(), FakeAgent())
        supervisor.register(mine)
        await supervisor.spawn("one", parent=self.parent)

        with self.assertRaises(SupervisorError):
            await supervisor.spawn("two", parent=self.parent)
        self.assertIsNone(supervisor.read(mine.job_id, caller=self.parent),
                          "an unowned job is invisible to every caller")

    async def test_an_empty_prompt_is_refused_before_a_pane_is_opened(self) -> None:
        supervisor = self.make()
        with self.assertRaises(SupervisorError):
            await supervisor.spawn("   ", parent=self.parent)
        self.assertEqual(self.jobs, [])

    async def test_a_failed_launch_is_reported_not_raised(self) -> None:
        async def launch(job_id, parent, model, agent):
            raise RuntimeError("no pane for you")

        supervisor = self.make(launch=launch)
        result = await supervisor.handle("spawn_agent", {"prompt": "go"}, caller=self.parent)
        self.assertIn("could not start an agent", result)
        self.assertEqual(supervisor.states(), {}, "a launch that failed leaves no record")

    async def test_the_agent_type_reaches_the_launcher(self) -> None:
        supervisor = self.make()
        await supervisor.handle("spawn_agent", {"prompt": "go", "agent": "explore"},
                                caller=self.parent)
        self.assertEqual(self.jobs[0].agent_type, "explore")
        await supervisor.handle("spawn_agent", {"prompt": "go"}, caller=self.parent)
        self.assertIsNone(self.jobs[1].agent_type)

    async def test_an_unknown_agent_type_is_a_tool_error_with_no_residue(self) -> None:
        async def launch(job_id, parent, model, agent):
            raise SupervisorError(f"unknown agent type {agent!r}; available: explore")

        supervisor = self.make(launch=launch)
        result = await supervisor.handle(
            "spawn_agent", {"prompt": "go", "agent": "nope"}, caller=self.parent)
        self.assertIn("unknown agent type 'nope'", result)
        self.assertEqual(supervisor.states(), {})


class DeliveryTest(SupervisorTestCase):
    async def test_a_busy_agent_queues_and_gets_one_prompt_per_turn(self) -> None:
        supervisor = self.make()
        job_id = await supervisor.spawn("first", parent=self.parent)
        agent = self.jobs[0].agent
        await settle()

        first = supervisor.send(job_id, "second", caller=self.parent)
        second = supervisor.send(job_id, "third", caller=self.parent)
        self.assertTrue(first.queued and second.queued)
        self.assertEqual(agent.prompts, ["first"])

        await self.finish()
        self.assertEqual(agent.prompts, ["first", "second"],
                         "queued prompts stay separate turns")
        self.assertEqual(supervisor.states()[job_id], State.RUNNING,
                         "an agent with a full inbox is not idle")

        await self.finish()
        self.assertEqual(agent.prompts, ["first", "second", "third"])

    async def test_an_idle_agent_starts_the_turn_at_once(self) -> None:
        supervisor = self.make()
        job_id = await supervisor.spawn("first", parent=self.parent)
        await self.finish()

        delivery = supervisor.send(job_id, "again", caller=self.parent)
        await settle()
        self.assertFalse(delivery.queued)
        self.assertEqual(self.jobs[0].agent.prompts, ["first", "again"])

    async def test_sending_to_a_stranger_is_refused(self) -> None:
        supervisor = self.make()
        job_id = await supervisor.spawn("first", parent=self.parent)

        delivery = supervisor.send(job_id, "hello", caller=object())
        self.assertFalse(delivery.accepted)
        self.assertIs(delivery.state, State.UNKNOWN)

    async def test_a_background_command_takes_no_instructions(self) -> None:
        supervisor = self.make()
        job_id, _ = await self.start_command(supervisor)

        answer = await supervisor.handle(
            "send_to_agent", {"job_id": job_id, "prompt": "stop"}, caller=self.parent)
        self.assertIn("background command", answer)
        self.assertIn("read_job", answer)


class ReadTest(SupervisorTestCase):
    async def test_each_caller_reads_only_what_is_new_to_it(self) -> None:
        supervisor = self.make()
        job_id = await supervisor.spawn("go", parent=self.parent)
        await self.finish(answer="first answer")

        self.assertEqual(supervisor.read(job_id, caller=self.parent).text, "first answer")
        self.assertEqual(supervisor.read(job_id, caller=self.parent).text, "",
                         "nothing new since the last read")

        supervisor.send(job_id, "again", caller=self.parent)
        await self.finish(answer="second answer")
        view = supervisor.read(job_id, caller=self.parent)
        self.assertEqual(view.text, "second answer")
        self.assertFalse(view.complete)

        everything = supervisor.read(job_id, caller=self.parent, mode="all")
        self.assertEqual(everything.text, "first answer\n\nsecond answer")
        self.assertTrue(everything.complete)

    async def test_reasoning_never_comes_back(self) -> None:
        supervisor = self.make()
        job_id = await supervisor.spawn("go", parent=self.parent)
        await self.finish(answer="just the answer")
        self.assertNotIn("secret reasoning", supervisor.read(job_id, caller=self.parent).text)

    async def test_a_stale_id_answers_instead_of_raising(self) -> None:
        supervisor = self.make()
        result = await supervisor.handle("read_job", {"job_id": "beef"}, caller=self.parent)
        self.assertIn("beef", result)
        self.assertIn("Error", result)

    async def test_a_stopped_agents_output_is_still_readable(self) -> None:
        supervisor = self.make()
        job_id = await supervisor.spawn("go", parent=self.parent)
        await self.finish(answer="what it managed")
        supervisor.released(self.jobs[0])

        view = supervisor.read(job_id, caller=self.parent, mode="all")
        self.assertIs(view.state, State.KILLED)
        self.assertEqual(view.text, "what it managed")


class WaitTest(SupervisorTestCase):
    async def test_waiting_returns_when_the_turn_ends(self) -> None:
        supervisor = self.make()
        job_id = await supervisor.spawn("go", parent=self.parent)

        async def finish_soon() -> None:
            await settle(2)
            self.jobs[0].agent.finish()

        import asyncio
        asyncio.ensure_future(finish_soon())
        self.assertIs(await supervisor.wait(job_id, caller=self.parent, timeout=5),
                      State.IDLE)

    async def test_a_wait_always_comes_back(self) -> None:
        """Without this the caller hangs on a job nobody is looking at."""
        supervisor = self.make()
        job_id = await supervisor.spawn("go", parent=self.parent)
        self.assertIs(await supervisor.wait(job_id, caller=self.parent, timeout=0.01),
                      State.RUNNING)

    async def test_waiting_on_a_strangers_job_is_not_a_wait_at_all(self) -> None:
        supervisor = self.make()
        job_id = await supervisor.spawn("go", parent=self.parent)
        self.assertIs(await supervisor.wait(job_id, caller=object(), timeout=5),
                      State.UNKNOWN)

    async def test_a_pending_confirmation_ends_the_wait_early(self) -> None:
        supervisor = self.make()
        job_id = await supervisor.spawn("go", parent=self.parent)
        await settle()
        self.jobs[0].mark_blocked(True)

        state = await supervisor.wait(job_id, caller=self.parent, timeout=5)
        self.assertIs(state, State.NEEDS_CONFIRM)
        answer = await supervisor.handle("wait_for_job", {"job_id": job_id, "timeout": 5},
                                         caller=self.parent)
        self.assertIn("confirm", answer)

    async def test_a_background_command_can_be_waited_for(self) -> None:
        supervisor = self.make()
        job_id, running = await self.start_command(supervisor)
        running.exit(0)

        answer = await supervisor.handle("wait_for_job", {"job_id": job_id, "timeout": 5},
                                         caller=self.parent)
        self.assertIn("done", answer)

    async def test_the_timeout_is_clamped(self) -> None:
        supervisor = self.make()
        job_id = await supervisor.spawn("go", parent=self.parent)
        await self.finish()
        # A missing or absurd timeout must not become "forever".
        for value in (None, "nonsense", -1, 10 ** 9):
            answer = await supervisor.handle(
                "wait_for_job", {"job_id": job_id, "timeout": value}, caller=self.parent)
            self.assertIn("idle", answer)


class CleanupTest(SupervisorTestCase):
    async def test_killing_children_closes_them_and_forgets_the_ids(self) -> None:
        supervisor = self.make()
        first = await supervisor.spawn("one", parent=self.parent)
        second = await supervisor.spawn("two", parent=self.parent)
        other = object()
        third = await supervisor.spawn("three", parent=other)

        killed = supervisor.kill_children(self.parent)

        self.assertEqual(sorted(killed), sorted([first, second]))
        self.assertEqual(len(self.closed), 2)
        self.assertEqual(list(supervisor.states()), [third], "another pane's agent is untouched")
        gone = await supervisor.handle("read_job", {"job_id": first}, caller=self.parent)
        self.assertIn("Error", gone)

    async def test_a_stopped_agent_stops_taking_deliveries(self) -> None:
        supervisor = self.make()
        job_id = await supervisor.spawn("go", parent=self.parent)
        await settle()
        supervisor.stop(job_id)

        delivery = supervisor.send(job_id, "more", caller=self.parent)
        await settle()
        self.assertFalse(delivery.accepted)
        self.assertIs(delivery.state, State.KILLED)
        self.assertEqual(self.jobs[0].agent.prompts, ["go"])

    async def test_the_model_can_stop_an_agent_it_started(self) -> None:
        """The one thing it could not do before: call off a runaway subagent."""
        supervisor = self.make()
        job_id = await supervisor.spawn("go the wrong way", parent=self.parent)
        await self.finish(answer="halfway there")

        answer = await supervisor.handle("stop_job", {"job_id": job_id}, caller=self.parent)
        self.assertIn("Stopped", answer)
        self.assertEqual(self.closed, [self.jobs[0]])
        view = supervisor.read(job_id, caller=self.parent, mode="all")
        self.assertIs(view.state, State.KILLED)
        self.assertEqual(view.text, "halfway there")

    async def test_a_stranger_cannot_stop_anything(self) -> None:
        supervisor = self.make()
        job_id = await supervisor.spawn("go", parent=self.parent)
        self.assertFalse(supervisor.stop(job_id, caller=object()))
        self.assertEqual(self.closed, [])


class StatusSummaryTest(SupervisorTestCase):
    async def test_news_is_reported_once_per_change(self) -> None:
        supervisor = self.make()
        job_id = await supervisor.spawn("go", parent=self.parent)
        await settle()

        self.assertIsNone(supervisor.status_summary(self.parent), "still running is not news")

        await self.finish()
        self.assertEqual(supervisor.status_summary(self.parent), f"{job_id} finished")
        self.assertIsNone(supervisor.status_summary(self.parent), "said once, not every turn")

        supervisor.send(job_id, "more", caller=self.parent)
        self.assertIsNone(supervisor.status_summary(self.parent))
        await self.finish()
        self.assertEqual(supervisor.status_summary(self.parent), f"{job_id} finished")

    async def test_a_failure_and_a_pending_confirmation_are_reported(self) -> None:
        supervisor = self.make()
        first = await supervisor.spawn("one", parent=self.parent)
        second = await supervisor.spawn("two", parent=self.parent)

        self.jobs[0].agent.fail = "the model said no"
        await self.finish(0)
        self.jobs[1].mark_blocked(True)

        self.assertEqual(supervisor.status_summary(self.parent),
                         f"{first} failed · {second} needs confirmation")

    async def test_only_the_parents_own_agents_are_reported(self) -> None:
        supervisor = self.make()
        await supervisor.spawn("go", parent=object())
        await self.finish()
        self.assertIsNone(supervisor.status_summary(self.parent))


class BackgroundCommandTest(SupervisorTestCase):
    """Background commands: started, read incrementally, waited for, stopped."""

    async def test_a_command_opens_a_pane_and_is_read_from_the_start(self) -> None:
        supervisor = self.make()
        job_id, running = await self.start_command(supervisor)
        running.output.append(b"listening on 3000\n")

        self.assertEqual(supervisor.states(), {job_id: State.RUNNING})
        self.assertEqual(self.commands[0].job_id, job_id)
        view = supervisor.read(job_id, caller=self.parent)
        self.assertIn("listening on 3000", view.text)
        self.assertEqual(supervisor.read(job_id, caller=self.parent).text, "",
                         "a second read only gets what is new")

    async def test_reading_all_starts_over(self) -> None:
        supervisor = self.make()
        job_id, running = await self.start_command(supervisor)
        running.output.append(b"first\n")
        supervisor.read(job_id, caller=self.parent)
        running.output.append(b"second\n")

        view = supervisor.read(job_id, caller=self.parent, mode="all")
        self.assertIn("first", view.text)
        self.assertIn("second", view.text)

    async def test_an_exit_code_is_reported_and_the_output_survives(self) -> None:
        supervisor = self.make()
        job_id, running = await self.start_command(supervisor)
        running.output.append(b"boom\n")
        running.exit(2)
        await settle()

        answer = await supervisor.handle("read_job", {"job_id": job_id}, caller=self.parent)
        self.assertIn("exited, code 2", answer)
        self.assertIn("boom", answer)

    async def test_another_caller_gets_nothing(self) -> None:
        supervisor = self.make()
        job_id, _ = await self.start_command(supervisor)

        self.assertIsNone(supervisor.read(job_id, caller=object()))
        self.assertFalse(supervisor.stop(job_id, caller=object()))

    async def test_stopping_kills_the_command_and_closes_its_pane(self) -> None:
        supervisor = self.make()
        job_id, running = await self.start_command(supervisor)
        running.output.append(b"partial\n")

        answer = await supervisor.handle("stop_job", {"job_id": job_id}, caller=self.parent)
        self.assertIn("Stopped", answer)
        self.assertTrue(running.killed)
        self.assertEqual(self.closed, [self.commands[0]])
        view = supervisor.read(job_id, caller=self.parent, mode="all")
        self.assertIs(view.state, State.KILLED)
        self.assertIn("partial", view.text, "what it printed is still readable")

    async def test_a_closed_pane_stops_the_command(self) -> None:
        supervisor = self.make()
        _, running = await self.start_command(supervisor)

        supervisor.released(self.commands[0])
        self.assertTrue(running.killed)
        self.assertEqual(self.closed, [], "the pane closed itself; it is not closed again")

    async def test_changing_the_session_stops_agents_and_commands_alike(self) -> None:
        supervisor = self.make()
        agent_id = await supervisor.spawn("look at the parser", parent=self.parent)
        job_id, running = await self.start_command(supervisor)

        killed = supervisor.kill_children(self.parent)
        self.assertEqual(sorted(killed), sorted([agent_id, job_id]))
        self.assertTrue(running.killed)
        self.assertEqual(supervisor.states(), {})

    async def test_commands_and_agents_share_the_pane_budget_and_the_id_space(self) -> None:
        supervisor = self.make(limit=2)
        agent_id = await supervisor.spawn("one", parent=self.parent)
        job_id, _ = await self.start_command(supervisor)
        self.assertNotEqual(agent_id, job_id)

        with self.assertRaises(SupervisorError):
            await self.start_command(supervisor)
        result = await supervisor.handle(
            "run_background", {"command": "sleep 1", "description": "x"}, caller=self.parent)
        self.assertIn("limit 2", result, "the model is told why, not raised at")

    async def test_a_stale_id_is_an_error_not_an_exception(self) -> None:
        supervisor = self.make()
        answer = await supervisor.handle("read_job", {"job_id": "beef"}, caller=self.parent)
        self.assertIn("Error", answer)
        self.assertIn("beef", answer)

    async def test_what_became_of_a_command_opens_the_next_turn(self) -> None:
        supervisor = self.make()
        job_id, running = await self.start_command(supervisor)
        self.assertIsNone(supervisor.status_summary(self.parent), "still running is not news")

        running.exit(1)
        await settle()
        self.assertEqual(supervisor.status_summary(self.parent), f"{job_id} exited (code 1)")
        self.assertIsNone(supervisor.status_summary(self.parent), "said once, not every turn")


if __name__ == "__main__":
    unittest.main()
