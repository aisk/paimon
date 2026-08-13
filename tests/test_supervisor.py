"""The agent pool, driven without a terminal.

Everything here runs against a fake runner: delivery, cursors and cleanup are
where concurrency bugs live, and reproducing those by clicking around a TUI is
not a debugging strategy.
"""

import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart

from paimon.supervisor import AgentState, Supervisor, SupervisorError, TaskState
from paimon.tools import _TaskOutput


class FakeCaller:
    """Stands in for the Agent that starts things.

    The pool only ever reads its identity — cursors and ownership are keyed on
    it — and, for a background command, where it works.
    """

    def __init__(self, cwd: Path = Path(".")) -> None:
        self.cwd = cwd


class FakeAgent:
    def __init__(self) -> None:
        self.history: list = []
        self.supervisor = None
        self.cwd = Path(".")


class FakeCommand:
    """A tools.BackgroundCommand, minus the process."""

    def __init__(self, command: str = "sleep 30") -> None:
        self.command = command
        self.output = _TaskOutput()
        self.exit_code = None
        self.killed = False

    @property
    def running(self) -> bool:
        return self.exit_code is None

    def kill(self) -> None:
        self.killed = True
        self.exit_code = -15

    def exit(self, code: int = 0) -> None:
        self.exit_code = code


class FakeRunner:
    """A SessionPane's contract, minus the pane."""

    def __init__(self) -> None:
        self.agent = FakeAgent()
        self.busy = False
        self.needs_confirm = False
        self.turn_failed = False
        self.prompts: list[str] = []
        self.closed = False

    @property
    def is_busy(self) -> bool:
        return self.busy

    def start_turn(self, text: str) -> None:
        assert not self.busy, "a second turn on a busy agent would cancel the first"
        self.prompts.append(text)
        self.busy = True

    def finish(self, answer: str = "done") -> None:
        self.agent.history.append(ModelResponse(parts=[
            ThinkingPart(content="secret reasoning"), TextPart(content=answer)]))
        self.busy = False


class SupervisorTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.runners: list[FakeRunner] = []
        self.task_runners: list = []
        self.closed: list = []
        self.parent = FakeCaller()

    def make(self, limit: int = 4, launch=None) -> Supervisor:
        async def default_launch(agent_id, parent, model):
            runner = FakeRunner()
            runner.agent_id = agent_id
            runner.model = model
            self.runners.append(runner)
            return runner

        async def launch_task(task_id, command, description):
            runner = SimpleNamespace(task_id=task_id, command=command,
                                     description=description)
            self.task_runners.append(runner)
            return runner

        return Supervisor(launch=launch or default_launch,
                          close=self.closed.append, limit=limit,
                          launch_task=launch_task, close_task=self.closed.append)

    async def start_task(self, supervisor, command: str = "npm run dev",
                         description: str = "dev server", parent=None) -> tuple:
        """A task backed by a fake command, so no process is ever started."""
        running = FakeCommand(command)
        with patch("paimon.supervisor.start_background",
                   new=AsyncMock(return_value=running)):
            task_id = await supervisor.start_task(
                command, description, parent=parent or self.parent, cwd=Path("."))
        return task_id, running


class SpawnTest(SupervisorTestCase):
    async def test_spawn_starts_a_turn_and_reports_running(self) -> None:
        supervisor = self.make()
        agent_id = await supervisor.spawn("look at the parser", parent=self.parent, model="x:y")

        runner = self.runners[0]
        self.assertEqual(runner.prompts, ["look at the parser"])
        self.assertEqual(runner.model, "x:y")
        self.assertEqual(supervisor.states(), {agent_id: AgentState.RUNNING})

    async def test_the_limit_refuses_instead_of_queueing(self) -> None:
        supervisor = self.make(limit=2)
        await supervisor.spawn("one", parent=self.parent)
        await supervisor.spawn("two", parent=self.parent)

        with self.assertRaises(SupervisorError):
            await supervisor.spawn("three", parent=self.parent)
        self.assertEqual(len(self.runners), 2)

    async def test_a_killed_agent_frees_its_slot(self) -> None:
        supervisor = self.make(limit=1)
        first = await supervisor.spawn("one", parent=self.parent)
        supervisor.kill(first)
        await supervisor.spawn("two", parent=self.parent)
        self.assertEqual(len(self.runners), 2)

    async def test_an_empty_prompt_is_refused_before_a_pane_is_opened(self) -> None:
        supervisor = self.make()
        with self.assertRaises(SupervisorError):
            await supervisor.spawn("   ", parent=self.parent)
        self.assertEqual(self.runners, [])

    async def test_a_failed_launch_is_reported_not_raised(self) -> None:
        async def launch(agent_id, parent, model):
            raise RuntimeError("no pane for you")

        supervisor = self.make(launch=launch)
        result = await supervisor.handle("spawn_agent", {"prompt": "go"}, caller=self.parent)
        self.assertIn("could not start an agent", result)
        self.assertEqual(supervisor.states(), {}, "a launch that failed leaves no record")


class DeliveryTest(SupervisorTestCase):
    async def test_a_busy_agent_queues_and_gets_one_prompt_per_turn(self) -> None:
        supervisor = self.make()
        agent_id = await supervisor.spawn("first", parent=self.parent)
        runner = self.runners[0]

        first = await supervisor.send(agent_id, "second", caller=self.parent)
        second = await supervisor.send(agent_id, "third", caller=self.parent)
        self.assertTrue(first.queued and second.queued)
        self.assertEqual(runner.prompts, ["first"])

        runner.finish()
        supervisor.pump()
        self.assertEqual(runner.prompts, ["first", "second"],
                         "queued prompts stay separate turns")
        self.assertEqual(supervisor.states()[agent_id], AgentState.RUNNING,
                         "an agent with a full inbox is not idle")

        runner.finish()
        supervisor.pump()
        self.assertEqual(runner.prompts, ["first", "second", "third"])

    async def test_an_idle_agent_starts_the_turn_at_once(self) -> None:
        supervisor = self.make()
        agent_id = await supervisor.spawn("first", parent=self.parent)
        self.runners[0].finish()

        delivery = await supervisor.send(agent_id, "again", caller=self.parent)
        self.assertFalse(delivery.queued)
        self.assertEqual(self.runners[0].prompts, ["first", "again"])

    async def test_sending_to_a_stranger_is_refused(self) -> None:
        supervisor = self.make()
        agent_id = await supervisor.spawn("first", parent=self.parent)

        delivery = await supervisor.send(agent_id, "hello", caller=object())
        self.assertFalse(delivery.accepted)
        self.assertIs(delivery.state, AgentState.UNKNOWN)


class ReadTest(SupervisorTestCase):
    async def test_each_caller_reads_only_what_is_new_to_it(self) -> None:
        supervisor = self.make()
        agent_id = await supervisor.spawn("go", parent=self.parent)
        runner = self.runners[0]
        runner.finish("first answer")

        self.assertEqual(supervisor.read(agent_id, caller=self.parent).text, "first answer")
        self.assertEqual(supervisor.read(agent_id, caller=self.parent).text, "",
                         "nothing new since the last read")

        runner.finish("second answer")
        view = supervisor.read(agent_id, caller=self.parent)
        self.assertEqual(view.text, "second answer")
        self.assertFalse(view.complete)

        everything = supervisor.read(agent_id, caller=self.parent, mode="all")
        self.assertEqual(everything.text, "first answer\n\nsecond answer")
        self.assertTrue(everything.complete)

    async def test_reasoning_never_comes_back(self) -> None:
        supervisor = self.make()
        agent_id = await supervisor.spawn("go", parent=self.parent)
        self.runners[0].finish("just the answer")
        self.assertNotIn("secret reasoning", supervisor.read(agent_id, caller=self.parent).text)

    async def test_a_stale_id_answers_instead_of_raising(self) -> None:
        supervisor = self.make()
        result = await supervisor.handle("read_agent", {"agent_id": "beef"}, caller=self.parent)
        self.assertIn("no agent beef", result)

    async def test_a_killed_agents_output_is_still_readable(self) -> None:
        supervisor = self.make()
        agent_id = await supervisor.spawn("go", parent=self.parent)
        self.runners[0].finish("what it managed")
        supervisor.released(self.runners[0])

        view = supervisor.read(agent_id, caller=self.parent, mode="all")
        self.assertIs(view.state, AgentState.KILLED)
        self.assertEqual(view.text, "what it managed")


class WaitTest(SupervisorTestCase):
    async def test_waiting_returns_when_the_turn_ends(self) -> None:
        supervisor = self.make()
        agent_id = await supervisor.spawn("go", parent=self.parent)

        async def finish_soon() -> None:
            await asyncio.sleep(0)
            self.runners[0].finish()
            supervisor.pump()

        asyncio.ensure_future(finish_soon())
        self.assertIs(await supervisor.wait(agent_id, caller=self.parent, timeout=5),
                      AgentState.IDLE)

    async def test_a_wait_always_comes_back(self) -> None:
        """Without this the caller hangs on an agent nobody is looking at."""
        supervisor = self.make()
        agent_id = await supervisor.spawn("go", parent=self.parent)
        self.assertIs(await supervisor.wait(agent_id, caller=self.parent, timeout=0.01),
                      AgentState.RUNNING)

    async def test_a_pending_confirmation_ends_the_wait_early(self) -> None:
        supervisor = self.make()
        agent_id = await supervisor.spawn("go", parent=self.parent)

        async def block_soon() -> None:
            await asyncio.sleep(0)
            self.runners[0].needs_confirm = True
            supervisor.pump()

        asyncio.ensure_future(block_soon())
        state = await supervisor.wait(agent_id, caller=self.parent, timeout=5)
        self.assertIs(state, AgentState.NEEDS_CONFIRM)
        answer = await supervisor.handle("wait_for_agent", {"agent_id": agent_id, "timeout": 5},
                                         caller=self.parent)
        self.assertIn("confirm", answer)

    async def test_the_timeout_is_clamped(self) -> None:
        supervisor = self.make()
        agent_id = await supervisor.spawn("go", parent=self.parent)
        self.runners[0].finish()
        # A missing or absurd timeout must not become "forever".
        for value in (None, "nonsense", -1, 10 ** 9):
            answer = await supervisor.handle(
                "wait_for_agent", {"agent_id": agent_id, "timeout": value}, caller=self.parent)
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
        gone = await supervisor.handle("read_agent", {"agent_id": first}, caller=self.parent)
        self.assertIn("no agent", gone)

    async def test_a_killed_agent_stops_taking_deliveries(self) -> None:
        supervisor = self.make()
        agent_id = await supervisor.spawn("go", parent=self.parent)
        supervisor.kill(agent_id)

        delivery = await supervisor.send(agent_id, "more", caller=self.parent)
        self.assertFalse(delivery.accepted)
        self.assertIs(delivery.state, AgentState.KILLED)
        self.assertEqual(self.runners[0].prompts, ["go"])


class StatusSummaryTest(SupervisorTestCase):
    async def test_news_is_reported_once_per_change(self) -> None:
        supervisor = self.make()
        agent_id = await supervisor.spawn("go", parent=self.parent)

        self.assertIsNone(supervisor.status_summary(self.parent), "still running is not news")

        self.runners[0].finish()
        self.assertEqual(supervisor.status_summary(self.parent), f"{agent_id} finished")
        self.assertIsNone(supervisor.status_summary(self.parent), "said once, not every turn")

        await supervisor.send(agent_id, "more", caller=self.parent)
        self.assertIsNone(supervisor.status_summary(self.parent))
        self.runners[0].finish()
        self.assertEqual(supervisor.status_summary(self.parent), f"{agent_id} finished")

    async def test_a_failure_and_a_pending_confirmation_are_reported(self) -> None:
        supervisor = self.make()
        first = await supervisor.spawn("one", parent=self.parent)
        second = await supervisor.spawn("two", parent=self.parent)

        self.runners[0].busy = False
        self.runners[0].turn_failed = True
        self.runners[1].needs_confirm = True

        self.assertEqual(supervisor.status_summary(self.parent),
                         f"{first} failed · {second} needs confirmation")

    async def test_only_the_parents_own_agents_are_reported(self) -> None:
        supervisor = self.make()
        await supervisor.spawn("go", parent=object())
        self.runners[0].finish()
        self.assertIsNone(supervisor.status_summary(self.parent))


class BackgroundTaskTest(SupervisorTestCase):
    """Background commands: started, read incrementally, stopped."""

    async def test_a_task_opens_a_pane_and_is_read_from_the_start(self) -> None:
        supervisor = self.make()
        task_id, running = await self.start_task(supervisor)
        running.output.append(b"listening on 3000\n")

        self.assertEqual(supervisor.task_states(), {task_id: TaskState.RUNNING})
        self.assertEqual(self.task_runners[0].task_id, task_id)
        view = supervisor.read_task(task_id, caller=self.parent)
        self.assertIn("listening on 3000", view.text)
        self.assertEqual(supervisor.read_task(task_id, caller=self.parent).text, "",
                         "a second read only gets what is new")

    async def test_reading_all_starts_over(self) -> None:
        supervisor = self.make()
        task_id, running = await self.start_task(supervisor)
        running.output.append(b"first\n")
        supervisor.read_task(task_id, caller=self.parent)
        running.output.append(b"second\n")

        view = supervisor.read_task(task_id, caller=self.parent, mode="all")
        self.assertIn("first", view.text)
        self.assertIn("second", view.text)

    async def test_an_exit_code_is_reported_and_the_output_survives(self) -> None:
        supervisor = self.make()
        task_id, running = await self.start_task(supervisor)
        running.output.append(b"boom\n")
        running.exit(2)

        answer = await supervisor.handle("read_task", {"task_id": task_id}, caller=self.parent)
        self.assertIn("exited, code 2", answer)
        self.assertIn("boom", answer)

    async def test_another_caller_gets_nothing(self) -> None:
        supervisor = self.make()
        task_id, _ = await self.start_task(supervisor)

        view = supervisor.read_task(task_id, caller=object())
        self.assertIs(view.state, TaskState.UNKNOWN)
        self.assertFalse(supervisor.kill_task(task_id, caller=object()))

    async def test_killing_stops_the_command_and_closes_its_pane(self) -> None:
        supervisor = self.make()
        task_id, running = await self.start_task(supervisor)
        running.output.append(b"partial\n")

        answer = await supervisor.handle("kill_task", {"task_id": task_id}, caller=self.parent)
        self.assertIn("Stopped", answer)
        self.assertTrue(running.killed)
        self.assertEqual(self.closed, [self.task_runners[0]])
        view = supervisor.read_task(task_id, caller=self.parent, mode="all")
        self.assertIs(view.state, TaskState.KILLED)
        self.assertIn("partial", view.text, "what it printed is still readable")

    async def test_a_closed_pane_stops_the_command(self) -> None:
        supervisor = self.make()
        task_id, running = await self.start_task(supervisor)

        supervisor.released(self.task_runners[0])
        self.assertTrue(running.killed)
        self.assertEqual(self.closed, [], "the pane closed itself; it is not closed again")

    async def test_changing_the_session_stops_agents_and_tasks_alike(self) -> None:
        supervisor = self.make()
        agent_id = await supervisor.spawn("look at the parser", parent=self.parent)
        task_id, running = await self.start_task(supervisor)

        killed = supervisor.kill_children(self.parent)
        self.assertEqual(sorted(killed), sorted([agent_id, task_id]))
        self.assertTrue(running.killed)
        self.assertEqual(supervisor.task_states(), {})

    async def test_tasks_and_agents_share_the_pane_budget(self) -> None:
        supervisor = self.make(limit=2)
        await supervisor.spawn("one", parent=self.parent)
        await self.start_task(supervisor)

        with self.assertRaises(SupervisorError):
            await self.start_task(supervisor)
        result = await supervisor.handle(
            "run_background", {"command": "sleep 1", "description": "x"}, caller=self.parent)
        self.assertIn("limit 2", result, "the model is told why, not raised at")

    async def test_a_stale_id_is_an_error_not_an_exception(self) -> None:
        supervisor = self.make()
        answer = await supervisor.handle("read_task", {"task_id": "beef"}, caller=self.parent)
        self.assertIn("no background task beef", answer)

    async def test_what_became_of_a_task_opens_the_next_turn(self) -> None:
        supervisor = self.make()
        task_id, running = await self.start_task(supervisor)
        self.assertIsNone(supervisor.status_summary(self.parent), "still running is not news")

        running.exit(1)
        self.assertEqual(supervisor.status_summary(self.parent), f"{task_id} exited (code 1)")
        self.assertIsNone(supervisor.status_summary(self.parent), "said once, not every turn")


if __name__ == "__main__":
    unittest.main()
