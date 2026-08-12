"""The agent pool, driven without a terminal.

Everything here runs against a fake runner: delivery, cursors and cleanup are
where concurrency bugs live, and reproducing those by clicking around a TUI is
not a debugging strategy.
"""

import asyncio
import unittest

from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart

from paimon.supervisor import AgentState, Supervisor, SupervisorError


class FakeAgent:
    def __init__(self) -> None:
        self.history: list = []
        self.supervisor = None


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
        self.closed: list[FakeRunner] = []
        self.parent = object()

    def make(self, limit: int = 4, launch=None) -> Supervisor:
        async def default_launch(agent_id, parent, model):
            runner = FakeRunner()
            runner.agent_id = agent_id
            runner.model = model
            self.runners.append(runner)
            return runner

        return Supervisor(launch=launch or default_launch,
                          close=self.closed.append, limit=limit)


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


if __name__ == "__main__":
    unittest.main()
