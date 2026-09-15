import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from paimon.agent import (
    Agent,
    AgentsNotice,
    CompactionNotice,
    ReasoningDelta,
    RequestStats,
    TextDelta,
    TodosUpdate,
    ToolEnd,
    ToolStart,
    TurnEnd,
    UserInput,
    replay_events,
)
from paimon.config import Config
from paimon.session import (
    is_agents_message,
    is_shell_message,
    shell_message,
    shell_text,
    summary_message,
)
from tests.support.agent import FakeSupervisor, make_session, session_records, stub_model


class RequestStatsTest(unittest.IsolatedAsyncioTestCase):
    async def test_each_request_yields_stats_from_reported_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")

            with patch("paimon.agent.build_model",
                       return_value=stub_model("read_file", '{"path": "missing.txt"}')):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                events = [event async for event in agent.run("hi")]

            stats = [event for event in events if isinstance(event, RequestStats)]
            # one per model request: the tool-call response and the final text
            self.assertEqual(len(stats), 2)
            for stat in stats:
                self.assertGreater(stat.output_tokens, 0)
                self.assertGreater(stat.seconds, 0)


class PendingMessagesTest(unittest.IsolatedAsyncioTestCase):
    """Messages queued mid-turn reach the model at the next request, not the
    next turn."""

    async def test_queued_message_is_injected_before_the_next_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"), mode="yolo")

            queue = ["use uv instead"]

            def take() -> list[str]:
                texts, queue[:] = list(queue), []
                return texts

            agent.pending = take
            seen: list[list[object]] = []
            requests = 0

            async def stream(messages, info):
                nonlocal requests
                requests += 1
                seen.append(list(messages))
                if requests == 1:
                    yield {0: DeltaToolCall(name="shell", json_args='{"command": "echo hi"}',
                                            tool_call_id="call-1")}
                else:
                    yield "done"

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                events = [event async for event in agent.run("go")
                          if not isinstance(event, RequestStats)]

            # Queued before the turn even started, so the first request already
            # carries it, ahead of the tool call it goes on to make.
            self.assertEqual(
                [type(event) for event in events],
                [UserInput, ToolStart, ToolEnd, TextDelta, TurnEnd],
            )
            self.assertEqual(events[0].text, "use uv instead")

            prompts = [part.content for message in seen[0] if isinstance(message, ModelRequest)
                       for part in message.parts if isinstance(part, UserPromptPart)]
            self.assertEqual(prompts, ["go", "use uv instead"])

    async def test_a_message_queued_during_a_tool_call_lands_after_its_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"), mode="yolo")

            queue: list[str] = []
            agent.pending = lambda: [queue.pop(0)] if queue else []
            requests = 0

            async def stream(messages, info):
                nonlocal requests
                requests += 1
                if requests == 1:
                    # Typed while the tool below is running.
                    queue.append("stop, wrong file")
                    yield {0: DeltaToolCall(name="shell", json_args='{"command": "echo hi"}',
                                            tool_call_id="call-1")}
                else:
                    yield "done"

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                events = [event async for event in agent.run("go")
                          if not isinstance(event, RequestStats)]

            self.assertEqual(
                [type(event) for event in events],
                [ToolStart, ToolEnd, UserInput, TextDelta, TurnEnd],
            )
            # The injected request follows the tool results rather than
            # replacing them, so no tool_call_id is left unanswered.
            requests_only = [m for m in agent.history if isinstance(m, ModelRequest)]
            self.assertTrue(any(isinstance(part, ToolReturnPart)
                                for part in requests_only[-2].parts))
            self.assertEqual([part.content for part in requests_only[-1].parts],
                             ["stop, wrong file"])

    async def test_a_queued_shell_run_is_injected_verbatim(self) -> None:
        """A "!" run comes through as a ready-made request: appended as it is,
        with no @path expansion and no event, since the pane already showed
        it."""
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "notes.md").write_text("secret")
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"), mode="yolo")

            queue = [shell_message("cat list", "@notes.md")]
            agent.pending = lambda: [queue.pop(0)] if queue else []

            with patch("paimon.agent.build_model", return_value=stub_model()):
                events = [event async for event in agent.run("go")
                          if not isinstance(event, RequestStats)]

            # No event of its own, unlike a queued prompt, which yields
            # UserInput: the pane logged the run when the command exited.
            self.assertEqual([type(event) for event in events], [TextDelta, TurnEnd])
            self.assertTrue(is_shell_message(agent.history[1]))
            # The mention is output, not a mention: the file is not inlined.
            self.assertEqual(shell_text(agent.history[1]), ("cat list", "@notes.md"))

    async def test_without_the_hook_nothing_is_injected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"), mode="yolo")

            with patch("paimon.agent.build_model", return_value=stub_model()):
                events = [event async for event in agent.run("go")
                          if not isinstance(event, RequestStats)]

            self.assertEqual([type(event) for event in events], [TextDelta, TurnEnd])


class ReplayEventsTest(unittest.TestCase):
    """History replays as the same event sequence a live run would yield."""

    def test_messages_replay_as_live_events(self) -> None:
        messages = [
            ModelRequest(parts=[UserPromptPart(content="do it")]),
            ModelResponse(parts=[
                ThinkingPart(content="planning"),
                TextPart(content="ok"),
                ToolCallPart(tool_name="shell", args='{"command": "ls"}', tool_call_id="c1"),
                ToolCallPart(tool_name="write_todos",
                             args='{"todos": [{"content": "x", "status": "pending"}]}', tool_call_id="c2"),
            ]),
            ModelRequest(parts=[
                ToolReturnPart(tool_name="shell", content="a.py", tool_call_id="c1"),
                ToolReturnPart(tool_name="write_todos", content="[ ] x", tool_call_id="c2"),
            ]),
            ModelResponse(parts=[TextPart(content="done")]),
        ]

        events = replay_events(messages)

        # SESSION-5: each call is followed by its own result, as live serial
        # execution orders them, not all starts of a batch first.
        self.assertEqual(
            [type(event) for event in events],
            [UserInput, ReasoningDelta, TextDelta, ToolStart, ToolEnd, TodosUpdate, TextDelta],
        )
        tool_end = events[4]
        self.assertEqual((tool_end.id, tool_end.name, tool_end.result), ("c1", "shell", "a.py"))

    def test_compaction_summary_becomes_notice(self) -> None:
        events = replay_events([
            summary_message("checkpoint"),
            ModelRequest(parts=[UserPromptPart(content="hi")]),
        ])
        self.assertEqual([type(event) for event in events], [CompactionNotice, UserInput])


class LiveReplayParityTest(unittest.IsolatedAsyncioTestCase):
    """SESSION-5: resumed history replays with the live event order and the
    live denied styling."""

    @staticmethod
    def _tool_events(events: list) -> list[tuple]:
        return [(type(e).__name__, e.id, getattr(e, "denied", None))
                for e in events if isinstance(e, (ToolStart, ToolEnd))]

    async def test_a_serial_tool_batch_replays_in_live_order(self) -> None:
        requests = 0

        async def stream(messages, info):
            nonlocal requests
            requests += 1
            if requests == 1:
                yield {0: DeltaToolCall(name="shell", json_args='{"command": "true"}',
                                        tool_call_id="c1"),
                       1: DeltaToolCall(name="shell", json_args='{"command": "true"}',
                                        tool_call_id="c2")}
            else:
                yield "done"

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                live = [event async for event in agent.run("go")]

            live_order = self._tool_events(live)
            self.assertEqual([item[:2] for item in live_order],
                             [("ToolStart", "c1"), ("ToolEnd", "c1"),
                              ("ToolStart", "c2"), ("ToolEnd", "c2")])
            replayed = replay_events(session.messages())
            self.assertEqual(self._tool_events(replayed), live_order)

    async def test_a_denied_tool_still_replays_as_denied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            arguments = '{"path": "a.txt", "content": "hi"}'
            confirm = AsyncMock(return_value=False)
            with patch("paimon.agent.build_model",
                       return_value=stub_model("write_file", arguments)):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"),
                                   confirm=confirm, mode="read")
                live = [event async for event in agent.run("go")]

            live_end = next(e for e in live if isinstance(e, ToolEnd))
            self.assertTrue(live_end.denied)
            replayed_end = next(e for e in replay_events(session.messages())
                                if isinstance(e, ToolEnd))
            self.assertTrue(replayed_end.denied,
                            "the denied state survives persistence and replay")


class TurnOutcomeTest(unittest.IsolatedAsyncioTestCase):
    """SESSION-1: every turn leaves a terminal turn_end record, and failures
    with partial output are persisted without entering the LLM context."""

    def _open(self, cwd: Path, model) -> Agent:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        patcher = patch("paimon.agent.build_model", return_value=model)
        patcher.start()
        self.addCleanup(patcher.stop)
        return Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))

    async def test_success_records_a_turn_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._open(Path(directory), stub_model())
            [event async for event in agent.run("go")]
            last = session_records(agent.session)[-1]
            self.assertEqual((last["type"], last["outcome"]), ("turn_end", "success"))

    async def test_zero_output_failure_records_the_error(self) -> None:
        async def explode(messages, info):
            raise RuntimeError("boom")
            yield  # pragma: no cover - makes this an async generator

        with tempfile.TemporaryDirectory() as directory:
            agent = self._open(Path(directory), FunctionModel(stream_function=explode))
            with self.assertRaises(RuntimeError):
                [event async for event in agent.run("go")]
            last = session_records(agent.session)[-1]
            self.assertEqual(last["outcome"], "error")
            self.assertEqual(last["error"], "RuntimeError: boom")
            self.assertNotIn("partial_text", last)
            # No error assistant enters the context: the history ends on the
            # user request, which is still resumable.
            self.assertIsInstance(agent.session.messages()[-1], ModelRequest)

    async def test_failure_after_partial_output_persists_the_partial(self) -> None:
        async def stream(messages, info):
            yield "half an answer"
            raise RuntimeError("dropped")

        with tempfile.TemporaryDirectory() as directory:
            agent = self._open(Path(directory), FunctionModel(stream_function=stream))
            with self.assertRaises(RuntimeError):
                [event async for event in agent.run("go")]
            last = session_records(agent.session)[-1]
            self.assertEqual(last["outcome"], "error")
            self.assertEqual(last["partial_text"], "half an answer")
            for message in agent.session.messages():
                if isinstance(message, ModelResponse):
                    for part in message.parts:
                        self.assertNotIn("half an answer", getattr(part, "content", ""),
                                         "the partial must not replay as a normal answer")

    async def test_transient_retries_are_recorded(self) -> None:
        attempts = 0

        async def stream(messages, info):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("dropped")
            yield "done"

        with tempfile.TemporaryDirectory() as directory:
            with patch("paimon.agent.asyncio.sleep"):
                agent = self._open(Path(directory), FunctionModel(stream_function=stream))
                [event async for event in agent.run("go")]
            records = session_records(agent.session)
            retries = [r for r in records if r["type"] == "model_retry"]
            self.assertEqual([r["attempt"] for r in retries], [1])
            self.assertEqual(records[-1]["outcome"], "success")

    async def test_cancellation_records_interrupted(self) -> None:
        async def stream(messages, info):
            yield "partial answer"
            raise asyncio.CancelledError

        with tempfile.TemporaryDirectory() as directory:
            agent = self._open(Path(directory), FunctionModel(stream_function=stream))
            with self.assertRaises(asyncio.CancelledError):
                [event async for event in agent.run("go")]
            ends = [r for r in session_records(agent.session) if r["type"] == "turn_end"]
            self.assertEqual([r["outcome"] for r in ends], ["interrupted"])
            self.assertEqual(ends[0]["partial_text"], "partial answer")
            # SESSION-2: no truncated answer is persisted as a completed one.
            self.assertIsInstance(agent.session.messages()[-1], ModelRequest)
            self.assertEqual(agent.history, agent.session.messages())

    async def test_budget_stop_records_max_tool_calls(self) -> None:
        arguments = '{"todos": [{"content": "x", "status": "pending"}]}'
        with tempfile.TemporaryDirectory() as directory:
            agent = self._open(Path(directory), stub_model("write_todos", arguments))
            [event async for event in agent.run("go", max_tool_calls=0)]
            self.assertEqual(session_records(agent.session)[-1]["outcome"], "max_tool_calls")

    async def test_compaction_failure_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("paimon.agent.Agent._maybe_compact",
                       new=AsyncMock(side_effect=[httpx.ConnectError("dropped"), None])):
                arguments = '{"todos": [{"content": "x", "status": "pending"}]}'
                agent = self._open(Path(directory), stub_model("write_todos", arguments))
                [event async for event in agent.run("go")]
            failures = [r for r in session_records(agent.session)
                        if r["type"] == "compaction_failed"]
            self.assertEqual(len(failures), 1)
            self.assertIn("dropped", failures[0]["error"])


class AgentStatusInjectionTest(unittest.IsolatedAsyncioTestCase):
    """What the agents this session started did reaches the model as history."""

    async def test_a_status_line_opens_the_turn_and_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
            agent.supervisor = FakeSupervisor(summary="a1f2 finished")

            with patch("paimon.agent.build_model", return_value=stub_model()):
                events = [event async for event in agent.run("what now")]

            notices = [event for event in events if isinstance(event, AgentsNotice)]
            self.assertEqual([notice.text for notice in notices], ["a1f2 finished"])
            self.assertTrue(is_agents_message(agent.history[0]),
                            "it goes in ahead of the user's own message")
            self.assertFalse(is_agents_message(agent.history[1]))

            # It survives a reload, and replays as a notice rather than as
            # something the user typed.
            replayed = replay_events(session.messages())
            self.assertEqual([type(event) for event in replayed][:2], [AgentsNotice, UserInput])

    async def test_nothing_is_injected_without_news(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
            agent.supervisor = FakeSupervisor(summary=None)

            with patch("paimon.agent.build_model", return_value=stub_model()):
                events = [event async for event in agent.run("what now")]

            self.assertFalse([event for event in events if isinstance(event, AgentsNotice)])
            self.assertFalse(any(is_agents_message(message) for message in agent.history))

    async def test_a_wake_up_turn_runs_on_the_status_line_alone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
            agent.supervisor = FakeSupervisor(summary="a1f2 finished")

            with patch("paimon.agent.build_model", return_value=stub_model()):
                events = [event async for event in agent.run(None)]

            self.assertIsInstance(events[0], AgentsNotice)
            self.assertNotIn(UserInput, [type(event) for event in events])
            self.assertTrue(is_agents_message(agent.history[0]))
            self.assertFalse(any(
                isinstance(part, UserPromptPart) and not is_agents_message(message)
                for message in agent.history if isinstance(message, ModelRequest)
                for part in message.parts), "no user message is fabricated")
            replayed = replay_events(session.messages())
            self.assertNotIn(UserInput, [type(event) for event in replayed])
            # One turn, one terminal record.
            outcomes = [record for record in session_records(session)
                        if record.get("type") == "turn_end"]
            self.assertEqual(len(outcomes), 1)

    async def test_an_empty_wake_up_never_happened(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
            agent.supervisor = FakeSupervisor(summary=None)
            before = session.path.read_bytes()

            events = [event async for event in agent.run(None)]

            self.assertEqual(events, [], "no events, and no model request either")
            self.assertEqual(session.path.read_bytes(), before,
                             "nothing was persisted")
