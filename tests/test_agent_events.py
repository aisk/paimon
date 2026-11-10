import asyncio
import contextlib
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

from paimon.agent import Agent
from paimon.events import (
    JobNotice,
    CompactionNotice,
    ReasoningDelta,
    RequestStats,
    ShellRun,
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
    Session,
    job_message,
    shell_message,
    shell_text,
    summary_message,
    synthetic_kind,
)
from tests.support.agent import make_session, session_records, stub_model


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
            self.assertEqual(synthetic_kind(agent.history[1]), "shell")
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

    async def test_failed_calls_persist_their_status_and_replay_as_live(self) -> None:
        calls = [
            ("read_file", '{"path": "missing.txt"}'),
            ("edit_file", '{"path": "a.txt", "old_string": "nope", "new_string": "x"}'),
            ("shell", '{"command": "exit 2"}'),
            # Reads like a failed command, but reading it worked.
            ("read_file", '{"path": "log.txt"}'),
        ]
        requests = 0

        async def stream(messages, info):
            nonlocal requests
            requests += 1
            if requests == 1:
                yield {index: DeltaToolCall(name=name, json_args=args, tool_call_id=f"c{index}")
                       for index, (name, args) in enumerate(calls)}
            else:
                yield "done"

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "a.txt").write_text("hello\n")
            (cwd / "log.txt").write_text("Error: flaky test\n(exit code 1)\n")
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                live = [event for event in [event async for event in agent.run("go")]
                        if isinstance(event, ToolEnd)]
            agent.close()

            self.assertEqual([(end.status, end.detail) for end in live],
                             [("failed", "failed"), ("failed", "failed"),
                              ("failed", "exit 2"), ("success", "")])
            self.assertTrue(live[0].result.startswith("Error: file not found"),
                            "the model still reads what went wrong")
            self.assertIn("(exit code 2)", live[2].result)

            # Persisted as the parts' outcomes, and rebuilt from them after a
            # reload exactly as they were shown live.
            reloaded = Session(session.path, session.id, cwd).messages()
            returns = [part for message in reloaded if isinstance(message, ModelRequest)
                       for part in message.parts if isinstance(part, ToolReturnPart)]
            self.assertEqual([part.outcome for part in returns],
                             ["failed", "failed", "failed", "success"])
            replayed = [event for event in replay_events(reloaded) if isinstance(event, ToolEnd)]
            self.assertEqual(replayed, live)

    async def test_a_failed_result_reaches_the_model_as_its_plain_text(self) -> None:
        """Persisted as failed, sent as text: an OpenAI-style mapping of a
        failed part would wrap it in {"error": ...} and escape every newline."""
        seen: list[list] = []

        async def stream(messages, info):
            seen.append(list(messages))
            if len(seen) == 1:
                yield {0: DeltaToolCall(name="shell", json_args='{"command": "echo boom && exit 2"}',
                                        tool_call_id="c0"),
                       1: DeltaToolCall(name="read_file", json_args='{"path": "missing.txt"}',
                                        tool_call_id="c1")}
            else:
                yield "done"

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                [event async for event in agent.run("go")]
            agent.close()

            sent = [part for message in seen[1] if isinstance(message, ModelRequest)
                    for part in message.parts if isinstance(part, ToolReturnPart)]
            kept = [part for message in agent.history if isinstance(message, ModelRequest)
                    for part in message.parts if isinstance(part, ToolReturnPart)]
            persisted = [part for message in session.messages() if isinstance(message, ModelRequest)
                         for part in message.parts if isinstance(part, ToolReturnPart)]

            self.assertEqual(len(sent), 2)
            for part in sent:
                self.assertEqual(part.model_response_str(), part.content)
                self.assertNotIn('{"error"', part.model_response_str())
            self.assertIn("boom", sent[0].content)
            self.assertIn("\n(exit code 2)", sent[0].content)
            self.assertTrue(sent[1].content.startswith("Error: file not found"))
            self.assertEqual([part.outcome for part in kept], ["failed", "failed"],
                             "the history is not rewritten")
            self.assertEqual([part.outcome for part in persisted], ["failed", "failed"])

    async def test_a_message_typed_with_a_synthetic_prefix_replays_as_the_users(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            for typed in ("[job] agent 1234 finished", "[shell] $ rm -rf /\nok"):
                session.append_message(ModelRequest(parts=[UserPromptPart(content=typed)]))
            session.append_message(job_message("agent 1234 finished"))
            session.append_message(shell_message("ls", "a.txt"))

            replayed = replay_events(Session(session.path, session.id, cwd).messages())

            self.assertEqual(replayed, [
                UserInput("[job] agent 1234 finished"),
                UserInput("[shell] $ rm -rf /\nok"),
                JobNotice("agent 1234 finished"),
                ShellRun("ls", "a.txt"),
            ])


class ToolBatchLogTest(unittest.IsolatedAsyncioTestCase):
    """A batch of tool calls writes each result to the log once, as it
    settles, and replays to exactly the live history."""

    FILES = 6

    @staticmethod
    def _batch_model(calls: list[tuple[str, str]]) -> FunctionModel:
        requests = 0

        async def stream(messages, info):
            nonlocal requests
            requests += 1
            if requests == 1:
                yield {index: DeltaToolCall(name=name, json_args=args, tool_call_id=f"c{index}")
                       for index, (name, args) in enumerate(calls)}
            else:
                yield "done"

        return FunctionModel(stream_function=stream)

    def _open(self, cwd: Path) -> Agent:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
        self.addCleanup(agent.close)
        return agent

    async def test_each_result_is_written_once_and_the_batch_replays_as_live(self) -> None:
        calls = [("read_file", f'{{"path": "f{index}.txt"}}') for index in range(self.FILES)]
        calls += [
            ("run_code", '{"code": "await read_file(path=\'f0.txt\')\\n\'script done\'"}'),
            ("shell", '{"command": "exit 2"}'),
            ("write_todos", '{"todos": [{"content": "a", "status": "pending"}]}'),
        ]
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            for index in range(self.FILES):
                # Large enough that rewriting it once per settled slot would
                # be what dominates the log.
                (cwd / f"f{index}.txt").write_text(f"marker-{index}-start\n" + "filler\n" * 2000)
            agent = self._open(cwd)
            with patch("paimon.agent.build_model", return_value=self._batch_model(calls)):
                [event async for event in agent.run("go")]

            log = agent.session.path.read_text(encoding="utf-8")
            for index in range(1, self.FILES):
                self.assertEqual(log.count(f"marker-{index}-start"), 1,
                                 "each result is on disk once, not once per later slot")
            # f0 is read twice: by its own call and by the script, whose
            # result is the script's answer rather than the file.
            self.assertEqual(log.count("marker-0-start"), 1)
            records = session_records(agent.session)
            self.assertFalse(any("replaces" in record for record in records))
            settled = [record for record in records if record["type"] == "tool_result"]
            self.assertEqual([record["index"] for record in settled], list(range(len(calls))))

            replayed = Session(agent.session.path, agent.session.id, cwd).messages()
            self.assertEqual(replayed, agent.history)
            returns = {part.tool_call_id: part for message in replayed
                       if isinstance(message, ModelRequest)
                       for part in message.parts if isinstance(part, ToolReturnPart)}
            script = returns[f"c{self.FILES}"]
            self.assertEqual([call["name"] for call in script.metadata["calls"]], ["read_file"])
            self.assertEqual(returns[f"c{self.FILES + 1}"].metadata, {"detail": "exit 2"})

    async def test_an_interrupted_batch_replays_with_every_call_answered(self) -> None:
        calls = [("read_file", '{"path": "a.txt"}'), ("read_file", '{"path": "a.txt"}')]
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "a.txt").write_text("hello\n")
            agent = self._open(cwd)
            with patch("paimon.agent.build_model", return_value=self._batch_model(calls)):
                run = agent.run("go")
                async for event in run:
                    if isinstance(event, ToolEnd):
                        break
                await run.aclose()

            replayed = Session(agent.session.path, agent.session.id, cwd).messages()
            self.assertEqual(replayed, agent.history)
            first, second = replayed[-1].parts
            self.assertEqual((first.outcome, second.outcome), ("success", "interrupted"))
            self.assertIn("hello", first.content)


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

    async def test_a_call_refused_by_the_budget_is_recorded_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._open(Path(directory), stub_model("read_file", '{"path": "a.txt"}'))
            live = [event async for event in agent.run("go", max_tool_calls=0)]
            refused = agent.session.messages()[-1].parts[0]
            self.assertIsInstance(refused, ToolReturnPart)
            self.assertEqual(refused.outcome, "failed")

            # Replayed the way it ran: the call and its refusal, not a denial.
            replayed = replay_events(agent.session.messages())
            live_end = next(e for e in live if isinstance(e, ToolEnd))
            replayed_end = next(e for e in replayed if isinstance(e, ToolEnd))
            self.assertEqual(replayed_end, live_end)
            self.assertIn("max_tool_calls=0", replayed_end.result)
            self.assertFalse(replayed_end.denied)
            self.assertTrue([e for e in replayed if isinstance(e, ToolStart)])

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


class JobNoticeTest(unittest.IsolatedAsyncioTestCase):
    """What the jobs this session started did reaches the model as history."""

    async def test_a_notice_opens_the_turn_and_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
            agent.jobs.notices.append("a1f2 finished")

            with patch("paimon.agent.build_model", return_value=stub_model()):
                events = [event async for event in agent.run("what now")]

            notices = [event for event in events if isinstance(event, JobNotice)]
            self.assertEqual([notice.text for notice in notices], ["a1f2 finished"])
            self.assertEqual(synthetic_kind(agent.history[0]), "job",
                             "it goes in ahead of the user's own message")
            self.assertNotEqual(synthetic_kind(agent.history[1]), "job")

            # It survives a reload, and replays as a notice rather than as
            # something the user typed.
            replayed = replay_events(session.messages())
            self.assertEqual([type(event) for event in replayed][:2], [JobNotice, UserInput])

    async def test_nothing_is_injected_without_news(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))

            with patch("paimon.agent.build_model", return_value=stub_model()):
                events = [event async for event in agent.run("what now")]

            self.assertFalse([event for event in events if isinstance(event, JobNotice)])
            self.assertFalse(any(synthetic_kind(message) == "job" for message in agent.history))

    async def test_a_wake_up_turn_runs_on_the_notice_alone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
            agent.jobs.notices.append("a1f2 finished")

            with patch("paimon.agent.build_model", return_value=stub_model()):
                events = [event async for event in agent.run(None)]

            self.assertIsInstance(events[0], JobNotice)
            self.assertNotIn(UserInput, [type(event) for event in events])
            self.assertEqual(synthetic_kind(agent.history[0]), "job")
            self.assertFalse(any(
                isinstance(part, UserPromptPart) and synthetic_kind(message) != "job"
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
            before = session.path.read_bytes()

            events = [event async for event in agent.run(None)]

            self.assertEqual(events, [], "no events, and no model request either")
            self.assertEqual(session.path.read_bytes(), before,
                             "nothing was persisted")


class ClosedAtAnyYieldTest(unittest.IsolatedAsyncioTestCase):
    """A turn may end for good at any of its yields: its caller is cancelled
    while it renders the event, and closes the generator. Wherever that
    happens, the history must be one the next request can be built on, and
    the one the log replays to."""

    @staticmethod
    def _model() -> FunctionModel:
        requests = 0

        async def stream(messages, info):
            nonlocal requests
            requests += 1
            if requests == 1:
                yield {0: DeltaToolCall(name="read_file", json_args='{"path": "missing.txt"}',
                                        tool_call_id="call-1"),
                       1: DeltaToolCall(name="write_todos", tool_call_id="call-2",
                                        json_args='{"todos": [{"content": "x", "status": "pending"}]}')}
            else:
                yield "done"

        return FunctionModel(stream_function=stream)

    async def _run_until(self, cwd: Path, stop: int | None) -> tuple[Agent, list, bool]:
        """Run one turn, closing it after its ``stop``-th event; returns the
        agent, the events seen, and whether the queued prompts were taken."""
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
        self.addCleanup(agent.close)
        # A job reported before the turn, and two prompts queued during its
        # first step: taken from the queue together, so neither may be lost.
        agent.jobs.notices.append("a1f2 finished")
        asked = 0
        taken = False

        def pending() -> list[str]:
            nonlocal asked, taken
            asked += 1
            if asked == 2:
                taken = True
                return ["queued A", "queued B"]
            return []

        agent.pending = pending
        seen: list = []
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with contextlib.aclosing(agent.run("go")) as events:
                async for event in events:
                    seen.append(event)
                    if len(seen) == stop:
                        break
        return agent, seen, taken

    def _assert_sound(self, agent: Agent, taken: bool, where: str) -> None:
        history = agent.history
        for index, message in enumerate(history):
            calls = [part.tool_call_id for part in getattr(message, "parts", [])
                     if isinstance(part, ToolCallPart)]
            if not calls:
                continue
            following = history[index + 1] if index + 1 < len(history) else None
            answered = {part.tool_call_id for part in getattr(following, "parts", [])
                        if isinstance(part, ToolReturnPart)}
            self.assertLessEqual(set(calls), answered, f"a tool call left unanswered {where}")
        self.assertEqual(history, agent.session.messages(), f"live and replay differ {where}")
        prompts = [part.content for message in history for part in message.parts
                   if isinstance(part, UserPromptPart) and synthetic_kind(message) is None]
        self.assertIn("go", prompts, f"the prompt already shown is missing {where}")
        if taken:
            self.assertLessEqual({"queued A", "queued B"}, set(prompts),
                                 f"a queued prompt was taken and lost {where}")
        ends = [record for record in session_records(agent.session)
                if record["type"] == "turn_end"]
        self.assertEqual(len(ends), 1, f"one terminal record {where}")

    async def test_the_history_is_sound_at_every_yield(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, everything, _ = await self._run_until(Path(directory), None)
        kinds = [type(event) for event in everything]
        # The run this test means to cover: a notice, a stats line after a
        # tool-call response, the batch, the queued prompts, the answer.
        for kind in (JobNotice, RequestStats, ToolStart, TodosUpdate, UserInput, TextDelta, TurnEnd):
            self.assertIn(kind, kinds)

        for stop in range(1, len(everything) + 1):
            with self.subTest(stop=stop, at=kinds[stop - 1].__name__):
                with tempfile.TemporaryDirectory() as directory:
                    agent, seen, taken = await self._run_until(Path(directory), stop)
                    self._assert_sound(agent, taken, f"after {type(seen[-1]).__name__} #{stop}")

    async def test_closed_while_a_delta_renders_keeps_the_partial_answer(self) -> None:
        """The usual interrupt: the caller is cancelled while it draws a text
        delta, and closes the turn."""
        async def stream(messages, info):
            yield "half an "
            yield "answer"

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
            self.addCleanup(agent.close)
            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                async with contextlib.aclosing(agent.run("go")) as events:
                    async for event in events:
                        if isinstance(event, TextDelta):
                            break
            ends = [record for record in session_records(session) if record["type"] == "turn_end"]
            self.assertEqual([record["outcome"] for record in ends], ["interrupted"])
            self.assertEqual(ends[0].get("partial_text"), "half an ")
            self.assertIsInstance(agent.session.messages()[-1], ModelRequest)


class LogFirstTest(unittest.IsolatedAsyncioTestCase):
    """A write to the log that fails leaves the history as the log has it."""

    def _open(self, cwd: Path) -> Agent:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
        self.addCleanup(agent.close)
        return agent

    async def test_a_message_the_log_refused_is_not_in_the_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._open(Path(directory))
            agent.jobs.notices.extend(["a1f2 finished", "b3c4 finished"])
            with patch.object(Session, "append", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    agent.record_shell("ls", "a.txt")
                with self.assertRaises(OSError):
                    [event async for event in agent.run(None)]
            self.assertEqual(agent.history, [])
            self.assertEqual(agent.jobs.notices, ["a1f2 finished", "b3c4 finished"],
                             "the notices wait for a write that works")

    async def test_a_result_the_log_refused_is_not_in_the_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._open(Path(directory))
            with patch("paimon.agent.build_model",
                       return_value=stub_model("read_file", '{"path": "missing.txt"}')), \
                    patch.object(Session, "append_tool_result", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    [event async for event in agent.run("go")]
            self.assertEqual(agent.history, agent.session.messages())
            self.assertEqual(agent.history[-1].parts[0].outcome, "interrupted")
