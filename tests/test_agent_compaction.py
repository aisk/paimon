import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from paimon import compaction
from paimon.agent import (
    Agent,
    ContextCompactionFailed,
    TextDelta,
)
from paimon.config import Config
from paimon.session import (
    is_summary_message,
)
from tests.support.agent import make_session, session_records, stub_model


class ManualCompactionTest(unittest.IsolatedAsyncioTestCase):
    async def test_compact_now_ignores_the_toggle_and_the_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            with (
                patch("paimon.agent.Session.create", return_value=session),
                patch("paimon.agent.build_system_prompt", return_value="snapshot"),
            ):
                # A tiny history under a disabled auto-compaction config: the
                # automatic path would decline on both counts.
                agent = Agent.open(cwd=cwd, config=Config(model="test:stub", compaction_enabled=False))
            old = ModelRequest(parts=[UserPromptPart(content="old")])
            recent = ModelResponse(parts=[TextPart(content="recent")])
            agent._append_message(old)
            agent._append_message(recent)

            result = compaction.CompactionResult("checkpoint", [recent], 100, 0)
            with (
                patch("paimon.agent.Agent._model", return_value=object()),
                patch("paimon.compaction.compact", new=AsyncMock(return_value=result)) as compact,
            ):
                self.assertIsNone(await agent._maybe_compact())
                returned = await agent.compact_now()

            self.assertIs(returned, result)
            compact.assert_awaited_once()
            self.assertTrue(is_summary_message(agent.history[0]))
            self.assertEqual(agent.history[1:], [recent])
            # the checkpoint is persisted, so a resume replays the same context
            replayed = session.messages()
            self.assertEqual(len(replayed), 2)
            self.assertIn("checkpoint", replayed[0].parts[0].content)


class CompactionTokenCountTest(unittest.IsolatedAsyncioTestCase):
    """Counting tokens serializes the whole history, so it stays off the loop."""

    @staticmethod
    def _agent(cwd: Path, **settings) -> Agent:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        return Agent.open(cwd=cwd, session=session, config=Config(**settings))

    async def test_an_unknown_window_counts_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            # "test:stub" matches no entry in the window table and there is no
            # override, so auto-compaction is off and counting is wasted work.
            agent = self._agent(Path(directory), model="test:stub")
            with patch("paimon.compaction.count_tokens") as count:
                self.assertIsNone(await agent._maybe_compact())
            count.assert_not_called()

    async def test_the_count_runs_in_a_worker_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory), model="test:stub",
                                compaction_context_window=1_000,
                                compaction_reserve_tokens=0)
            threads: list[threading.Thread] = []

            def count(*args, **kwargs) -> int:
                threads.append(threading.current_thread())
                return 1  # well under the window, so nothing is compacted

            with patch("paimon.compaction.count_tokens", side_effect=count):
                self.assertIsNone(await agent._maybe_compact())

            self.assertEqual(len(threads), 1)
            self.assertIsNot(threads[0], threading.main_thread())


class UsageAnchorTest(unittest.IsolatedAsyncioTestCase):
    """COMPACT-1: provider-reported usage drives the context count when
    available; the chars/4 heuristic only covers what came after it."""

    def _agent(self, cwd: Path) -> Agent:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        return Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))

    async def test_count_prefers_the_anchor_and_estimates_only_the_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            agent._append_message(ModelRequest(parts=[UserPromptPart(content="x" * 100_000)]))
            agent._usage_anchor = (len(agent.history), 5_000)

            self.assertEqual(await agent.count_context_tokens(), 5_000)

            agent._append_message(ModelRequest(parts=[UserPromptPart(content="y" * 400)]))
            count = await agent.count_context_tokens()
            self.assertGreater(count, 5_000)
            self.assertLess(count, 5_600, "only the appended suffix is estimated")

    async def test_a_completed_request_sets_the_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("paimon.agent.build_model", return_value=stub_model()):
                agent = self._agent(Path(directory))
                [event async for event in agent.run("go")]
            self.assertIsNotNone(agent._usage_anchor)
            self.assertEqual(agent._usage_anchor[0], len(agent.history))

    async def test_compaction_drops_the_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            agent._append_message(ModelRequest(parts=[UserPromptPart(content="old")]))
            recent = ModelResponse(parts=[TextPart(content="recent")])
            agent._append_message(recent)
            agent._usage_anchor = (2, 9_000)
            result = compaction.CompactionResult("checkpoint", [recent], 100, 0)
            with patch("paimon.agent.build_model", return_value=stub_model()), \
                    patch("paimon.agent.compaction.compact", new=AsyncMock(return_value=result)):
                await agent.compact_now()
            self.assertIsNone(agent._usage_anchor)


class OverflowRecoveryTest(unittest.IsolatedAsyncioTestCase):
    """COMPACT-1: a provider context-overflow compacts and retries once."""

    async def test_overflow_compacts_and_retries_once(self) -> None:
        from pydantic_ai.exceptions import ModelHTTPError

        attempts = 0

        async def stream(messages, info):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ModelHTTPError(400, "stub", {"message": "context length exceeded"})
            yield "done"

        result = compaction.CompactionResult("checkpoint", [], 100, 10)

        async def fake_compact(force: bool = False):
            return result if force else None

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            with (
                patch("paimon.agent.build_model",
                      return_value=FunctionModel(stream_function=stream)),
                patch("paimon.agent.Agent._maybe_compact", new=AsyncMock(side_effect=fake_compact)),
            ):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                events = [event async for event in agent.run("go")]

            self.assertEqual(attempts, 2, "the request is retried after compaction")
            self.assertEqual("".join(e.text for e in events if isinstance(e, TextDelta)), "done")
            records = session_records(session)
            self.assertTrue([r for r in records if r["type"] == "context_overflow"])
            self.assertEqual(records[-1]["outcome"], "success")

    async def test_a_second_overflow_is_raised_not_looped(self) -> None:
        from pydantic_ai.exceptions import ModelHTTPError

        attempts = 0

        async def stream(messages, info):
            nonlocal attempts
            attempts += 1
            raise ModelHTTPError(400, "stub", {"message": "context length exceeded"})
            yield  # pragma: no cover - makes this an async generator

        result = compaction.CompactionResult("checkpoint", [], 100, 10)

        async def fake_compact(force: bool = False):
            return result if force else None

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            with (
                patch("paimon.agent.build_model",
                      return_value=FunctionModel(stream_function=stream)),
                patch("paimon.agent.Agent._maybe_compact", new=AsyncMock(side_effect=fake_compact)),
            ):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                with self.assertRaises(ModelHTTPError):
                    [event async for event in agent.run("go")]

            self.assertEqual(attempts, 2, "exactly one retry, then the error surfaces")


class CompactionFailureTest(unittest.IsolatedAsyncioTestCase):
    """A failed compaction must not silently disable the safety net for the turn."""

    async def _run_with(self, failure: Exception) -> tuple[list, AsyncMock]:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            # write_todos needs no confirmation, so the turn takes two model
            # requests and therefore two compaction checks.
            arguments = '{"todos": [{"content": "x", "status": "pending"}]}'
            compact = AsyncMock(side_effect=[failure, None])
            with (
                patch("paimon.agent.build_model",
                      return_value=stub_model("write_todos", arguments)),
                patch("paimon.agent.Agent._maybe_compact", new=compact),
            ):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                events = [event async for event in agent.run("go")]
            return events, compact

    async def test_transient_failure_is_retried_on_the_next_step(self) -> None:
        events, compact = await self._run_with(httpx.ConnectError("dropped"))

        self.assertEqual(compact.await_count, 2)
        self.assertEqual(len([e for e in events if isinstance(e, ContextCompactionFailed)]), 1)

    async def test_a_failure_that_will_not_fix_itself_stops_for_the_turn(self) -> None:
        events, compact = await self._run_with(ValueError("no context window"))

        self.assertEqual(compact.await_count, 1)
        self.assertEqual(len([e for e in events if isinstance(e, ContextCompactionFailed)]), 1)
