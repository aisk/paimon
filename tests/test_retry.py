import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from helpers import make_session, stub_model
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import FunctionModel

from paimon import retry
from paimon.agent import Agent, ModelRetry, TextDelta
from paimon.config import Config


class RetryPolicyTest(unittest.TestCase):
    def test_transient_failures_are_recognized(self) -> None:
        for status in (408, 409, 429, 500, 502, 503):
            self.assertTrue(retry.is_transient(ModelHTTPError(status, "m")), status)
        self.assertTrue(retry.is_transient(httpx.ConnectError("refused")))
        self.assertTrue(retry.is_transient(httpx.ReadTimeout("slow")))
        self.assertTrue(retry.is_transient(asyncio.TimeoutError()))

    def test_permanent_failures_are_not_retried(self) -> None:
        for status in (400, 401, 403, 404, 422):
            self.assertFalse(retry.is_transient(ModelHTTPError(status, "m")), status)
        self.assertFalse(retry.is_transient(ValueError("bad model string")))

    def test_backoff_grows_and_is_capped(self) -> None:
        for attempt, base in enumerate([1.0, 2.0, 4.0, 8.0, 16.0, 16.0, 16.0], start=1):
            delay = retry.backoff(attempt)
            self.assertGreaterEqual(delay, base / 2, attempt)
            self.assertLessEqual(delay, base, attempt)

    def test_backoff_is_jittered(self) -> None:
        """Concurrent agents sharing a key must not retry in lockstep."""
        self.assertGreater(len({retry.backoff(4) for _ in range(20)}), 1)


class ContextOverflowTest(unittest.TestCase):
    """COMPACT-1: provider overflow errors are recognized for compact-and-retry."""

    def test_400_with_an_overflow_message_matches(self) -> None:
        exc = ModelHTTPError(400, "m", {
            "message": "This model's maximum context length is 8192 tokens"})
        self.assertTrue(retry.is_context_overflow(exc))
        self.assertTrue(retry.is_context_overflow(
            ModelHTTPError(400, "m", {"code": "context_length_exceeded"})))

    def test_unrelated_400s_do_not_match(self) -> None:
        self.assertFalse(retry.is_context_overflow(
            ModelHTTPError(400, "m", {"message": "invalid tool schema"})))

    def test_other_statuses_and_exceptions_do_not_match(self) -> None:
        self.assertFalse(retry.is_context_overflow(
            ModelHTTPError(429, "m", {"message": "context length exceeded"})))
        self.assertFalse(retry.is_context_overflow(ConnectionError("context length")))


def _failing_model(failures: int, exc: Exception) -> FunctionModel:
    """Fails the first ``failures`` requests, then streams a normal turn."""
    attempts = 0

    async def stream(messages, info):
        nonlocal attempts
        attempts += 1
        if attempts <= failures:
            raise exc
        yield "done"

    model = FunctionModel(stream_function=stream)
    model.attempts = lambda: attempts  # type: ignore[attr-defined]
    return model


class AgentRetryTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _agent(cwd: Path) -> Agent:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        return Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))

    async def _run(self, model) -> list[object]:
        """Run one turn, recording the backoff sleeps even when the turn fails."""
        self.sleeps: list[float] = []
        with tempfile.TemporaryDirectory() as directory:
            with patch("paimon.agent.build_model", return_value=model), \
                    patch("paimon.agent.asyncio.sleep") as sleep:
                agent = self._agent(Path(directory))
                try:
                    return [event async for event in agent.run("go")]
                finally:
                    self.sleeps = [call.args[0] for call in sleep.await_args_list]

    async def test_transient_failure_is_retried_until_it_succeeds(self) -> None:
        model = _failing_model(2, ModelHTTPError(429, "stub"))

        events = await self._run(model)

        retries = [event for event in events if isinstance(event, ModelRetry)]
        self.assertEqual([r.attempt for r in retries], [1, 2])
        # Jittered, so only the range is fixed (see retry.backoff).
        self.assertTrue(0.5 <= retries[0].delay <= 1.0, retries[0].delay)
        self.assertTrue(1.0 <= retries[1].delay <= 2.0, retries[1].delay)
        self.assertEqual(retries[0].error, "HTTP 429")
        self.assertEqual(self.sleeps, [r.delay for r in retries])
        self.assertEqual("".join(e.text for e in events if isinstance(e, TextDelta)), "done")

    async def test_retries_stop_at_the_attempt_limit(self) -> None:
        model = _failing_model(retry.MAX_ATTEMPTS, ModelHTTPError(503, "stub"))

        with self.assertRaises(ModelHTTPError):
            await self._run(model)

        self.assertEqual(len(self.sleeps), retry.MAX_ATTEMPTS - 1)

    async def test_permanent_failure_is_raised_immediately(self) -> None:
        model = _failing_model(1, ModelHTTPError(401, "stub"))

        with self.assertRaises(ModelHTTPError):
            await self._run(model)

        self.assertEqual(self.sleeps, [])

    async def test_a_stream_that_already_emitted_is_not_restarted(self) -> None:
        """Retrying here would replay the deltas the caller has already seen."""

        async def stream(messages, info):
            yield "partial"
            raise ModelHTTPError(503, "stub")

        with self.assertRaises(ModelHTTPError):
            await self._run(FunctionModel(stream_function=stream))

        self.assertEqual(self.sleeps, [])

    async def test_cancellation_is_never_retried(self) -> None:
        model = _failing_model(1, asyncio.CancelledError())

        with self.assertRaises(asyncio.CancelledError):
            await self._run(model)

        self.assertEqual(self.sleeps, [])


class RetryPersistenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_retried_turn_persists_one_response(self) -> None:
        """A failed attempt writes nothing, so the log has no partial response."""
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            model = _failing_model(1, ModelHTTPError(429, "stub"))

            with patch("paimon.agent.build_model", return_value=model), \
                    patch("paimon.agent.asyncio.sleep"):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                _events = [event async for event in agent.run("go")]

            self.assertEqual(len(session.messages()), 2)  # the prompt and one response
            self.assertEqual(agent.history, session.messages())


class StubModelSanityTest(unittest.IsolatedAsyncioTestCase):
    async def test_the_shared_stub_still_runs_without_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            with patch("paimon.agent.build_model", return_value=stub_model()):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                events = [event async for event in agent.run("go")]
        self.assertFalse([e for e in events if isinstance(e, ModelRetry)])
