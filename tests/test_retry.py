import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models.function import FunctionModel

from paimon import retry
from paimon.agent import Agent, ModelRetry, TextDelta
from paimon.config import Config
from tests.support.agent import make_session


class RetryPolicyTest(unittest.TestCase):
    def test_transient_failures_are_recognized(self) -> None:
        for status in (408, 409, 429, 500, 502, 503):
            self.assertTrue(retry.is_transient(ModelHTTPError(status, "m")), status)
        self.assertTrue(retry.is_transient(httpx.ConnectError("refused")))
        self.assertTrue(retry.is_transient(httpx.ReadTimeout("slow")))
        self.assertTrue(retry.is_transient(asyncio.TimeoutError()))

    def test_wrapped_connection_failures_are_transient(self) -> None:
        """pydantic-ai wraps connect-phase httpx errors in ModelAPIError."""
        self.assertTrue(retry.is_transient(ModelAPIError("m", "connection refused")))

    def test_permanent_failures_are_not_retried(self) -> None:
        for status in (400, 401, 403, 404, 422):
            self.assertFalse(retry.is_transient(ModelHTTPError(status, "m")), status)
        self.assertFalse(retry.is_transient(ValueError("bad model string")))

    def test_backoff_grows_and_is_capped(self) -> None:
        # Lower edge, midpoint, upper edge; later attempts keep the same cap.
        expected = [(0.5, 0.75, 1.0), (1.0, 1.5, 2.0), (2.0, 3.0, 4.0),
                    (4.0, 6.0, 8.0), (8.0, 12.0, 16.0), (8.0, 12.0, 16.0),
                    (8.0, 12.0, 16.0)]
        for attempt, delays in enumerate(expected, start=1):
            for random_value, expected_delay in zip((0.0, 0.5, 1.0), delays):
                with self.subTest(attempt=attempt, random_value=random_value):
                    with patch("paimon.retry.random.random", return_value=random_value):
                        delay = retry.backoff(attempt)
                    self.assertEqual(delay, expected_delay)


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

    return FunctionModel(stream_function=stream)


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

    async def test_success_needs_no_retry(self) -> None:
        events = await self._run(_failing_model(0, ModelHTTPError(429, "stub")))

        self.assertEqual("".join(e.text for e in events if isinstance(e, TextDelta)), "done")
        self.assertFalse(any(isinstance(e, ModelRetry) for e in events))
        self.assertEqual(self.sleeps, [])

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
