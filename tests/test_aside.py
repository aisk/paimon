import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers import make_session
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from paimon import aside, retry
from paimon.agent import Agent
from paimon.config import Config


def _config() -> Config:
    return Config(model="test:stub")


def _user(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


def _call(tool_call_id: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name="shell", args={"command": "ls"},
                                             tool_call_id=tool_call_id)])


def _result(tool_call_id: str) -> ModelRequest:
    return ModelRequest(parts=[ToolReturnPart(tool_name="shell", content="out",
                                              tool_call_id=tool_call_id)])


def _prompts(messages: list) -> list[str]:
    return [part.content for message in messages if isinstance(message, ModelRequest)
            for part in message.parts if isinstance(part, UserPromptPart)]


class UsableHistoryTest(unittest.TestCase):
    """The snapshot an aside sends is always a valid request on its own."""

    def test_answered_tool_calls_are_kept(self) -> None:
        history = [_user("go"), _call("c1"), _result("c1")]

        self.assertEqual(aside.usable_history(history), history)

    def test_a_tool_call_nothing_has_answered_yet_is_dropped(self) -> None:
        # The window inside a turn between appending the response and
        # appending the results answering it.
        history = [_user("go"), _call("c1")]

        self.assertEqual(aside.usable_history(history), history[:1])

    def test_a_placeholder_result_counts_as_answered(self) -> None:
        # What the turn pre-seeds before running anything, so an interrupt
        # never leaves a dangling call.
        history = [_user("go"), _call("c1"),
                   ModelRequest(parts=[ToolReturnPart(tool_name="shell",
                                                      content="Interrupted by user.",
                                                      tool_call_id="c1")])]

        self.assertEqual(aside.usable_history(history), history)

    def test_the_snapshot_does_not_follow_later_appends(self) -> None:
        history = [_user("go")]

        snapshot = aside.usable_history(history)
        history.append(_call("c1"))

        self.assertEqual(len(snapshot), 1)


class AsideRequestTest(unittest.IsolatedAsyncioTestCase):
    """What the model is asked, and what the answer comes back as."""

    def _agent(self, cwd: Path) -> Agent:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        # Resume rebuilds the dynamic prompt; pin it so assertions on the
        # system part stay literal.
        with patch("paimon.agent.build_system_prompt", return_value="snapshot"):
            return Agent.open(cwd=cwd, session=session, config=_config())

    async def _ask(self, agent: Agent, model, question: str = "what now?") -> str:
        with patch("paimon.agent.build_model", return_value=model):
            return "".join([delta async for delta in agent.ask_aside(question)])

    async def test_the_question_is_asked_over_the_current_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            seen: list[list] = []

            async def stream(messages, info):
                seen.append(list(messages))
                yield "the "
                yield "answer"

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                agent = self._agent(cwd)
                agent.history.extend([_user("go"), ModelResponse(parts=[TextPart(content="done")])])
                answer = "".join([delta async for delta in agent.ask_aside("what now?")])

            self.assertEqual(answer, "the answer")
            system = [part.content for part in seen[0][0].parts if isinstance(part, SystemPromptPart)]
            self.assertEqual(system, ["snapshot"])
            # The history verbatim, then the question, marked so the model
            # answers it instead of taking it for the next instruction.
            self.assertEqual(len(_prompts(seen[0])), 2)
            self.assertEqual(_prompts(seen[0])[0], "go")
            self.assertTrue(_prompts(seen[0])[1].startswith(aside.DEFAULT_INSTRUCTIONS))
            self.assertIn("what now?", _prompts(seen[0])[1])

    async def test_custom_instructions_replace_the_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seen: list[list] = []

            async def stream(messages, info):
                seen.append(list(messages))
                yield "ok"

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                agent = self._agent(Path(directory))
                async for _delta in agent.ask_aside("recap", instructions="[recap] summarize"):
                    pass

            self.assertEqual(_prompts(seen[0])[-1], "[recap] summarize\n\nrecap")

    async def test_the_agents_tools_are_offered_so_the_request_keeps_its_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            offered: list[list[str]] = []

            async def stream(messages, info):
                offered.append([tool.name for tool in info.function_tools])
                yield "ok"

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                agent = self._agent(Path(directory))
                await self._ask(agent, FunctionModel(stream_function=stream))

            self.assertIn("read_file", offered[0])

    async def test_a_tool_call_still_in_flight_is_left_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seen: list[list] = []

            async def stream(messages, info):
                seen.append(list(messages))
                yield "ok"

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                agent = self._agent(Path(directory))
                agent.history.extend([_user("go"), _call("c1")])
                await self._ask(agent, FunctionModel(stream_function=stream))

            calls = [part for message in seen[0] if isinstance(message, ModelResponse)
                     for part in message.parts if isinstance(part, ToolCallPart)]
            self.assertEqual(calls, [])

    async def test_thinking_from_another_model_is_stripped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            seen: list[list] = []

            async def stream(messages, info):
                seen.append(list(messages))
                yield "ok"

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                agent = self._agent(Path(directory))
                agent.history.extend([
                    _user("go"),
                    ModelResponse(parts=[ThinkingPart(content="hm"), TextPart(content="done")],
                                  model_name="claude", provider_name="anthropic"),
                ])
                await self._ask(agent, FunctionModel(stream_function=stream))

            thinking = [part for message in seen[0] if isinstance(message, ModelResponse)
                        for part in message.parts if isinstance(part, ThinkingPart)]
            self.assertEqual(thinking, [])

    async def test_an_answer_of_nothing_but_a_tool_call_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            async def stream(messages, info):
                yield {0: DeltaToolCall(name="shell", json_args='{"command": "ls"}',
                                        tool_call_id="c1")}

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                agent = self._agent(Path(directory))
                with self.assertRaises(aside.AsideError):
                    async for _delta in agent.ask_aside("what now?"):
                        pass


class AsideLeavesNoTraceTest(unittest.IsolatedAsyncioTestCase):
    """Neither the question nor the answer is part of the conversation."""

    async def test_nothing_reaches_the_history_or_the_session_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")

            async def stream(messages, info):
                yield "an answer"

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                agent = Agent.open(cwd=cwd, session=session, config=_config())
                agent.history.extend([_user("go"), ModelResponse(parts=[TextPart(content="done")])])
                before = list(agent.history)
                lines_before = session.path.read_text().count("\n")

                await aside.ask(agent._model(), agent.system_prompt, agent.history, "what now?")
                async for _delta in agent.ask_aside("what now?"):
                    pass

            self.assertEqual(agent.history, before)
            self.assertEqual(session.path.read_text().count("\n"), lines_before)
            self.assertNotIn("what now?", session.path.read_text())


class AsideDuringATurnTest(unittest.IsolatedAsyncioTestCase):
    """The point of the whole thing: asking while the turn waits on the model."""

    async def test_an_aside_answers_while_a_turn_is_still_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            requested = asyncio.Event()
            release = asyncio.Event()

            async def stream(messages, info):
                if aside.DEFAULT_INSTRUCTIONS in (_prompts(messages)[-1] if _prompts(messages) else ""):
                    yield "still on the tool"
                    return
                requested.set()
                await release.wait()
                yield "done"

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                agent = Agent.open(cwd=cwd, session=session, config=_config())
                turn = asyncio.create_task(_drain(agent.run("go")))
                await requested.wait()

                answer = "".join([delta async for delta in agent.ask_aside("what now?")])

                self.assertEqual(answer, "still on the tool")
                self.assertFalse(turn.done(), "the aside did not wait for the turn")
                release.set()
                await turn

            # The turn recorded exactly what it would have without the aside.
            self.assertEqual(_prompts(agent.history), ["go"])
            self.assertEqual(_prompts(session.messages()), ["go"])


async def _drain(events) -> None:
    async for _event in events:
        pass


class AsideRetryTest(unittest.IsolatedAsyncioTestCase):
    """Same policy as a turn: retry a transient failure, never a started stream."""

    async def _ask(self, model) -> str:
        self.sleeps: list[float] = []
        with patch("paimon.aside.asyncio.sleep") as sleep:
            try:
                return await aside.ask(model, "snapshot", [_user("go")], "what now?")
            finally:
                self.sleeps = [call.args[0] for call in sleep.await_args_list]

    async def test_a_transient_failure_is_retried(self) -> None:
        attempts = 0

        async def stream(messages, info):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ModelHTTPError(429, "stub")
            yield "an answer"

        answer = await self._ask(FunctionModel(stream_function=stream))

        self.assertEqual(answer, "an answer")
        self.assertEqual(len(self.sleeps), 1)

    async def test_a_permanent_failure_is_raised(self) -> None:
        async def stream(messages, info):
            raise ModelHTTPError(401, "stub")
            yield ""

        with self.assertRaises(ModelHTTPError):
            await self._ask(FunctionModel(stream_function=stream))

        self.assertEqual(self.sleeps, [])

    async def test_retries_stop_at_the_attempt_limit(self) -> None:
        async def stream(messages, info):
            raise ModelHTTPError(503, "stub")
            yield ""

        with self.assertRaises(ModelHTTPError):
            await self._ask(FunctionModel(stream_function=stream))

        self.assertEqual(len(self.sleeps), retry.MAX_ATTEMPTS - 1)

    async def test_a_stream_that_already_answered_is_not_restarted(self) -> None:
        async def stream(messages, info):
            yield "partial"
            raise ModelHTTPError(503, "stub")

        with self.assertRaises(ModelHTTPError):
            await self._ask(FunctionModel(stream_function=stream))

        self.assertEqual(self.sleeps, [])


if __name__ == "__main__":
    unittest.main()
