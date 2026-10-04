import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from paimon.agent import RECAP_PROMPT, Agent
from paimon.config import Config
from tests.support.agent import make_session


def _user(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


def _prompts(messages: list) -> list[str]:
    return [part.content for message in messages if isinstance(message, ModelRequest)
            for part in message.parts if isinstance(part, UserPromptPart)]


class RecapTest(unittest.IsolatedAsyncioTestCase):
    """A question over the conversation that the conversation never hears of."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cwd = Path(directory.name)
        self.seen: list[list] = []
        self.offered: list[list[str]] = []
        self.reply = ModelResponse(parts=[TextPart(content=" where things stand \n")])

    def _answer(self, messages, info) -> ModelResponse:
        self.seen.append(list(messages))
        self.offered.append([tool.name for tool in info.function_tools])
        return self.reply

    async def _recap(self, *history) -> tuple[Agent, str]:
        session = make_session(self.cwd)
        session.append_system_prompt("snapshot")
        # Resume rebuilds the dynamic prompt; pin it so assertions on the
        # system part stay literal.
        with patch("paimon.agent.build_system_prompt", return_value="snapshot"), \
                patch("paimon.agent.build_model", return_value=FunctionModel(self._answer)):
            agent = Agent.open(cwd=self.cwd, session=session, config=Config(model="test:stub"))
            self.addCleanup(agent.close)
            agent.history.extend(history)
            return agent, await agent.recap()

    async def test_it_is_asked_over_the_context_a_turn_would_send(self) -> None:
        _, answer = await self._recap(_user("go"), ModelResponse(parts=[TextPart(content="done")]))

        self.assertEqual(answer, "where things stand")
        sent = self.seen[0]
        self.assertEqual([part.content for part in sent[0].parts
                          if isinstance(part, SystemPromptPart)], ["snapshot"])
        self.assertEqual(_prompts(sent), ["go", RECAP_PROMPT])
        # Tools included: a history full of calls is replayed to a request
        # that still declares them, and the cached prefix stays the same.
        self.assertIn("read_file", self.offered[0])

    async def test_nothing_reaches_the_history_or_the_session_log(self) -> None:
        session = make_session(self.cwd)
        session.append_system_prompt("snapshot")
        with patch("paimon.agent.build_model", return_value=FunctionModel(self._answer)):
            with Agent.open(cwd=self.cwd, session=session, config=Config(model="test:stub")) as agent:
                agent.history.extend([_user("go"), ModelResponse(parts=[TextPart(content="done")])])
                before = list(agent.history)
                lines_before = session.path.read_text().count("\n")

                await agent.recap()

        self.assertEqual(agent.history, before)
        self.assertEqual(session.path.read_text().count("\n"), lines_before)
        self.assertNotIn("[recap]", session.path.read_text())

    async def test_thinking_from_another_model_is_stripped(self) -> None:
        await self._recap(
            _user("go"),
            ModelResponse(parts=[ThinkingPart(content="hm"), TextPart(content="done")],
                          model_name="claude", provider_name="anthropic"))

        thinking = [part for message in self.seen[0] if isinstance(message, ModelResponse)
                    for part in message.parts if isinstance(part, ThinkingPart)]
        self.assertEqual(thinking, [])

    async def test_an_answer_of_nothing_but_a_tool_call_is_no_recap(self) -> None:
        self.reply = ModelResponse(parts=[ToolCallPart(tool_name="shell", args={"command": "ls"},
                                                       tool_call_id="c1")])

        _, answer = await self._recap(_user("go"))

        self.assertEqual(answer, "")
