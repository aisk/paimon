"""Reasoning round-trip: the point of the pydantic-ai migration.

GLM's preserved-thinking mode requires each assistant turn's reasoning_content
to be sent back complete and verbatim in the message history. These tests run
the real ZaiModel request pipeline against a mocked HTTP endpoint and inspect
the raw request bodies.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from helpers import make_session
from openai import AsyncOpenAI
from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart
from pydantic_ai.models.zai import ZaiModel
from pydantic_ai.providers.zai import ZaiProvider

from paimon.agent import Agent, ReasoningDelta, _strip_foreign_thinking
from paimon.config import Config


def _sse(chunks: list[dict]) -> bytes:
    return ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()


def _chunk(delta: dict, finish: str | None = None) -> dict:
    return {"id": "1", "object": "chat.completion.chunk", "created": 0, "model": "glm-5.2",
            "choices": [{"index": 0, "delta": delta, **({"finish_reason": finish} if finish else {})}]}


def _zai_model(handler) -> ZaiModel:
    client = AsyncOpenAI(
        base_url="https://api.z.ai/api/coding/paas/v4",
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return ZaiModel("glm-5.2", provider=ZaiProvider(openai_client=client))


class ReasoningRoundTripTest(unittest.IsolatedAsyncioTestCase):
    async def test_second_request_carries_reasoning_content_back(self) -> None:
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                chunks = [
                    _chunk({"role": "assistant", "reasoning_content": "I should "}),
                    _chunk({"reasoning_content": "add a todo."}),
                    _chunk({"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {
                        "name": "write_todos",
                        "arguments": '{"todos": [{"content": "x", "status": "pending"}]}'}}]}),
                    _chunk({}, finish="tool_calls"),
                ]
            else:
                chunks = [_chunk({"role": "assistant", "content": "done"}), _chunk({}, finish="stop")]
            return httpx.Response(200, content=_sse(chunks), headers={"content-type": "text/event-stream"})

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="zai/glm-5.2"))

            with patch("paimon.agent.build_model", return_value=_zai_model(handler)):
                events = [event async for event in agent.run("plan it")]

            reasoning = "".join(event.text for event in events if isinstance(event, ReasoningDelta))
            self.assertEqual(reasoning, "I should add a todo.")
            self.assertEqual(len(requests), 2)

            assistants = [m for m in requests[1]["messages"] if m["role"] == "assistant"]
            self.assertEqual(len(assistants), 1)
            self.assertEqual(assistants[0]["reasoning_content"], "I should add a todo.")
            self.assertEqual(assistants[0]["tool_calls"][0]["function"]["name"], "write_todos")
            # Z.ai preserved thinking stays enabled across turns.
            self.assertEqual(requests[1]["thinking"], {"clear_thinking": False})

    async def test_resumed_session_still_replays_reasoning_to_same_model(self) -> None:
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            chunks = [_chunk({"role": "assistant", "content": "hi"}), _chunk({}, finish="stop")]
            return httpx.Response(200, content=_sse(chunks), headers={"content-type": "text/event-stream"})

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            first = Agent.open(cwd=cwd, session=session, config=Config(model="zai/glm-5.2"))
            first._append_message(ModelResponse(
                parts=[ThinkingPart(content="earlier thoughts", id="reasoning_content", provider_name="zai"),
                       TextPart(content="earlier answer")],
                model_name="glm-5.2", provider_name="zai",
            ))

            # A session may only be open once at a time, here as in a pane.
            first.session.unlock()
            resumed = Agent.open(cwd=cwd, session=session, config=Config(model="zai/glm-5.2"))
            with patch("paimon.agent.build_model", return_value=_zai_model(handler)):
                _ = [event async for event in resumed.run("continue")]

            assistants = [m for m in requests[0]["messages"] if m["role"] == "assistant"]
            self.assertEqual(assistants[0]["reasoning_content"], "earlier thoughts")


class StripForeignThinkingTest(unittest.IsolatedAsyncioTestCase):
    async def test_other_models_thinking_is_stripped_from_requests(self) -> None:
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            chunks = [_chunk({"role": "assistant", "content": "ok"}), _chunk({}, finish="stop")]
            return httpx.Response(200, content=_sse(chunks), headers={"content-type": "text/event-stream"})

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="zai/glm-5.2"))
            agent._append_message(ModelResponse(
                parts=[ThinkingPart(content="deepseek thoughts", id="reasoning_content",
                                    provider_name="deepseek"),
                       TextPart(content="old answer")],
                model_name="deepseek-reasoner", provider_name="deepseek",
            ))

            with patch("paimon.agent.build_model", return_value=_zai_model(handler)):
                _ = [event async for event in agent.run("hi")]

            body = json.dumps(requests[0]["messages"])
            self.assertNotIn("deepseek thoughts", body)
            self.assertIn("old answer", body)

    def test_matching_model_keeps_thinking(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # never called
            raise AssertionError

        model = _zai_model(handler)
        matching = ModelResponse(
            parts=[ThinkingPart(content="keep me"), TextPart(content="a")],
            model_name="glm-5.2", provider_name="zai",
        )
        foreign = ModelResponse(
            parts=[ThinkingPart(content="drop me"), TextPart(content="b")],
            model_name="glm-5", provider_name="zai",
        )
        stripped = _strip_foreign_thinking([matching, foreign], model)
        self.assertTrue(any(isinstance(p, ThinkingPart) for p in stripped[0].parts))
        self.assertFalse(any(isinstance(p, ThinkingPart) for p in stripped[1].parts))


if __name__ == "__main__":
    unittest.main()
