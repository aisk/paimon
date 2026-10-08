"""Sign in with Google for an Antigravity plan: the login, and the envelope
Cloud Code Assist takes a Gemini request in. Nothing here reaches Google:
requests run through a MockTransport and the token endpoint is patched."""

import asyncio
import base64
import json
import socket
import time
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import httpx
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import (ModelRequest, ModelResponse, SystemPromptPart, ThinkingPart, ToolCallPart,
                                  ToolReturnPart, UserPromptPart)
from pydantic_ai.models import ModelRequestParameters, override_allow_model_requests
from pydantic_ai.tools import ToolDefinition

from paimon import antigravity, oauth, retry
from paimon.config import read_provider, update_provider
from paimon.llm import ask_once, build_model, request_settings

_SIGNATURE = base64.b64encode(b"\xff\xfesigned").decode()
_TOOL = ToolDefinition(name="read", description="Read a file", parameters_json_schema={
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})


def _store(**fields) -> None:
    update_provider("default", "antigravity", {"access": "at-1", "refresh": "rt-1", "project_id": "proj-1",
                                               "expires": time.time() + 3600, **fields})


def _stored() -> dict:
    return read_provider("default", "antigravity")


def _sse(*chunks: dict) -> bytes:
    return "".join(f"data: {json.dumps({'response': chunk})}\n\n" for chunk in chunks).encode()


def _chunk(*parts: dict, finish: bool = False) -> dict:
    candidate: dict = {"content": {"role": "model", "parts": list(parts)}}
    if finish:
        candidate["finishReason"] = "STOP"
    return {"candidates": [candidate], "modelVersion": "something-else", "responseId": "r-1",
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3, "totalTokenCount": 10}}


def _model(name: str, handler) -> antigravity.AntigravityModel:
    return antigravity.AntigravityModel(
        name, oauth.Credentials(antigravity._LOGIN),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


class RequestShapeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _store()
        self.requests: list[httpx.Request] = []
        allow = override_allow_model_requests(True)
        allow.__enter__()
        self.addCleanup(allow.__exit__, None, None, None)

    def _answering(self, *chunks: dict):
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, content=_sse(*chunks), headers={"content-type": "text/event-stream"})
        return handler

    async def test_a_request_goes_out_in_the_cloud_code_assist_envelope(self) -> None:
        model = _model("gemini-3.8-flash-high", self._answering(_chunk({"text": "hello"}, finish=True)))
        messages = [ModelRequest(parts=[SystemPromptPart(content="be brief"), UserPromptPart(content="hi")])]
        self.assertEqual(await ask_once(model, messages, max_tokens=64, tools=[_TOOL], cache_key="session-1"),
                         "hello")

        request = self.requests[0]
        self.assertEqual(str(request.url),
                         "https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse")
        self.assertEqual(request.headers["authorization"], "Bearer at-1")
        self.assertTrue(request.headers["user-agent"].startswith("antigravity/"))
        body = json.loads(request.content)
        self.assertEqual((body["project"], body["model"], body["requestType"], body["userAgent"]),
                         ("proj-1", "gemini-3.8-flash-high", "agent", "antigravity"))
        self.assertTrue(body["requestId"].startswith("agent/"))
        self.assertTrue(body["requestId"].endswith("/1"), "the step is the number of contents")
        inner = body["request"]
        self.assertEqual(inner["sessionId"], "session-1")
        self.assertEqual(inner["contents"], [{"role": "user", "parts": [{"text": "hi"}]}])
        self.assertEqual(inner["systemInstruction"]["parts"], [{"text": "be brief"}])
        # The nested keys go out the way the google-genai SDK spells them,
        # which the service reads as it does the camel case ones.
        self.assertEqual(inner["generationConfig"], {
            "maxOutputTokens": 65536, "thinkingConfig": {"include_thoughts": True, "thinking_budget": -1}})
        self.assertNotIn("toolConfig", inner)
        self.assertEqual(inner["tools"], [{"functionDeclarations": [{
            "name": "read", "description": "Read a file", "parameters_json_schema": _TOOL.parameters_json_schema}]}])

    async def test_a_callers_small_limit_does_not_starve_the_answer_of_thinking_room(self) -> None:
        answer = _chunk({"text": "ok"}, finish=True)
        for name, limit, budget in (("gemini-3.8-flash-medium", 65536, 4000), ("gemini-pro-agent", 65535, 10001),
                                    ("gemini-3.1-pro-low", 65535, 1001), ("claude-opus-4-6-thinking", 64000, 1024),
                                    ("gpt-oss-120b-medium", 32768, 8192)):
            with self.subTest(model=name):
                self.requests.clear()
                model = _model(name, self._answering(answer))
                await ask_once(model, [ModelRequest(parts=[UserPromptPart(content="hi")])], max_tokens=2048)
                config = json.loads(self.requests[0].content)["request"]["generationConfig"]
                self.assertEqual((config["maxOutputTokens"], config["thinkingConfig"]["thinking_budget"]),
                                 (limit, budget))

    async def test_two_turns_in_a_row_from_one_side_go_out_as_one(self) -> None:
        model = _model("gemini-3.8-flash-low", self._answering(_chunk({"text": "ok"}, finish=True)))
        call = ToolCallPart(tool_name="read", args={"path": "a.py"}, tool_call_id="call-1")
        messages = [
            ModelRequest(parts=[UserPromptPart(content="interrupted")]),
            ModelRequest(parts=[UserPromptPart(content="read a.py")]),
            ModelResponse(parts=[call], provider_name="google", model_name="gemini-3.8-flash-low"),
            ModelRequest(parts=[ToolReturnPart(tool_name="read", content="x = 1", tool_call_id="call-1"),
                                UserPromptPart(content="and b.py")]),
        ]
        await ask_once(model, messages, max_tokens=8, tools=[_TOOL])
        body = json.loads(self.requests[0].content)
        contents = body["request"]["contents"]
        self.assertEqual([content["role"] for content in contents], ["user", "model", "user"])
        self.assertEqual(contents[0]["parts"], [{"text": "interrupted"}, {"text": "read a.py"}])
        self.assertEqual([sorted(part) for part in contents[2]["parts"]], [["functionResponse"], ["text"]])
        self.assertTrue(body["requestId"].endswith("/3"))

    async def test_the_same_session_keeps_one_conversation_id(self) -> None:
        model = _model("gemini-3.8-flash-low", self._answering(_chunk({"text": "ok"}, finish=True)))
        messages = [ModelRequest(parts=[UserPromptPart(content="hi")])]
        for key in ("session-1", "session-1", "session-2"):
            await ask_once(model, messages, max_tokens=8, cache_key=key)
        first, second, other = (json.loads(request.content)["requestId"].split("/")[1]
                                for request in self.requests)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    async def test_claude_gets_its_tool_schemas_in_the_older_field(self) -> None:
        model = _model("claude-sonnet-4-6", self._answering(_chunk({"text": "ok"}, finish=True)))
        await ask_once(model, [ModelRequest(parts=[UserPromptPart(content="hi")])], max_tokens=8, tools=[_TOOL])
        declaration = json.loads(self.requests[0].content)["request"]["tools"][0]["functionDeclarations"][0]
        self.assertEqual(declaration["parameters"]["required"], ["path"])
        self.assertEqual([key for key in declaration if "schema" in key.lower()], [])

    async def test_a_call_the_model_did_not_sign_is_replayed_unsigned_to_claude_only(self) -> None:
        call = ToolCallPart(tool_name="read", args={"path": "a.py"}, tool_call_id="call-1")
        for name, signed in (("claude-sonnet-4-6", False), ("gemini-3.8-flash-low", True)):
            with self.subTest(model=name):
                self.requests.clear()
                model = _model(name, self._answering(_chunk({"text": "ok"}, finish=True)))
                await ask_once(model, [
                    ModelRequest(parts=[UserPromptPart(content="read a.py")]),
                    ModelResponse(parts=[call], provider_name="google", model_name=name),
                    ModelRequest(parts=[ToolReturnPart(tool_name="read", content="x", tool_call_id="call-1")]),
                ], max_tokens=8, tools=[_TOOL])
                replayed = json.loads(self.requests[0].content)["request"]["contents"][1]["parts"][0]
                self.assertEqual(replayed["functionCall"]["name"], "read")
                self.assertEqual("thoughtSignature" in replayed, signed)

    async def test_thinking_and_tool_calls_stream_back_and_replay_with_their_signature(self) -> None:
        model = _model("gemini-3.8-flash-high", self._answering(
            _chunk({"text": "let me look", "thought": True}),
            _chunk({"functionCall": {"name": "read", "args": {"path": "a.py"}}, "thoughtSignature": _SIGNATURE},
                   finish=True)))
        asked = ModelRequest(parts=[UserPromptPart(content="read a.py")])
        async with model_request_stream(
                model, [asked], model_settings=request_settings(model, "session-1"),
                model_request_parameters=ModelRequestParameters(function_tools=[_TOOL])) as stream:
            async for _ in stream:
                pass
        answer = stream.get()
        self.assertIsInstance(answer, ModelResponse)
        self.assertEqual(answer.model_name, "gemini-3.8-flash-high", "not the version the endpoint reported")
        self.assertEqual((answer.usage.input_tokens, answer.usage.output_tokens), (7, 3))
        thinking = [part for part in answer.parts if isinstance(part, ThinkingPart)]
        self.assertEqual([part.content for part in thinking], ["let me look"])
        call = next(part for part in answer.parts if isinstance(part, ToolCallPart))
        self.assertEqual((call.tool_name, call.args_as_dict()), ("read", {"path": "a.py"}))

        returned = ModelRequest(parts=[ToolReturnPart(tool_name="read", content="x = 1",
                                                      tool_call_id=call.tool_call_id)])
        await ask_once(model, [asked, answer, returned], max_tokens=8, tools=[_TOOL])
        contents = json.loads(self.requests[1].content)["request"]["contents"]
        self.assertEqual([content["role"] for content in contents], ["user", "model", "user"])
        replayed = next(part for part in contents[1]["parts"] if "functionCall" in part)
        # The SDK's URL-safe spelling of the same bytes.
        self.assertEqual(replayed["thoughtSignature"], base64.urlsafe_b64encode(b"\xff\xfesigned").decode())
        self.assertEqual(contents[2]["parts"][0]["functionResponse"]["name"], "read")

    async def test_a_refused_request_is_an_http_error_the_retry_policy_can_read(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": {"code": 429, "message": "Quota reached."}})

        model = _model("gemini-3.8-flash-high", handler)
        with self.assertRaises(ModelHTTPError) as raised:
            await ask_once(model, [ModelRequest(parts=[UserPromptPart(content="hi")])], max_tokens=8)
        self.assertEqual(raised.exception.status_code, 429)
        self.assertIn("Quota reached", str(raised.exception))

    async def test_lines_that_carry_no_answer_are_passed_over(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"data:\n\ndata: not json\n\ndata: {\"response\": null}\n\n"
                                  + _sse(_chunk({"text": "hello"}, finish=True)) + b"data: [DONE]\n\n")

        model = _model("gemini-3.8-flash-high", handler)
        self.assertEqual(
            await ask_once(model, [ModelRequest(parts=[UserPromptPart(content="hi")])], max_tokens=8), "hello")

    async def test_an_empty_stream_is_a_failure_worth_retrying(self) -> None:
        model = _model("gemini-3.8-flash-high", lambda request: httpx.Response(200, content=b""))
        with self.assertRaises(ModelAPIError) as raised:
            await ask_once(model, [ModelRequest(parts=[UserPromptPart(content="hi")])], max_tokens=8)
        self.assertTrue(retry.is_transient(raised.exception))

    async def test_an_error_sent_inside_the_stream_is_raised(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b'data: {"error": {"code": 503, "message": "overloaded"}}\n\n')

        model = _model("gemini-3.8-flash-high", handler)
        with self.assertRaises(ModelHTTPError) as raised:
            await ask_once(model, [ModelRequest(parts=[UserPromptPart(content="hi")])], max_tokens=8)
        self.assertEqual(raised.exception.status_code, 503)


class CredentialsTest(unittest.IsolatedAsyncioTestCase):
    def test_building_the_model_without_a_login_says_how_to_get_one(self) -> None:
        with self.assertRaisesRegex(antigravity.AntigravityAuthError, "paimon login --model antigravity:"):
            build_model("antigravity:gemini-3.8-flash-high")

    async def test_an_expired_token_is_refreshed_and_keeps_its_refresh_token(self) -> None:
        _store(expires=time.time() - 1)
        renewed = httpx.Response(200, json={"access_token": "at-2", "expires_in": 3600})
        with patch("paimon.antigravity.httpx.post", return_value=renewed) as post:
            credentials = oauth.Credentials(antigravity._LOGIN)
            self.assertEqual(await credentials(), "at-2")
            self.assertEqual(await credentials(), "at-2")
        post.assert_called_once()
        form = post.call_args.kwargs["data"]
        self.assertEqual((form["grant_type"], form["refresh_token"]), ("refresh_token", "rt-1"))
        stored = _stored()
        self.assertEqual((stored["access"], stored["refresh"], stored["project_id"]), ("at-2", "rt-1", "proj-1"))


class LoginTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        port = patch("paimon.antigravity._CALLBACK_PORT", self.port)
        port.start()
        self.addCleanup(port.stop)

    async def _login(self, project: object) -> tuple[dict, list]:
        """Sign in with the browser coming straight back; ``project`` is what
        loadCodeAssist answers. Returns the authorization query and the posts."""
        shown: list = []
        posts: list = []

        async def browser(state: str) -> None:
            reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
            writer.write(f"GET /oauth-callback?code=c0de&state={state} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
            await reader.read()
            writer.close()

        def show_url(url: str) -> None:
            shown.append(url)
            state = parse_qs(urlsplit(url).query)["state"][0]
            asyncio.get_running_loop().create_task(browser(state))

        def post(url, **kwargs) -> httpx.Response:
            posts.append((url, kwargs))
            if "oauth2" in url:
                return httpx.Response(200, json={"access_token": "at-1", "refresh_token": "rt-1",
                                                 "expires_in": 3600})
            return httpx.Response(200, json=project)

        with patch("paimon.antigravity.httpx.post", side_effect=post):
            await antigravity.login(None, show_url)
        return parse_qs(urlsplit(shown[0]).query), posts

    async def test_the_browser_callback_completes_the_login_and_finds_the_project(self) -> None:
        query, posts = await self._login({"cloudaicompanionProject": {"id": "proj-9"}})
        self.assertEqual((query["access_type"], query["code_challenge_method"]), (["offline"], ["S256"]))
        self.assertIn("cloud-platform", query["scope"][0])
        form = posts[0][1]["data"]
        self.assertEqual((form["grant_type"], form["code"]), ("authorization_code", "c0de"))
        self.assertEqual((form["redirect_uri"], form["client_id"]),
                         (antigravity.REDIRECT_URI, query["client_id"][0]))
        self.assertTrue(form["client_secret"] and len(form["code_verifier"]) >= 43)
        self.assertTrue(posts[1][0].endswith("/v1internal:loadCodeAssist"))
        self.assertEqual(posts[1][1]["headers"]["Authorization"], "Bearer at-1")
        stored = _stored()
        self.assertEqual((stored["access"], stored["refresh"], stored["project_id"]), ("at-1", "rt-1", "proj-9"))

    async def test_the_project_is_found_wherever_the_account_keeps_it(self) -> None:
        await self._login({"projectId": "proj-direct"})
        self.assertEqual(_stored()["project_id"], "proj-direct")

    async def test_an_account_without_a_project_is_not_stored(self) -> None:
        with self.assertRaisesRegex(antigravity.AntigravityAuthError, "no Antigravity project"):
            await self._login({})
        self.assertEqual(_stored(), {})
