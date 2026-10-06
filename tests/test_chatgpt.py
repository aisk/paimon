"""Sign in with ChatGPT: the login, the rotating credential, and the request
shape the plan endpoint insists on. Nothing here reaches OpenAI: requests run
through a MockTransport and the token endpoint is patched."""

import asyncio
import functools
import json
import socket
import time
import unittest
import webbrowser
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

import httpx
from openai import AsyncOpenAI
from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.models import override_allow_model_requests

from paimon import chatgpt
from paimon.llm import ask_once, build_model


def _store(**fields) -> None:
    path = chatgpt.credential_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"access": "at-1", "refresh": "rt-1", "client_id": "client-1",
                                "expires": time.time() + 3600, **fields}))


def _token_response(access: str, refresh: str, scope: str = "openid chatgpt.tokens.use.direct") -> httpx.Response:
    return httpx.Response(200, json={"access_token": access, "refresh_token": refresh,
                                     "expires_in": 3600, "scope": scope})


def _sse(events: list[dict]) -> bytes:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


def _text_stream(text: str) -> bytes:
    response = {"id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-5.5", "status": "in_progress",
                "output": [], "parallel_tool_calls": True, "tool_choice": "auto", "tools": []}
    message = {"id": "msg_1", "type": "message", "role": "assistant", "status": "in_progress", "content": []}
    done = {**message, "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}
    return _sse([
        {"type": "response.created", "sequence_number": 0, "response": response},
        {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": message},
        {"type": "response.output_text.delta", "sequence_number": 2, "output_index": 0, "content_index": 0,
         "item_id": "msg_1", "delta": text, "logprobs": []},
        {"type": "response.output_item.done", "sequence_number": 3, "output_index": 0, "item": done},
        {"type": "response.completed", "sequence_number": 4,
         "response": {**response, "status": "completed", "output": [done],
                      "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                                "input_tokens_details": {"cached_tokens": 0},
                                "output_tokens_details": {"reasoning_tokens": 0}}}},
    ])


class RequestShapeTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_non_streamed_ask_is_sent_the_way_the_plan_endpoint_accepts(self) -> None:
        _store()
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, content=_text_stream("hello"),
                                  headers={"content-type": "text/event-stream"})

        client = functools.partial(
            AsyncOpenAI, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        with patch("paimon.chatgpt.AsyncOpenAI", client):
            model = build_model("chatgpt:gpt-5.5")
        messages = [ModelRequest(parts=[SystemPromptPart(content="be brief"), UserPromptPart(content="hi")])]
        with override_allow_model_requests(True):
            answer = await ask_once(model, messages, max_tokens=64)

        self.assertEqual(answer, "hello")
        request = requests[0]
        self.assertEqual(str(request.url), "https://api.openai.com/v1/responses")
        self.assertEqual(request.headers["authorization"], "Bearer at-1")
        body = json.loads(request.content)
        self.assertIs(body["stream"], True)
        self.assertIs(body["store"], False)
        self.assertNotIn("max_output_tokens", body)
        self.assertIsInstance(body["input"], list)
        self.assertEqual([item["role"] for item in body["input"]], ["developer", "user"])


class CredentialsTest(unittest.IsolatedAsyncioTestCase):
    def test_building_the_model_without_a_login_says_how_to_get_one(self) -> None:
        with self.assertRaisesRegex(chatgpt.ChatGPTAuthError, "paimon login"):
            build_model("chatgpt:gpt-5.5")

    async def test_a_live_token_is_used_without_asking_for_a_new_one(self) -> None:
        _store()
        with patch("paimon.chatgpt.httpx.post", side_effect=AssertionError("no refresh expected")):
            self.assertEqual(await chatgpt.Credentials()(), "at-1")

    async def test_an_expired_token_is_refreshed_and_the_rotation_is_stored(self) -> None:
        _store(expires=time.time() - 1, host_id="urn:uuid:host")
        with patch("paimon.chatgpt.httpx.post", return_value=_token_response("at-2", "rt-2")) as post:
            credentials = chatgpt.Credentials()
            self.assertEqual(await credentials(), "at-2")
            self.assertEqual(await credentials(), "at-2")
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["data"], {
            "grant_type": "refresh_token", "client_id": "client-1", "refresh_token": "rt-1",
            "resource": "https://api.openai.com/v1"})
        stored = json.loads(chatgpt.credential_path().read_text())
        self.assertEqual((stored["access"], stored["refresh"], stored["host_id"]),
                         ("at-2", "rt-2", "urn:uuid:host"))

    async def test_a_refresh_another_process_already_did_is_not_repeated(self) -> None:
        _store(expires=time.time() - 1)
        credentials = chatgpt.Credentials()
        _store(access="at-theirs", refresh="rt-theirs")
        with patch("paimon.chatgpt.httpx.post", side_effect=AssertionError("no refresh expected")):
            self.assertEqual(await credentials(), "at-theirs")

    async def test_a_refused_refresh_is_an_auth_error(self) -> None:
        _store(expires=time.time() - 1)
        refused = httpx.Response(400, json={"error": "refresh_token_invalidated"})
        with patch("paimon.chatgpt.httpx.post", return_value=refused):
            with self.assertRaisesRegex(chatgpt.ChatGPTAuthError, "refresh_token_invalidated"):
                await chatgpt.Credentials()()


class LoginTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        port = patch("paimon.chatgpt._CALLBACK_PORT", self.port)
        port.start()
        self.addCleanup(port.stop)

    async def _login(self, respond, pasted=None) -> list:
        """Run a login; ``respond`` gets the authorization query once the URL is shown."""
        shown: list = []

        def show_url(url: str) -> None:
            shown.append(url)
            asyncio.get_running_loop().create_task(respond(parse_qs(urlsplit(url).query)))

        await chatgpt.login(None, show_url, pasted)
        return shown

    async def _browser(self, query: str) -> bytes:
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write(f"GET /auth/callback?{query} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        answer = await reader.read()
        writer.close()
        return answer

    async def test_the_browser_callback_completes_the_login(self) -> None:
        async def respond(query: dict) -> None:
            self.assertIn(b"400", await self._browser("code=stale&state=other&client_id=x"))
            self.assertIn(b"200 OK", await self._browser(f"code=c0de&client_id=issued-1&state={query['state'][0]}"))

        with patch("paimon.chatgpt.httpx.post", return_value=_token_response("at-1", "rt-1")) as post:
            shown = await self._login(respond)

        query = parse_qs(urlsplit(shown[0]).query)
        self.assertEqual(query["client_id"], ["dynamic_agent_client"])
        self.assertIn("chatgpt.tokens.use.direct", query["scope"][0])
        self.assertTrue(query["ext_agent_host_id"][0].startswith("urn:uuid:"))
        form = post.call_args.kwargs["data"]
        self.assertEqual((form["grant_type"], form["code"], form["client_id"]),
                         ("authorization_code", "c0de", "issued-1"))
        stored = json.loads(chatgpt.credential_path().read_text())
        self.assertEqual((stored["access"], stored["refresh"], stored["client_id"]),
                         ("at-1", "rt-1", "issued-1"))
        self.assertEqual(stored["host_id"], query["ext_agent_host_id"][0])

    async def test_a_pasted_redirect_url_completes_the_login_and_keeps_the_host_id(self) -> None:
        _store(host_id="urn:uuid:kept")
        pasted = asyncio.get_running_loop().create_future()

        async def respond(query: dict) -> None:
            pasted.set_result(f"{chatgpt.REDIRECT_URI}?code=c0de&client_id=issued-2&state={query['state'][0]}\n")

        with patch("paimon.chatgpt.httpx.post", return_value=_token_response("at-9", "rt-9")):
            await self._login(respond, pasted)
        stored = json.loads(chatgpt.credential_path().read_text())
        self.assertEqual((stored["access"], stored["client_id"], stored["host_id"]),
                         ("at-9", "issued-2", "urn:uuid:kept"))

    async def test_a_denied_authorization_fails_the_login(self) -> None:
        async def respond(query: dict) -> None:
            await self._browser("error=access_denied")

        with self.assertRaisesRegex(chatgpt.ChatGPTAuthError, "access_denied"):
            await self._login(respond)
        self.assertFalse(chatgpt.credential_path().exists())

    async def test_a_plan_that_cannot_be_shared_is_refused(self) -> None:
        async def respond(query: dict) -> None:
            await self._browser(f"code=c0de&client_id=issued-1&state={query['state'][0]}")

        with patch("paimon.chatgpt.httpx.post", return_value=_token_response("at", "rt", scope="openid")):
            with self.assertRaisesRegex(chatgpt.ChatGPTAuthError, "did not grant"):
                await self._login(respond)
        self.assertFalse(chatgpt.credential_path().exists())

    async def test_a_taken_callback_port_is_reported(self) -> None:
        with socket.socket() as holder:
            holder.bind(("127.0.0.1", self.port))
            holder.listen()
            with self.assertRaisesRegex(chatgpt.ChatGPTAuthError, "in use"):
                await chatgpt.login(None, lambda url: None)


class OpenBrowserTest(unittest.TestCase):
    def _open(self, name: str) -> tuple[bool, Mock]:
        controller = Mock()
        controller.name = name
        controller.open.return_value = True
        with patch("paimon.chatgpt.webbrowser.get", return_value=controller):
            return chatgpt.open_browser("https://example/"), controller

    def test_a_graphical_browser_is_opened(self) -> None:
        opened, controller = self._open("xdg-open")
        self.assertTrue(opened)
        controller.open.assert_called_once_with("https://example/")

    def test_terminal_browsers_are_never_launched(self) -> None:
        for name in ("w3m", "lynx", "www-browser", "/usr/bin/elinks", "links -g"):
            with self.subTest(name=name):
                opened, controller = self._open(name)
                self.assertFalse(opened)
                controller.open.assert_not_called()

    def test_no_browser_at_all_is_not_an_error(self) -> None:
        with patch("paimon.chatgpt.webbrowser.get", side_effect=webbrowser.Error("none")):
            self.assertFalse(chatgpt.open_browser("https://example/"))
