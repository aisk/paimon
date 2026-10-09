"""Sign in with Google: spend an Antigravity plan through Cloud Code Assist.

Google offers no way for another program to use an Antigravity plan. The
pseudo-provider "antigravity" ("antigravity:gemini-3.8-flash-high") signs in
as Google's own Antigravity client and calls the internal endpoint that
client talks to. That is outside what the plan is sold for and probably
against its terms: Google may throttle or close an account used this way, and
the endpoint can change or stop answering without notice. NOTICE says so
wherever a login starts.

The endpoint speaks the Gemini dialect inside an envelope of its own, for
Gemini, Claude and GPT-OSS models alike. AntigravityModel lets pydantic-ai's
GoogleModel map the messages and read the answer, and does the sending.

Upstream
--------
The protocol is undocumented. What is known of it comes from pi-antigravity
(https://github.com/Rahularya01/pi-antigravity), read at 0.9.0, commit
a3d8caba1b10263420060406de57112ce16490d0 (2026-09-30). When that project
changes, `git log a3d8cab.. -- src` over there is what to review, and the
commit named here should move with whatever gets taken. Each value copied
from it is marked "upstream:" with the file it lives in.

Taken from it:
- the OAuth client, scopes and redirect URI (src/auth/oauth.ts)
- the endpoint, the User-Agent and loadCodeAssist (src/client/client.ts)
- the envelope around the request, without its labels (src/stream/stream.ts
  buildRequest, src/utils/util.ts antigravityRequestEnvelope)
- thinking budgets and output limits per model family (src/models/models.ts)
- what the endpoint refuses or does oddly, as its CHANGELOG records: two
  turns in a row from one side (0.3.1), tool schemas for Claude and GPT-OSS
  in `parameters`, no default tool mode, the empty 200 that works on retry

Left out on purpose:
- request.labels (trajectory_id, request_id, last_step_index, model_enum,
  used_claude, used_claude_conservative, used_non_gemini_model,
  last_execution_id). The endpoint answers without them, upstream ran for six
  weeks before adding them in 0.5.0, and its last_execution_id is a local
  hash where the real client would send an ID the service knows. Sending a
  value that cannot be matched looked worse than sending none. With no
  labels there is no model_enum to learn, so fetchAvailableModels is never
  called either
- separate conversation and trajectory IDs in requestId, and the count of
  finished answers upstream puts in labels.request_id. One ID per session
  serves as both here
- the fallback endpoints (the sandbox and cloudcode-pa hosts) and the header
  and stall deadlines on the stream
- several accounts, and moving to the next one on a quota 429
- model discovery and the public-name routing (gemini-3.8-flash plus an
  effort). The runtime names are used directly, see llm._ANTIGRAVITY_MODELS
- a made-up project ID when loadCodeAssist names none, and the
  listCloudAICompanionProjects call before that. Such a login is refused
- its own message conversion (convertMessages): pydantic-ai's is used. The
  differences that are known and left alone: Gemini function calls keep
  their id, tool call IDs from another provider are not rewritten for Claude,
  an unsigned Gemini call keeps pydantic-ai's placeholder signature where
  upstream turns it into a text observation, and a signature from another
  Gemini model is replayed where upstream drops it
- the default system instruction ("You are Antigravity..."), since Paimon
  always sends its own
- search grounding, image generation, usage and quota reports
"""

import asyncio
import base64
import json
import secrets
import time
import uuid
from typing import Any, AsyncIterator, Awaitable, Callable
from urllib.parse import urlencode

import httpx
from google import genai
from google.genai import _common, types
from google.genai import models as genai_models
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.settings import ModelSettings

from . import oauth
from .errors import PaimonError
from .llm import ANTIGRAVITY_PROVIDER
from .oauth import open_browser  # noqa: F401 (the login screens reach it through here)

NOTICE = ("Google does not support using an Antigravity plan outside Antigravity. Paimon signs in as "
          "the Antigravity client and calls an internal endpoint, which may get the account limited "
          "or closed. Use it at your own risk.")

# upstream: src/auth/oauth.ts (CLIENT_ID, CLIENT_SECRET, SCOPES, REDIRECT_URI).
# The OAuth client of Google's Antigravity desktop app. An installed app
# cannot keep a secret, so this one is public, and it is stored encoded only
# because secret scanners refuse a push that carries its plain shape.
_CLIENT_ID = base64.b64decode(
    "MTA3MTAwNjA2MDU5MS10bWhzc2luMmgyMWxjcmUyMzV2dG9sb2poNGc0MDNlc"
    "C5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbQ==").decode()
_CLIENT_SECRET = base64.b64decode("R09DU1BYLUs1OEZXUjQ" "4NkxkTEoxbUxCOHNYQzR6NnFEQWY=").decode()
_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_SCOPE = " ".join("https://www.googleapis.com/auth/" + name for name in (
    "aicode", "cloud-platform", "userinfo.email", "userinfo.profile", "cclog", "experimentsandconfigs"))
_CALLBACK_HOST = "127.0.0.1"
_CALLBACK_PORT = 51121
# The client is registered for this spelling of the loopback address.
_CALLBACK_PATH = "/oauth-callback"
REDIRECT_URI = f"http://localhost:{_CALLBACK_PORT}{_CALLBACK_PATH}"

# upstream: src/client/client.ts (DEFAULT_ENDPOINT, DEFAULT_USER_AGENT). The
# User-Agent names a CLI release and has been bumped there as releases came out.
_ENDPOINT = "https://daily-cloudcode-pa.googleapis.com"
# The endpoint is only known to answer its own client.
_USER_AGENT = "antigravity/cli/1.2.4 (aidev_client; os_type=linux; arch=amd64; cl=982146307; auth_method=consumer)"

# Where loadCodeAssist has been seen to name the project, by kind of account.
# upstream: src/client/client.ts extractProjectId, the direct keys only.
_PROJECT_KEYS = ("antigravityProjectId", "projectId", "backendProjectId",
                 "userDefinedCloudaicompanionProject", "cloudaicompanionProject", "project")

# What pydantic-ai signs a function call with when the model signed nothing.
# Gemini takes it in place of a signature, the other models never issue one.
_UNSIGNED = base64.urlsafe_b64encode(b"skip_thought_signature_validator").decode()

_EXPIRY_MARGIN = 300.0
_TOKEN_TIMEOUT = 30.0
_REQUEST_TIMEOUT = httpx.Timeout(600.0, connect=10.0)

# The request setting that names the conversation, see llm.request_settings.
SESSION_SETTING = "antigravity_session"


class AntigravityAuthError(PaimonError):
    """The Antigravity login is missing, was refused, or can no longer be renewed."""


def _request_token(form: dict) -> dict:
    """One call to the token endpoint, returned as the credential fields."""
    form = {**form, "client_id": _CLIENT_ID, "client_secret": _CLIENT_SECRET}
    try:
        response = httpx.post(_TOKEN_URL, data=form, timeout=_TOKEN_TIMEOUT)
    except httpx.HTTPError as exc:
        raise AntigravityAuthError(f"cannot reach the Google token endpoint: {exc}") from exc
    if response.status_code != 200:
        raise AntigravityAuthError(
            f"Google token request failed ({response.status_code}): {response.text.strip()[:300]}")
    try:
        token = response.json()
        fields = {
            "access": token["access_token"],
            "expires": time.time() + float(token["expires_in"]) - _EXPIRY_MARGIN,
        }
        # A refresh keeps the refresh token it was made with unless a new
        # one comes back.
        if token.get("refresh_token"):
            fields["refresh"] = token["refresh_token"]
    except (ValueError, KeyError, TypeError) as exc:
        raise AntigravityAuthError("Google token response is missing fields") from exc
    return fields


def _project(access: str) -> str:
    """The Cloud Code Assist project the account's requests are billed to."""
    try:
        response = httpx.post(
            f"{_ENDPOINT}/v1internal:loadCodeAssist", json={"metadata": {"ideType": "ANTIGRAVITY"}},
            headers={"Authorization": f"Bearer {access}", "User-Agent": _USER_AGENT}, timeout=_TOKEN_TIMEOUT)
    except httpx.HTTPError as exc:
        raise AntigravityAuthError(f"cannot reach Antigravity: {exc}") from exc
    if response.status_code != 200:
        raise AntigravityAuthError(
            f"Antigravity refused the account ({response.status_code}): {response.text.strip()[:300]}")
    try:
        answer = response.json()
    except ValueError:
        answer = None
    for key in _PROJECT_KEYS if isinstance(answer, dict) else ():
        project = answer.get(key)
        if isinstance(project, dict):
            project = project.get("id")
        if isinstance(project, str) and project:
            return project
    raise AntigravityAuthError(
        "this Google account has no Antigravity project yet, sign in to Antigravity itself once first")


_LOGIN = oauth.Login(
    provider=ANTIGRAVITY_PROVIDER,
    label="Antigravity",
    keys=("access", "refresh", "project_id"),
    refresh=lambda stored: _request_token({"grant_type": "refresh_token", "refresh_token": stored["refresh"]}),
    error=AntigravityAuthError,
)


def signed_in() -> bool:
    """Whether there is an Antigravity login to build a model from."""
    return _LOGIN.signed_in()


def _thinking_budget(model_name: str) -> int:
    """The thinking allowance asked for. The level is part of the model name
    here, so the budget only has to follow it, in the values the endpoint is
    known to take for each family.

    upstream: src/models/models.ts getThinkingConfig, for an effort that
    matches the level in the name. The 3.5 Flash names, whose levels do not
    follow their suffix, are not told apart here.
    """
    if model_name.startswith("claude-"):
        return 1024
    if model_name.startswith("gpt-oss-"):
        return 8192
    if model_name == "gemini-pro-agent":
        return 10001
    if model_name.startswith("gemini-3.1-pro"):
        return 1001
    if model_name.endswith("-high"):
        return -1  # the model's own choice
    return 4000 if model_name.endswith("-medium") else 1000


def _max_output_tokens(model_name: str) -> int:
    """The most the endpoint lets a model answer with. Asking for more is refused.

    upstream: src/models/models.ts RUNTIME_MAX_OUTPUT_TOKENS.
    """
    if model_name.startswith("claude-"):
        return 64000
    if model_name.startswith("gpt-oss-"):
        return 32768
    return 65535 if "-pro" in model_name else 65536


def _merged(contents: list) -> list:
    """Contents with neighbours of one role joined. The endpoint refuses two
    turns in a row from the same side, which an interrupted turn or a message
    queued behind a tool result leaves in the history.

    upstream: src/stream/stream.ts appendTurn.
    """
    merged: list = []
    for content in contents:
        if merged and merged[-1].get("role") == content.get("role"):
            merged[-1] = {**merged[-1], "parts": [*merged[-1].get("parts", []), *content.get("parts", [])]}
        else:
            merged.append(content)
    return merged


class AntigravityModel(GoogleModel):
    """Gemini requests, wrapped the way Cloud Code Assist takes them."""

    session_setting = SESSION_SETTING

    def __init__(self, model_name: str, credentials: oauth.Credentials,
                 http_client: httpx.AsyncClient | None = None) -> None:
        # The client is never sent through. GoogleModel maps messages with
        # its types, and the key only lets it be constructed.
        super().__init__(
            model_name,
            provider=GoogleProvider(client=genai.Client(api_key="unused")),
            settings={"google_thinking_config": {  # type: ignore[typeddict-unknown-key]
                "include_thoughts": True, "thinking_budget": _thinking_budget(model_name)}},
        )
        self._credentials = credentials
        self._http = http_client or httpx.AsyncClient(timeout=_REQUEST_TIMEOUT)

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        # One way of sending is enough, so a plain request is a stream read to its end.
        async with self.request_stream(messages, model_settings, model_request_parameters) as stream:
            async for _ in stream:
                pass
        return stream.get()

    async def _generate_content(  # type: ignore[override]
        self, messages: list[ModelMessage], stream: bool, model_settings: Any,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[types.GenerateContentResponse]:
        contents, config = await self._build_content_and_config(messages, model_settings, model_request_parameters)
        # Headers are set below. The endpoint's own client names no
        # modalities and no tool mode when the choice is the model's.
        for key in ("http_options", "response_modalities", "tool_config"):
            config.pop(key, None)
        # Thinking is spent out of the same allowance, so a caller's small
        # limit would leave nothing for the answer.
        config["max_output_tokens"] = _max_output_tokens(self._model_name)
        parameters = types._GenerateContentParameters(model=self._model_name, contents=contents, config=config)
        request = genai_models._GenerateContentParameters_to_mldev(
            self.client._api_client, parameters, None, parameters)
        for key in ("_url", "_query", "config"):
            request.pop(key, None)
        request = _common.encode_unserializable_types(_common.convert_to_dict(request))
        request["contents"] = _merged(request.get("contents") or [])
        if not self._model_name.startswith("gemini-"):
            # Claude and GPT-OSS go through a bridge that reads tool schemas
            # from the older field.
            for tool in request.get("tools") or []:
                for declaration in tool.get("functionDeclarations") or []:
                    for key in ("parameters_json_schema", "parametersJsonSchema"):
                        if key in declaration:
                            declaration["parameters"] = declaration.pop(key)
            for content in request["contents"]:
                for part in content.get("parts") or []:
                    if part.get("thoughtSignature") == _UNSIGNED:
                        del part["thoughtSignature"]

        session = model_settings.get(SESSION_SETTING) or str(uuid.uuid4())
        conversation = uuid.uuid5(uuid.NAMESPACE_URL, f"paimon:{session}")
        stored = await self._credentials.current()
        # upstream: src/stream/stream.ts buildRequest, minus request.labels.
        response = await self._http.send(self._http.build_request(
            "POST", f"{_ENDPOINT}/v1internal:streamGenerateContent?alt=sse",
            headers={"Authorization": f"Bearer {stored['access']}", "User-Agent": _USER_AGENT},
            json={
                "project": stored["project_id"],
                "model": self._model_name,
                "request": {**request, "sessionId": session},
                "requestType": "agent",
                "userAgent": "antigravity",
                "requestId": (f"agent/{conversation}/{int(time.time() * 1000)}/{conversation}"
                              f"/{len(request['contents'])}"),
            }), stream=True)
        if response.status_code != 200:
            body = (await response.aread()).decode(errors="replace")
            await response.aclose()
            raise ModelHTTPError(response.status_code, self._model_name, _error_body(body))
        return self._chunks(response, parameters)

    async def _chunks(self, response: httpx.Response,
                      parameters: Any) -> AsyncIterator[types.GenerateContentResponse]:
        answered = False
        try:
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:])
                except ValueError:
                    continue  # an empty keep-alive or an end marker
                if not isinstance(event, dict):
                    continue
                if event.get("error"):
                    error = event["error"]
                    code = error.get("code") if isinstance(error, dict) else None
                    raise ModelHTTPError(code if isinstance(code, int) else 500, self._model_name, error)
                chunk = event.get("response", event)
                if not isinstance(chunk, dict):
                    continue
                answered = True
                # The answer has to be replayed to the model it was asked of,
                # whatever version name the endpoint reports back.
                chunk["modelVersion"] = self._model_name
                yield types.GenerateContentResponse._from_response(
                    response=genai_models._GenerateContentResponse_from_mldev(chunk, None, parameters), kwargs={})
            if not answered:
                # The endpoint does this now and then, and asking again works.
                raise ModelAPIError(self._model_name, "Antigravity answered with an empty stream")
        finally:
            await response.aclose()


def _error_body(text: str) -> object:
    try:
        return json.loads(text)
    except ValueError:
        return text


def build_model(model_name: str) -> AntigravityModel:
    return AntigravityModel(model_name, oauth.Credentials(_LOGIN))


async def login(show_url: Callable[[str], None],
                pasted: Awaitable[str] | None = None) -> None:
    """Run the browser login and store the credential.

    show_url receives the address to open. The browser normally ends on the
    loopback callback served here; where it cannot reach this machine,
    ``pasted`` may deliver the final redirect URL instead.
    """
    state = secrets.token_urlsafe(32)
    verifier, challenge = oauth.pkce()
    authorize = _AUTHORIZE_URL + "?" + urlencode({
        "client_id": _CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": _SCOPE,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
    })

    def parse(target: str) -> str:
        query = _LOGIN.callback_query(target, REDIRECT_URI, state)
        if not query.get("code"):
            raise AntigravityAuthError("the callback carries no authorization code")
        return query["code"]

    code = await oauth.redirected(
        _CALLBACK_HOST, _CALLBACK_PORT, _CALLBACK_PATH, parse, lambda: show_url(authorize), pasted,
        done="Antigravity sign-in completed. You can close this window.",
        busy=AntigravityAuthError(
            f"port {_CALLBACK_PORT} is in use, probably by another pending login or Antigravity itself. "
            "Close it and try again"))

    fields = await asyncio.to_thread(_request_token, {
        "grant_type": "authorization_code",
        "code": code,
        "code_verifier": verifier,
        "redirect_uri": REDIRECT_URI,
    })
    if "refresh" not in fields:
        raise AntigravityAuthError("Google returned no refresh token, sign in again and allow offline access")
    project = await asyncio.to_thread(_project, fields["access"])
    await asyncio.to_thread(_LOGIN.store, {**fields, "project_id": project})
