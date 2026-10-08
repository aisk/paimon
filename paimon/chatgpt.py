"""Sign in with ChatGPT: spend a ChatGPT plan through the OpenAI Responses API.

The pseudo-provider "chatgpt" ("chatgpt:gpt-5.5") is the OpenAI Responses
dialect with two differences. The bearer token comes from an OAuth login
instead of an API key, and the endpoint rejects a list of request fields that
the regular API accepts, so ChatGPTModel shapes every request to fit.

Open-source, locally run apps need no registration: each login asks OpenAI for
a fresh client ID (https://developers.openai.com/siwc/token-sharing-open-source).
The login itself is kept and renewed by paimon.oauth.
"""

import asyncio
import secrets
import time
import uuid
from typing import Awaitable, Callable, Optional
from urllib.parse import urlencode

import httpx
from openai import AsyncOpenAI
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from . import oauth
from .errors import PaimonError
from .llm import CHATGPT_PROVIDER
from .oauth import open_browser  # noqa: F401 (the login screens reach it through here)

# Every login registers a new client under this ID; OpenAI returns the issued
# client ID in the callback, and the token endpoint only knows that one.
_DYNAMIC_CLIENT_ID = "dynamic_agent_client"
_AGENT_NAME = "Paimon"
_AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
_TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
_RESOURCE = "https://api.openai.com/v1"
_CALLBACK_HOST = "127.0.0.1"
_CALLBACK_PORT = 1455
_CALLBACK_PATH = "/auth/callback"
REDIRECT_URI = f"http://{_CALLBACK_HOST}:{_CALLBACK_PORT}{_CALLBACK_PATH}"
_DIRECT_TOKEN_SCOPE = "chatgpt.tokens.use.direct"
_SCOPE = f"openid profile email offline_access resource.invoke {_DIRECT_TOKEN_SCOPE}"

# Refresh this long before the real expiry, so no request starts on a token
# about to lapse.
_EXPIRY_MARGIN = 180.0
_TOKEN_TIMEOUT = 30.0

# Model settings the endpoint rejects outright.
_REJECTED_SETTINGS = frozenset({
    "max_tokens", "temperature", "top_p",
    "openai_user", "openai_truncation", "openai_top_logprobs", "openai_logprobs",
    "openai_prompt_cache_retention", "openai_background", "openai_moderation",
    "openai_previous_response_id", "openai_conversation_id",
})


class ChatGPTAuthError(PaimonError):
    """The ChatGPT login is missing, was refused, or can no longer be renewed."""


def _request_token(form: dict) -> dict:
    """One call to the token endpoint, returned as the credential fields."""
    try:
        response = httpx.post(_TOKEN_URL, data=form, headers={"accept": "application/json"},
                              timeout=_TOKEN_TIMEOUT)
    except httpx.HTTPError as exc:
        raise ChatGPTAuthError(f"cannot reach the ChatGPT token endpoint: {exc}") from exc
    if response.status_code != 200:
        raise ChatGPTAuthError(
            f"ChatGPT token request failed ({response.status_code}): {response.text.strip()[:300]}")
    try:
        token = response.json()
        fields = {
            "access": token["access_token"],
            "refresh": token["refresh_token"],
            "expires": time.time() + float(token["expires_in"]) - _EXPIRY_MARGIN,
        }
        scopes = token["scope"].split()
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ChatGPTAuthError("ChatGPT token response is missing fields") from exc
    if _DIRECT_TOKEN_SCOPE not in scopes:
        raise ChatGPTAuthError(f"ChatGPT did not grant {_DIRECT_TOKEN_SCOPE}, "
                               "this account cannot share its plan with Paimon")
    return fields


_LOGIN = oauth.Login(
    provider=CHATGPT_PROVIDER,
    label="ChatGPT",
    keys=("access", "refresh", "client_id"),
    refresh=lambda stored: _request_token({
        "grant_type": "refresh_token",
        "client_id": stored["client_id"],
        "refresh_token": stored["refresh"],
        "resource": _RESOURCE,
    }),
    error=ChatGPTAuthError,
)


def signed_in(profile: str) -> bool:
    """Whether the profile holds a ChatGPT login to build a model from."""
    return _LOGIN.signed_in(profile)


class Credentials(oauth.Credentials):
    def __init__(self, profile: Optional[str] = None) -> None:
        super().__init__(_LOGIN, profile)


class ChatGPTModel(OpenAIResponsesModel):
    """OpenAI Responses, restricted to what a ChatGPT plan token may send."""

    def prepare_request(
        self, model_settings: Optional[ModelSettings], model_request_parameters: ModelRequestParameters,
    ) -> tuple[Optional[ModelSettings], ModelRequestParameters]:
        settings, parameters = super().prepare_request(model_settings, model_request_parameters)
        allowed = {key: value for key, value in (settings or {}).items()
                   if key not in _REJECTED_SETTINGS}
        allowed["openai_store"] = False
        # The endpoint keeps a conversation on the machine holding its prompt
        # cache by this header. The cache key alone does not route here, and
        # most requests then miss the cache and are charged in full.
        if session := allowed.get("openai_prompt_cache_key"):
            allowed["extra_headers"] = {**(allowed.get("extra_headers") or {}), "session_id": session}
        return allowed, parameters  # type: ignore[return-value]

    async def request(
        self, messages: list[ModelMessage], model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        # The endpoint only answers streamed requests.
        async with self.request_stream(messages, model_settings, model_request_parameters) as stream:
            async for _ in stream:
                pass
        return stream.get()


def build_model(model_name: str, profile: Optional[str] = None) -> ChatGPTModel:
    provider = OpenAIProvider(openai_client=AsyncOpenAI(base_url=_RESOURCE, api_key=Credentials(profile)))
    return ChatGPTModel(
        model_name,
        provider=provider,
        # System-role messages are rejected, developer ones are not.
        profile={**provider.model_profile(model_name), "openai_system_prompt_role": "developer"},
    )


def _host_id(stored: dict) -> str:
    """The stable ID OpenAI knows this installation by, kept across logins."""
    host_id = stored.get("host_id")
    return host_id if isinstance(host_id, str) and host_id else f"urn:uuid:{uuid.uuid4()}"


def _parse_callback(target: str, state: str) -> tuple[str, str]:
    """The (code, issued client ID) carried by a callback URL or request target."""
    query = _LOGIN.callback_query(target, REDIRECT_URI, state)
    if not query.get("code") or not query.get("client_id"):
        raise ChatGPTAuthError("the callback carries no authorization code or client ID")
    return query["code"], query["client_id"]


async def login(profile: Optional[str], show_url: Callable[[str], None],
                pasted: Optional[Awaitable[str]] = None) -> None:
    """Run the browser login and store the credential for the profile.

    show_url receives the address to open. The browser normally ends on the
    loopback callback served here; where it cannot reach this machine,
    ``pasted`` may deliver the final redirect URL instead.
    """
    profile = profile or oauth.DEFAULT_PROFILE
    host_id = _host_id(_LOGIN.read(profile))
    state = secrets.token_urlsafe(32)
    verifier, challenge = oauth.pkce()
    authorize = _AUTHORIZE_URL + "?" + urlencode({
        "client_id": _DYNAMIC_CLIENT_ID,
        "agent_name_hint": _AGENT_NAME,
        "ext_agent_host_id": host_id,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "resource": _RESOURCE,
        "scope": _SCOPE,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "nonce": secrets.token_urlsafe(32),
    })
    code, client_id = await oauth.redirected(
        _CALLBACK_HOST, _CALLBACK_PORT, _CALLBACK_PATH, lambda target: _parse_callback(target, state),
        lambda: show_url(authorize), pasted,
        done="ChatGPT sign-in completed. You can close this window.",
        busy=ChatGPTAuthError(
            f"port {_CALLBACK_PORT} is in use, probably by another pending login or the Codex CLI. "
            "Cancel that login and try again"))

    fields = await asyncio.to_thread(_request_token, {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": code,
        "code_verifier": verifier,
        "redirect_uri": REDIRECT_URI,
        "resource": _RESOURCE,
    })
    await asyncio.to_thread(_LOGIN.store, profile, {"host_id": host_id, "client_id": client_id, **fields})

