"""Sign in with ChatGPT: spend a ChatGPT plan through the OpenAI Responses API.

The pseudo-provider "chatgpt" ("chatgpt:gpt-5.5") is the OpenAI Responses
dialect with two differences. The bearer token comes from an OAuth login
instead of an API key, and the endpoint rejects a list of request fields that
the regular API accepts, so ChatGPTModel shapes every request to fit.

Open-source, locally run apps need no registration: each login asks OpenAI for
a fresh client ID (https://developers.openai.com/siwc/token-sharing-open-source).
The credential lives in the profile's config.json, under
"providers": {"chatgpt": ...}, and is rewritten whenever the refresh token
rotates.
"""

import asyncio
import base64
import hashlib
import secrets
import threading
import time
import uuid
import webbrowser
from contextlib import contextmanager
from pathlib import Path
from typing import Awaitable, Callable, Iterator, Optional
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
from openai import AsyncOpenAI
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from . import lockfile
from .config import DEFAULT_PROFILE, ConfigError, config_dir, read_provider, update_provider
from .errors import PaimonError
from .llm import CHATGPT_PROVIDER

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
_REFRESH_LOCK_TIMEOUT = 90.0
_TOKEN_TIMEOUT = 30.0
_STORE_ATTEMPTS = 3
_STORE_RETRY_SECONDS = 1.0

# The sidecar lock keeps processes apart but is reentrant within one, and
# every Agent builds its own model, so threads need their own exclusion. Two
# refreshes racing on one refresh token leave the loser's login revoked.
_REFRESH_MUTEX = threading.Lock()

# Model settings the endpoint rejects outright.
_REJECTED_SETTINGS = frozenset({
    "max_tokens", "temperature", "top_p",
    "openai_user", "openai_truncation", "openai_top_logprobs", "openai_logprobs",
    "openai_prompt_cache_retention", "openai_background", "openai_moderation",
    "openai_previous_response_id", "openai_conversation_id",
})

# Browsers that draw in the terminal. They cannot get through the sign-in
# page, and one launched from the TUI takes the screen over.
_TERMINAL_BROWSERS = frozenset({
    "www-browser", "links", "links2", "elinks", "lynx", "w3m", "browsh", "carbonyl"})


def open_browser(url: str) -> bool:
    """Open url in a graphical browser. False when there is none to open,
    leaving the caller's printed address as the way in."""
    try:
        controller = webbrowser.get()
    except webbrowser.Error:
        return False
    command = (getattr(controller, "name", "") or "").split()
    if command and Path(command[0]).name in _TERMINAL_BROWSERS:
        return False
    try:
        return bool(controller.open(url))
    except (OSError, webbrowser.Error):
        return False


class ChatGPTAuthError(PaimonError):
    """The ChatGPT login is missing, was refused, or can no longer be renewed."""


def _read(profile: str) -> dict:
    return read_provider(profile, CHATGPT_PROVIDER)


def _signed_in(stored: dict) -> bool:
    return all(isinstance(stored.get(key), str) and stored[key]
               for key in ("access", "refresh", "client_id"))


def _expired(stored: dict) -> bool:
    expires = stored.get("expires")
    return not isinstance(expires, (int, float)) or time.time() >= expires


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


@contextmanager
def _credential_lock(profile: str) -> Iterator[None]:
    """Held by whoever replaces the stored tokens, a refresh or a login.

    Not the config's own lock: this one is held across a network call, and an
    unrelated save (a theme change) must not wait that out.
    """
    lock = config_dir(profile) / "chatgpt.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with _REFRESH_MUTEX:
        if not lockfile.acquire(lock, _REFRESH_LOCK_TIMEOUT):
            raise ChatGPTAuthError(f"another Paimon held {lock} for over {_REFRESH_LOCK_TIMEOUT:.0f}s")
        try:
            yield
        finally:
            lockfile.release(lock)


def _store_rotation(profile: str, fields: dict) -> dict:
    """Write freshly rotated tokens, trying again if the config cannot be
    written. The old refresh token is already dead, so a write given up on
    is a login lost."""
    for attempt in range(_STORE_ATTEMPTS):
        try:
            return update_provider(profile, CHATGPT_PROVIDER, fields)
        except ConfigError:
            if attempt == _STORE_ATTEMPTS - 1:
                raise
            time.sleep(_STORE_RETRY_SECONDS)
    raise AssertionError("unreachable")


def _renewed(profile: str) -> dict:
    """The stored credential, refreshed first if it has expired.

    Synchronous and run on a thread: a turn cancelled mid-refresh must still
    get the rotated refresh token onto disk, or the login is lost.
    """
    with _credential_lock(profile):
        # Re-read inside the lock: another process may have refreshed.
        stored = _read(profile)
        if not _signed_in(stored):
            raise ChatGPTAuthError("not signed in to ChatGPT, log in again")
        if _expired(stored):
            stored = _store_rotation(profile, _request_token({
                "grant_type": "refresh_token",
                "client_id": stored["client_id"],
                "refresh_token": stored["refresh"],
                "resource": _RESOURCE,
            }))
        return stored


def _store_login(profile: str, credential: dict) -> None:
    # Under the lock, so a refresh still in flight for the previous login
    # lands first and cannot mix its tokens into this client's entry.
    with _credential_lock(profile):
        update_provider(profile, CHATGPT_PROVIDER, credential)


class Credentials:
    """The bearer token for one profile, as the callable AsyncOpenAI asks
    before every request."""

    def __init__(self, profile: Optional[str] = None) -> None:
        self.profile = profile or DEFAULT_PROFILE
        self._stored = _read(self.profile)
        if not _signed_in(self._stored):
            raise ChatGPTAuthError(
                f"not signed in to ChatGPT, run 'paimon login --model {CHATGPT_PROVIDER}:MODEL'")

    async def __call__(self) -> str:
        if _expired(self._stored):
            self._stored = await asyncio.to_thread(_renewed, self.profile)
        return self._stored["access"]


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
    parts = urlsplit(target.strip())
    if parts.path != _CALLBACK_PATH:
        raise ChatGPTAuthError(f"the callback URL must start with {REDIRECT_URI}")
    query = {key: values[0] for key, values in parse_qs(parts.query).items()}
    if query.get("error"):
        raise ChatGPTAuthError(f"ChatGPT authorization failed: {query['error']}")
    if query.get("state") != state:
        raise ChatGPTAuthError("the callback belongs to a different login attempt")
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
    profile = profile or DEFAULT_PROFILE
    host_id = _host_id(_read(profile))
    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    callback: asyncio.Future = asyncio.get_running_loop().create_future()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = (await reader.readline()).decode("latin-1").split()
            target = request_line[1] if len(request_line) > 1 else ""
            if urlsplit(target).path != _CALLBACK_PATH:
                status, body = "404 Not Found", "Not found."
            else:
                try:
                    result = _parse_callback(target, state)
                except ChatGPTAuthError as exc:
                    status, body = "400 Bad Request", str(exc)
                    # A stale tab from an earlier attempt must not end this one.
                    if "error=" in target and not callback.done():
                        callback.set_exception(exc)
                else:
                    status, body = "200 OK", "ChatGPT sign-in completed. You can close this window."
                    if not callback.done():
                        callback.set_result(result)
            payload = body.encode()
            writer.write(f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
                         f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode() + payload)
            await writer.drain()
        finally:
            writer.close()

    try:
        server = await asyncio.start_server(serve, _CALLBACK_HOST, _CALLBACK_PORT)
    except OSError as exc:
        raise ChatGPTAuthError(
            f"port {_CALLBACK_PORT} is in use, probably by another pending login or the Codex CLI. "
            "Cancel that login and try again") from exc

    async def from_paste() -> None:
        result = _parse_callback(await pasted, state)  # type: ignore[misc]
        if not callback.done():
            callback.set_result(result)

    paste_task = asyncio.ensure_future(from_paste()) if pasted is not None else None
    try:
        show_url(_AUTHORIZE_URL + "?" + urlencode({
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
        }))
        waiting = {callback} | ({paste_task} if paste_task else set())
        done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        for finished in done:
            finished.result()  # a bad paste or a refused login raises here
        code, client_id = callback.result()
    finally:
        if paste_task is not None:
            paste_task.cancel()
        server.close()

    fields = await asyncio.to_thread(_request_token, {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": code,
        "code_verifier": verifier,
        "redirect_uri": REDIRECT_URI,
        "resource": _RESOURCE,
    })
    await asyncio.to_thread(_store_login, profile, {"host_id": host_id, "client_id": client_id, **fields})

