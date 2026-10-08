"""What the browser logins share: opening the browser, catching the redirect
on a loopback port, and keeping the stored tokens fresh.

A login lives in the profile's config.json, under "providers": {NAME: ...},
and is rewritten whenever its tokens change. Each provider describes itself
with a Login and brings its own endpoints and token requests.
"""

import asyncio
import base64
import hashlib
import secrets
import threading
import time
import webbrowser
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Iterator, Optional, TypeVar
from urllib.parse import parse_qs, urlsplit

from . import lockfile
from .config import DEFAULT_PROFILE, ConfigError, config_dir, read_provider, update_provider
from .errors import PaimonError

T = TypeVar("T")

_REFRESH_LOCK_TIMEOUT = 90.0
_STORE_ATTEMPTS = 3
_STORE_RETRY_SECONDS = 1.0

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


def pkce() -> tuple[str, str]:
    """A fresh (verifier, S256 challenge) pair."""
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


@dataclass(frozen=True)
class Login:
    """How one provider's browser login is stored and renewed."""

    provider: str  # the key under "providers" in config.json
    label: str  # the name the user knows the account by
    keys: tuple  # the fields a complete login holds
    refresh: Callable[[dict], dict]  # the stored entry -> the fields a refresh replaces
    error: type  # the PaimonError raised for a login that is missing or refused
    # The sidecar lock keeps processes apart but is reentrant within one, and
    # every Agent builds its own model, so threads need their own exclusion.
    # Two refreshes racing on one refresh token leave the loser's login revoked.
    _mutex: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def read(self, profile: str) -> dict:
        return read_provider(profile, self.provider)

    def complete(self, stored: dict) -> bool:
        return all(isinstance(stored.get(key), str) and stored[key] for key in self.keys)

    def signed_in(self, profile: str) -> bool:
        """Whether the profile holds a login to build a model from."""
        return self.complete(self.read(profile))

    @contextmanager
    def _lock(self, profile: str) -> Iterator[None]:
        """Held by whoever replaces the stored tokens, a refresh or a login.

        Not the config's own lock: this one is held across a network call, and
        an unrelated save (a theme change) must not wait that out.
        """
        lock = config_dir(profile) / f"{self.provider}.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with self._mutex:
            if not lockfile.acquire(lock, _REFRESH_LOCK_TIMEOUT):
                raise self.error(f"another Paimon held {lock} for over {_REFRESH_LOCK_TIMEOUT:.0f}s")
            try:
                yield
            finally:
                lockfile.release(lock)

    def _store_rotation(self, profile: str, fields: dict) -> dict:
        """Write freshly rotated tokens, trying again if the config cannot be
        written. The old refresh token may already be dead, so a write given
        up on is a login lost."""
        for attempt in range(_STORE_ATTEMPTS):
            try:
                return update_provider(profile, self.provider, fields)
            except ConfigError:
                if attempt == _STORE_ATTEMPTS - 1:
                    raise
                time.sleep(_STORE_RETRY_SECONDS)
        raise AssertionError("unreachable")

    def renewed(self, profile: str) -> dict:
        """The stored login, refreshed first if it has expired.

        Synchronous and run on a thread: a turn cancelled mid-refresh must
        still get the rotated refresh token onto disk, or the login is lost.
        """
        with self._lock(profile):
            # Re-read inside the lock: another process may have refreshed.
            stored = self.read(profile)
            if not self.complete(stored):
                raise self.error(f"not signed in to {self.label}, log in again")
            if expired(stored):
                stored = self._store_rotation(profile, self.refresh(stored))
            return stored

    def store(self, profile: str, credential: dict) -> None:
        # Under the lock, so a refresh still in flight for the previous login
        # lands first and cannot mix its tokens into this one's entry.
        with self._lock(profile):
            update_provider(profile, self.provider, credential)

    def callback_query(self, target: str, redirect_uri: str, state: str) -> dict:
        """The query of a callback URL or request target, once it is known to
        belong to this login attempt and to carry no refusal."""
        parts = urlsplit(target.strip())
        if parts.path != urlsplit(redirect_uri).path:
            raise self.error(f"the callback URL must start with {redirect_uri}")
        query = {key: values[0] for key, values in parse_qs(parts.query).items()}
        if query.get("error"):
            raise self.error(f"{self.label} authorization failed: {query['error']}")
        if query.get("state") != state:
            raise self.error("the callback belongs to a different login attempt")
        return query


def expired(stored: dict) -> bool:
    expires = stored.get("expires")
    return not isinstance(expires, (int, float)) or time.time() >= expires


class Credentials:
    """The bearer token for one profile, as the callable an API client asks
    before every request."""

    def __init__(self, login: Login, profile: Optional[str] = None) -> None:
        self.login = login
        self.profile = profile or DEFAULT_PROFILE
        self._stored = login.read(self.profile)
        if not login.complete(self._stored):
            raise login.error(
                f"not signed in to {login.label}, run 'paimon login --model {login.provider}:MODEL'")

    async def current(self) -> dict:
        """The stored login with a live access token."""
        if expired(self._stored):
            self._stored = await asyncio.to_thread(self.login.renewed, self.profile)
        return self._stored

    async def __call__(self) -> str:
        return (await self.current())["access"]


async def redirected(host: str, port: int, path: str, parse: Callable[[str], T], show: Callable[[], None],
                     pasted: Optional[Awaitable[str]] = None, *, done: str, busy: PaimonError) -> T:
    """Wait on a loopback port for the browser to come back from a sign-in.

    ``show`` is called once the port is listening. ``parse`` turns a request
    target under ``path`` into the result, and raises a PaimonError for one
    that is not this login's. Where the browser cannot reach this machine,
    ``pasted`` may deliver the redirect URL instead. ``busy`` is raised when
    the port is taken.
    """
    callback: asyncio.Future = asyncio.get_running_loop().create_future()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = (await reader.readline()).decode("latin-1").split()
            target = request_line[1] if len(request_line) > 1 else ""
            if urlsplit(target).path != path:
                status, body = "404 Not Found", "Not found."
            else:
                try:
                    result = parse(target)
                except PaimonError as exc:
                    status, body = "400 Bad Request", str(exc)
                    # A stale tab from an earlier attempt must not end this one.
                    if "error=" in target and not callback.done():
                        callback.set_exception(exc)
                else:
                    status, body = "200 OK", done
                    if not callback.done():
                        callback.set_result(result)
            payload = body.encode()
            writer.write(f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
                         f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode() + payload)
            await writer.drain()
        finally:
            writer.close()

    try:
        server = await asyncio.start_server(serve, host, port)
    except OSError as exc:
        raise busy from exc

    async def from_paste() -> None:
        result = parse(await pasted)  # type: ignore[misc]
        if not callback.done():
            callback.set_result(result)

    paste_task = asyncio.ensure_future(from_paste()) if pasted is not None else None
    try:
        show()
        waiting = {callback} | ({paste_task} if paste_task else set())
        finished, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        for task in finished:
            task.result()  # a bad paste or a refused login raises here
        return callback.result()
    finally:
        if paste_task is not None:
            paste_task.cancel()
        server.close()
