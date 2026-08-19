"""Anonymous usage statistics, sent to Google Analytics.

One event per invocation, delivered to GA4's gtag collection endpoint from a
background thread so launch never waits on the network. The only identity is
a random UUID minted on first use and kept in telemetry.json next to the
profiles (shared by every profile, so one person counts once). No prompt
text, paths, credentials or anything else from a session ever leaves the
machine. Setting DO_NOT_TRACK or PAIMON_NO_TELEMETRY disables it entirely,
including the state file.
"""

import json
import os
import platform
import threading
import time
import uuid
from typing import Optional

from .config import config_root

_MEASUREMENT_ID = "G-Y22LTJ7MKT"
_ENDPOINT = "https://www.google-analytics.com/g/collect"
_TIMEOUT = 3.0


def enabled() -> bool:
    for var in ("PAIMON_NO_TELEMETRY", "DO_NOT_TRACK"):
        value = os.environ.get(var, "")
        if value and value != "0":
            return False
    return True


def _state_path():
    return config_root() / "telemetry.json"


def _load_state() -> dict:
    """The stored client id and launch counter, or {} when absent or damaged.

    Unlike config.json, a broken file here is not worth surfacing: the worst
    outcome of regenerating is one user counted twice.
    """
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("client_id"), str):
            return data
    except (OSError, ValueError):
        pass
    return {}


def _save_state(state: dict) -> None:
    path = _state_path()
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("paimon")
    except Exception:
        return "unknown"


def _language() -> Optional[str]:
    lang = os.environ.get("LC_ALL") or os.environ.get("LANG") or ""
    lang = lang.split(".")[0].replace("_", "-").lower()
    return lang if lang and lang != "c" else None


def _user_agent() -> str:
    # GA derives the OS dimension from a browser-shaped User-Agent and may
    # drop hits from unrecognized clients as bots, so send a minimal one with
    # a real platform token rather than the HTTP library's default.
    token = {
        "Linux": "X11; Linux x86_64",
        "Darwin": "Macintosh; Intel Mac OS X 10_15_7",
        "Windows": "Windows NT 10.0; Win64; x64",
    }.get(platform.system(), platform.system() or "Unknown")
    return f"Mozilla/5.0 ({token}) paimon/{_version()}"


def _prepare(mode: str, model: Optional[str] = None) -> Optional[dict]:
    """Advance the on-disk state and build one event's query parameters.

    Returns None when telemetry is off. Each invocation is its own GA
    session: sid is the launch time, sct the lifetime launch count, and _fv
    marks the very first launch so GA counts a new user.
    """
    if not enabled():
        return None
    state = _load_state()
    first_visit = "client_id" not in state
    if first_visit:
        state["client_id"] = str(uuid.uuid4())
    state["session_count"] = int(state.get("session_count", 0) or 0) + 1
    _save_state(state)

    params = {
        "v": "2",
        "tid": _MEASUREMENT_ID,
        "cid": state["client_id"],
        "sid": str(int(time.time())),
        "sct": str(state["session_count"]),
        "seg": "1",
        "_s": "1",
        "_ss": "1",
        "en": "app_start",
        "ep.mode": mode,
        "ep.app_version": _version(),
        "ep.os": platform.system() or "unknown",
    }
    if first_visit:
        params["_fv"] = "1"
    if model:
        # The provider goes up as a user property (36 char cap) so users can
        # be segmented by it; the full qualified model string stays an event
        # parameter, whose 100 char cap fits long model names.
        params["ep.model"] = model[:100]
        provider = model.partition(":")[0]
        if provider != model:
            params["up.provider"] = provider[:36]
    language = _language()
    if language:
        params["ul"] = language
    return params


def _send(params: dict) -> None:
    # Imported here: httpx is not needed on the fast path of subcommands
    # that never send (telemetry disabled).
    try:
        import httpx

        httpx.post(_ENDPOINT, params=params,
                   headers={"User-Agent": _user_agent()}, timeout=_TIMEOUT)
    except Exception:
        pass


def record_launch(mode: str, model: Optional[str] = None) -> None:
    """Count this invocation under the given mode (tui, headless, web, or a
    subcommand name), optionally tagged with the qualified model in use.
    Never raises and never blocks the caller."""
    try:
        params = _prepare(mode, model)
    except Exception:
        return
    if params:
        threading.Thread(target=_send, args=(params,), daemon=True,
                         name="paimon-telemetry").start()
