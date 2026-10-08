"""Model settings loaded from a JSON config file.

$PAIMON_CONFIG_HOME (default ~/.config/paimon/) holds config.json: model,
keys, theme, everything. Credentials are scoped per provider ("providers":
{"zai": {"api_base": ..., "api_key": ...}}), so it holds an account per
provider and switching models never borrows another provider's key. A
provider that signs in through the browser keeps its tokens in the same map
(see paimon.chatgpt). The stored model plus its provider's credentials are
turned into a pydantic-ai model by paimon.llm; provider environment variables
are the fallback when unset.
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import lockfile
from .errors import PaimonError

# Default for save() arguments, so passing None can mean "clear the stored
# value" (re-logging in with a blank api_base must drop the old override).
UNSET: object = object()

# How long save() waits for another Paimon's write to finish before giving up
# with a diagnostic. The lock is only held for the few milliseconds of one
# read-merge-replace, so hitting this means something is genuinely stuck.
_SAVE_LOCK_TIMEOUT = 10.0

# The sidecar lock keeps processes apart but is reentrant within one process
# (lockfile refcounts per path), so threads of the same process need their own
# mutual exclusion. The TUI saves from worker threads.
_SAVE_MUTEX = threading.Lock()


class ConfigError(PaimonError):
    """The stored config cannot be read or written (corrupt JSON, a lock
    that never frees, a write that cannot reach the disk)."""


def _provider_of(model: str) -> str:
    """The provider half of a qualified model string.

    Imported lazily: pulling paimon.llm (and with it pydantic-ai) at module
    import would slow down every subcommand that only reads the config.
    """
    from .llm import split_model_string

    return split_model_string(model)[0]


def _parse_providers(raw: object) -> dict:
    """The stored per-provider credential map, dropping malformed entries."""
    providers: dict = {}
    if isinstance(raw, dict):
        for name, entry in raw.items():
            if isinstance(entry, dict):
                fields = {key: value for key, value in entry.items()
                          if key in ("api_base", "api_key") and isinstance(value, str)}
                if fields:
                    providers[name] = fields
    return providers


def config_root() -> Path:
    override = os.environ.get("PAIMON_CONFIG_HOME")
    return Path(override) if override else Path.home() / ".config" / "paimon"


def config_path() -> Path:
    return config_root() / "config.json"


def _adopt_default_profile() -> None:
    """Move the config up from where it lived while there were profiles, each
    in a directory of its own with "default" the one in use."""
    path = config_path()
    old = config_root() / "default" / "config.json"
    if not path.exists() and old.exists():
        try:
            old.replace(path)
        except OSError:
            pass


def _lock_path(path: Path) -> Path:
    """The sidecar lock for the config.

    Never lock config.json itself: the atomic replace swaps the file's inode,
    which would strand the lock on the file readers just replaced.
    """
    return path.with_name(path.name + ".lock")


def _read_file_config(path: Path) -> dict:
    """The stored JSON object, or {} when no file exists yet.

    Raises ConfigError on a damaged file. Swallowing the parse error here is
    how one torn write used to become permanent: the next save would happily
    write "{} plus my fields" over the remains of every other setting.
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{path} is not valid JSON ({exc.msg} at line {exc.lineno}). "
            "The file was left untouched, fix or remove it") from exc
    except FileNotFoundError:
        # Deleted between is_file() and the read. Same as never existing.
        return {}
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} holds {type(data).__name__}, not a JSON object. "
                          "The file was left untouched, fix or remove it")
    return data


def _sync_directory(directory: Path) -> None:
    """Make a completed rename durable; a no-op where directories cannot be
    opened (Windows) or fsynced (some network filesystems)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _replace(tmp: Path, path: Path) -> None:
    """os.replace, waiting out Windows readers holding the destination open.

    The rename is atomic on POSIX regardless of readers. On Windows it fails
    with PermissionError while another process has config.json open, and
    load() reads without locking, so briefly retry there.
    """
    attempts = 10 if os.name == "nt" else 1
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05)


def write_atomic(path: Path, payload: str) -> None:
    """Replace path's contents in one step. Call only under the config lock.

    The new bytes go to a sibling temp file that is fsynced and renamed over
    path, so a reader (load() never locks) always sees either the complete
    old config or the complete new one, and a crash mid-write leaves the
    previous config intact instead of a torn one. The temp file name is
    fixed because writers are serialized by the lock, so an orphan left by
    a killed writer is simply overwritten by the next save.
    """
    # Write through a symlinked config.json (dotfiles setups) instead of
    # replacing the link itself with a detached plain file.
    path = Path(os.path.realpath(path))
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            data = memoryview(payload.encode("utf-8"))
            while data:  # os.write may write fewer bytes than asked for
                data = data[os.write(fd, data):]
            os.fsync(fd)
        finally:
            os.close(fd)
        # The file may hold an API key. The open mode above is masked by the
        # umask, so make it private explicitly, before it becomes the config.
        os.chmod(tmp, 0o600)
        _replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ConfigError(f"cannot write {path}: {exc}") from exc
    _sync_directory(path.parent)


def _update_file(path: Path, change: Callable[[dict], None]) -> dict:
    """Apply change to the stored object and write it back. Returns what was
    written.

    Writers serialize on a sidecar lock, re-read inside it and replace the
    file atomically, so two Paimons running at once cannot lose each
    other's fields and a reader never sees a half-written config.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock_path(path)
    with _SAVE_MUTEX:
        if not lockfile.acquire(lock, _SAVE_LOCK_TIMEOUT):
            raise ConfigError(
                f"another Paimon held {lock} for over {_SAVE_LOCK_TIMEOUT:.0f}s, "
                "nothing was written")
        try:
            data = _read_file_config(path)
            change(data)
            write_atomic(path, json.dumps(data, indent=2, ensure_ascii=False))
            return data
        finally:
            lockfile.release(lock)


def _merge_provider(data: dict, provider: str, fields: dict) -> None:
    """Merge fields into one provider's stored entry; None or "" removes a
    field, and an entry or map left empty is dropped."""
    providers = data.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    entry = providers.get(provider)
    if not isinstance(entry, dict):
        entry = {}
    for key, value in fields.items():
        if value is None or value == "":
            entry.pop(key, None)
        else:
            entry[key] = value
    if entry:
        providers[provider] = entry
    else:
        providers.pop(provider, None)
    if providers:
        data["providers"] = providers
    else:
        data.pop("providers", None)


def read_provider(provider: str) -> dict:
    """One provider's entry exactly as stored, {} when there is none.

    Config.providers only carries api_base and api_key. This is for the
    providers that keep more than that, and need it fresh from the file.
    """
    providers = _read_file_config(config_path()).get("providers")
    entry = providers.get(provider) if isinstance(providers, dict) else None
    return dict(entry) if isinstance(entry, dict) else {}


def update_provider(provider: str, fields: dict) -> dict:
    """Merge fields into one provider's stored entry and return the entry."""
    data = _update_file(config_path(), lambda data: _merge_provider(data, provider, fields))
    return dict(data.get("providers", {}).get(provider, {}))


@dataclass
class Config:
    """Settings owned by whoever constructed them — no module-level state."""

    model: Optional[str] = None
    # Per-provider credentials: {"zai": {"api_base": ..., "api_key": ...}}.
    # Keyed by provider so switching models never sends one provider's key
    # to another provider's endpoint.
    providers: dict = field(default_factory=dict)
    theme: Optional[str] = None
    # Stream reasoning expanded in the TUI (it folds once the block ends) and
    # print it in headless mode. When off the TUI folds it behind a line-count
    # stub instead; either way it is still generated, persisted and sent back.
    show_reasoning: bool = False
    # Auto-allow clearly read-only shell commands (ls, git status, ...) in
    # read/auto modes. A guardrail toggle, not a security boundary.
    safe_commands: bool = True
    # The model auto mode reviews held tool calls with, as "provider:name".
    # None takes the reviewer paired with the working model in
    # review.DEFAULT_REVIEWERS, or the working model itself. Edited by hand.
    review_model: Optional[str] = None
    # Offer a short recap once a turn that did some work is followed by this
    # many idle seconds. Seconds rather than a count so a test can turn the
    # wait down; the TUI never writes these back, they are edited by hand.
    recap_enabled: bool = True
    recap_idle_seconds: float = 30.0
    compaction_enabled: bool = True
    compaction_reserve_tokens: int = 16_384
    compaction_keep_recent_tokens: int = 20_000
    # Stands in for the built-in window table on model names it does not know.
    compaction_context_window: Optional[int] = None
    # Extra skill files or directories, on top of the default locations.
    # Edited by hand (or extended by --skill for one run); never written back.
    skills: list[str] = field(default_factory=list)
    # Cleared by --no-skills for one run; not a config file setting.
    include_default_skills: bool = True
    # Cleared by --no-web-search for one run; not a config file setting.
    web_search: bool = True

    @classmethod
    def load(cls) -> "Config":
        _adopt_default_profile()
        data = _read_file_config(config_path())
        compaction = data.get("compaction") if isinstance(data.get("compaction"), dict) else {}
        skills = data.get("skills")
        return cls(
            model=data.get("model"),
            providers=_parse_providers(data.get("providers")),
            theme=data.get("theme"),
            show_reasoning=data.get("show_reasoning", cls.show_reasoning),
            safe_commands=data.get("safe_commands", cls.safe_commands),
            review_model=data.get("review_model"),
            recap_enabled=data.get("recap_enabled", cls.recap_enabled),
            recap_idle_seconds=data.get("recap_idle_seconds", cls.recap_idle_seconds),
            compaction_enabled=compaction.get("enabled", cls.compaction_enabled),
            compaction_reserve_tokens=compaction.get("reserve_tokens", cls.compaction_reserve_tokens),
            compaction_keep_recent_tokens=compaction.get("keep_recent_tokens", cls.compaction_keep_recent_tokens),
            compaction_context_window=compaction.get("context_window"),
            skills=[str(p) for p in skills] if isinstance(skills, list) else [],
        )

    def provider_auth(self, model: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        """The stored (api_base, api_key) for the model's provider.

        Defaults to the configured model. A provider without a stored entry
        (or no model at all) yields (None, None), which paimon.llm turns into
        the provider's own environment-variable resolution.
        """
        model = model or self.model
        if not model:
            return None, None
        try:
            provider = _provider_of(model)
        except ValueError:
            return None, None
        entry = self.providers.get(provider) or {}
        return entry.get("api_base"), entry.get("api_key")

    def save(
        self,
        model: object = UNSET,
        api_base: object = UNSET,
        api_key: object = UNSET,
        theme: object = UNSET,
        show_reasoning: object = UNSET,
        recap_enabled: object = UNSET,
        provider: Optional[str] = None,
    ) -> None:
        """Persist the fields passed to config.json and update self.

        Passing None (or an empty string) removes the stored value; fields not
        passed and other keys already in the file are preserved. api_base and
        api_key are stored under provider, so every provider keeps its own
        credentials. Left out, it is the provider of the model being saved
        (the model argument, else the configured model).
        """
        auth_passed = [(key, value) for key, value in (
            ("api_base", api_base), ("api_key", api_key),
        ) if value is not UNSET]
        if auth_passed and provider is None:
            target = model if model is not UNSET and model else self.model
            if not target:
                raise ConfigError("credentials need a model to scope them to a provider; "
                                  "pass a model along with the api base/key")
            try:
                provider = _provider_of(target)
            except ValueError as exc:
                raise ConfigError(str(exc)) from exc

        passed = [(key, value) for key, value in (
            ("model", model),
            ("theme", theme),
            ("show_reasoning", show_reasoning),
            ("recap_enabled", recap_enabled),
        ) if value is not UNSET]

        def change(data: dict) -> None:
            for key, value in passed:
                if value is None or value == "":
                    data.pop(key, None)
                else:
                    data[key] = value
            if auth_passed:
                _merge_provider(data, provider, dict(auth_passed))

        data = _update_file(config_path(), change)

        # Only the fields this call wrote are refreshed from the file. A
        # runtime override the caller set on the instance (paimon --model X
        # assigns self.model) must survive an unrelated save, such as the TUI
        # persisting a theme change.
        for key, _ in passed:
            setattr(self, key, data.get(key, getattr(type(self), key)))
        if auth_passed:
            self.providers = _parse_providers(data.get("providers"))
