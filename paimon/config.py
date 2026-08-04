"""Model settings loaded from a JSON config file.

$PAIMON_CONFIG_HOME (default ~/.config/paimon/) holds one directory per
profile, each fully independent — model, key, theme, everything. The profile
named "default" is used unless another one is activated. The stored
model/api_base/api_key are turned into a pydantic-ai model by paimon.llm;
provider environment variables are the fallback when unset.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_PROFILE = "default"

# Default for save() arguments, so passing None can mean "clear the stored
# value" (re-logging in with a blank api_base must drop the old override).
UNSET: object = object()

_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]*"


def config_root() -> Path:
    """The directory holding all profiles — not itself a profile."""
    override = os.environ.get("PAIMON_CONFIG_HOME")
    return Path(override) if override else Path.home() / ".config" / "paimon"


def validate_profile(name: Optional[str]) -> str:
    """Normalize a --profile value: None means the default profile.

    Raises ValueError on a name that could escape the config root.
    """
    name = name or DEFAULT_PROFILE
    if not re.fullmatch(_NAME_PATTERN, name):
        raise ValueError(f"invalid profile name {name!r}")
    return name


def list_profiles() -> list[str]:
    """Profile names present on disk, always including the default."""
    names = {DEFAULT_PROFILE}
    root = config_root()
    if root.is_dir():
        names.update(entry.name for entry in root.iterdir()
                     if entry.is_dir() and re.fullmatch(_NAME_PATTERN, entry.name))
    return sorted(names)


def config_dir(profile: str = DEFAULT_PROFILE) -> Path:
    return config_root() / profile


def config_path(profile: str = DEFAULT_PROFILE) -> Path:
    return config_dir(profile) / "config.json"


def _load_file_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class Config:
    """Settings owned by whoever constructed them — no module-level state.

    ``profile`` names the config directory this instance was loaded from and
    is where ``save()`` writes back, so concurrent Config instances bound to
    different profiles never interfere.
    """

    profile: str = DEFAULT_PROFILE
    model: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    theme: Optional[str] = None
    # Display-only: reasoning is still generated, persisted and sent back.
    show_reasoning: bool = False
    # Auto-allow clearly read-only shell commands (ls, git status, ...) in
    # read/edit modes. A guardrail toggle, not a security boundary.
    safe_commands: bool = True
    compaction_enabled: bool = True
    compaction_reserve_tokens: int = 16_384
    compaction_keep_recent_tokens: int = 20_000
    # Overrides the built-in window table, for model names it does not know.
    compaction_context_window: Optional[int] = None

    @classmethod
    def load(cls, profile: Optional[str] = None) -> "Config":
        """Load the named profile's settings (None means the default profile).

        Raises ValueError on an invalid profile name.
        """
        profile = validate_profile(profile)
        data = _load_file_config(config_path(profile))
        compaction = data.get("compaction") if isinstance(data.get("compaction"), dict) else {}
        return cls(
            profile=profile,
            model=data.get("model"),
            api_base=data.get("api_base"),
            api_key=data.get("api_key"),
            theme=data.get("theme"),
            show_reasoning=data.get("show_reasoning", cls.show_reasoning),
            safe_commands=data.get("safe_commands", cls.safe_commands),
            compaction_enabled=compaction.get("enabled", cls.compaction_enabled),
            compaction_reserve_tokens=compaction.get("reserve_tokens", cls.compaction_reserve_tokens),
            compaction_keep_recent_tokens=compaction.get("keep_recent_tokens", cls.compaction_keep_recent_tokens),
            compaction_context_window=compaction.get("context_window"),
        )

    def save(
        self,
        model: object = UNSET,
        api_base: object = UNSET,
        api_key: object = UNSET,
        theme: object = UNSET,
        show_reasoning: object = UNSET,
    ) -> None:
        """Persist the fields passed to config.json and update self.

        Passing None (or an empty string) removes the stored value; fields not
        passed and other keys already in the file are preserved.
        """
        path = config_path(self.profile)
        data = _load_file_config(path)
        for key, value in (("model", model), ("api_base", api_base),
                           ("api_key", api_key), ("theme", theme),
                           ("show_reasoning", show_reasoning)):
            if value is UNSET:
                continue
            if value is None or value == "":
                data.pop(key, None)
            else:
                data[key] = value

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        # The file may hold an API key: create it private, repair old copies.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        os.chmod(path, 0o600)

        self.model = data.get("model")
        self.api_base = data.get("api_base")
        self.api_key = data.get("api_key")
        self.theme = data.get("theme")
        self.show_reasoning = data.get("show_reasoning", type(self).show_reasoning)
