"""Model settings loaded from a JSON config file.

File location: $PAIMON_CONFIG_HOME/config.json (default ~/.config/paimon/).
API key resolution for the actual request is delegated to litellm via
agent.py; paimon only stores what the user entered at login.
"""

import json
import os
from pathlib import Path
from typing import Optional


def config_dir() -> Path:
    override = os.environ.get("PAIMON_CONFIG_HOME")
    return Path(override) if override else Path.home() / ".config" / "paimon"


def config_path() -> Path:
    return config_dir() / "config.json"


def _load_file_config() -> dict:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


_cfg = _load_file_config()
MODEL: Optional[str] = _cfg.get("model")
API_BASE: Optional[str] = _cfg.get("api_base")
API_KEY: Optional[str] = _cfg.get("api_key")
