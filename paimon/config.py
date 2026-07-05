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


def save(
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    """Persist fields to config.json and refresh module-level constants.

    Only the fields passed (and not None) are written; others are preserved.
    """
    cfg = _load_file_config()
    if model is not None:
        cfg["model"] = model
    if api_base is not None:
        cfg["api_base"] = api_base
    if api_key is not None:
        cfg["api_key"] = api_key

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    global MODEL, API_BASE, API_KEY
    MODEL = cfg.get("model")
    API_BASE = cfg.get("api_base")
    API_KEY = cfg.get("api_key")
