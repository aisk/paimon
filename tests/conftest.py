"""Suite-wide guards."""

from unittest.mock import patch

import pytest
from pydantic_ai.models import override_allow_model_requests


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch, tmp_path):
    """CLI tests must not read or write the developer's config and sessions.

    Tests can override these defaults in setUp or a local patch. In particular,
    telemetry tests opt back in while replacing the sender.
    """
    monkeypatch.setenv("PAIMON_NO_TELEMETRY", "1")
    monkeypatch.setenv("PAIMON_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("PAIMON_DATA_HOME", str(tmp_path / "data"))


@pytest.fixture(autouse=True)
def _no_model_requests():
    """FunctionModel stubs work; accidentally using a real provider fails fast."""
    with override_allow_model_requests(False):
        yield


@pytest.fixture(autouse=True)
def _no_default_skill_dirs():
    """Keep the developer's own ~/.agents/skills and config skills out of the
    suite. Tests of the default locations patch default_skill_dirs themselves."""
    with patch("paimon.skills.default_skill_dirs", return_value=[]):
        yield


@pytest.fixture(autouse=True)
def _no_default_agent_dirs():
    """The same guard for ~/.agents/agents and config agent types. Built-in
    types stay: they are part of the behavior under test."""
    with patch("paimon.agents.default_agent_dirs", return_value=[]):
        yield
