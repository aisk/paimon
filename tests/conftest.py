"""Suite-wide guards."""

import os
from unittest.mock import patch

import pytest

# The suite exercises cli.main() freely; without this every such test would
# send a real event to Google Analytics and write telemetry state into the
# developer's home directory. Telemetry tests opt back in explicitly.
os.environ["PAIMON_NO_TELEMETRY"] = "1"


@pytest.fixture(autouse=True)
def _no_default_skill_dirs():
    """Keep the developer's own ~/.agents/skills and config skills out of the
    suite. Tests of the default locations patch default_skill_dirs themselves."""
    with patch("paimon.skills.default_skill_dirs", return_value=[]):
        yield
