"""Suite-wide guards."""

import os

# The suite exercises cli.main() freely; without this every such test would
# send a real event to Google Analytics and write telemetry state into the
# developer's home directory. Telemetry tests opt back in explicitly.
os.environ["PAIMON_NO_TELEMETRY"] = "1"
