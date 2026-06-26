"""Configuration: model settings read from environment variables.
"""

import os

# litellm uses the "openai/" prefix to talk to any OpenAI-compatible endpoint.
MODEL = os.environ.get("PAIMON_MODEL")
API_BASE = os.environ.get("PAIMON_API_BASE")
API_KEY = os.environ.get("PAIMON_API_KEY")
