"""Retry policy for model requests.

A turn is a long-lived thing to lose to a rate limit or a dropped connection,
so transient failures are retried with exponential backoff instead of ending
the turn. A streamed request is only retried *before* it produces its first
event: once deltas have reached the UI and the accumulated text, restarting
would duplicate them.
"""

import asyncio
import random

import httpx
from pydantic_ai.exceptions import ModelHTTPError

MAX_ATTEMPTS = 4  # the initial request plus three retries

_BASE_DELAY = 1.0
_MAX_DELAY = 16.0

# 408 request timeout, 409 conflict, 429 rate limit; 5xx is covered by the
# range check. Anything else (401, 404, 400) is a request that will not get
# better by being sent again.
_TRANSIENT_STATUS = {408, 409, 429}


def is_transient(exc: BaseException) -> bool:
    """Whether resending the identical request could plausibly succeed."""
    if isinstance(exc, ModelHTTPError):
        return exc.status_code in _TRANSIENT_STATUS or exc.status_code >= 500
    # Connection resets, DNS failures and read timeouts: the request never
    # reached the model, or its answer never made it back.
    return isinstance(exc, (httpx.TransportError, ConnectionError, asyncio.TimeoutError))


# Provider phrasings that mean the request exceeded the model's context
# window. Deliberately broad: a miss only costs the compact-and-retry chance,
# while a false positive costs one extra compaction attempt.
_OVERFLOW_HINTS = (
    "context length",
    "context window",
    "maximum context",
    "context_length_exceeded",
    "prompt is too long",
    "input is too long",
    "too many tokens",
    "token limit",
    "tokens exceed",
    "exceeds the maximum",
    "request too large",
)


def is_context_overflow(exc: BaseException) -> bool:
    """Whether the provider rejected the request for exceeding its context
    window — the one 400 that can get better, after compaction."""
    if not isinstance(exc, ModelHTTPError) or exc.status_code not in (400, 413):
        return False
    text = str(exc).lower()
    return any(hint in text for hint in _OVERFLOW_HINTS)


def backoff(attempt: int) -> float:
    """Seconds to wait before ``attempt`` (1 is the first retry).

    Jittered over the upper half of the exponential delay. Concurrent agents
    share one API key, so a rate limit hits all of them at the same instant;
    a deterministic delay would have them retry in lockstep and exhaust every
    attempt on the same collision.
    """
    delay = min(_BASE_DELAY * 2 ** (attempt - 1), _MAX_DELAY)
    return delay / 2 + random.random() * delay / 2


def describe(exc: BaseException) -> str:
    if isinstance(exc, ModelHTTPError):
        return f"HTTP {exc.status_code}"
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
