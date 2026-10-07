"""Build a pydantic-ai Model from the configured model string.

The config keeps litellm-style strings ("zai/glm-5.2") as well as pydantic-ai
style ones ("zai:glm-5.2"). The provider prefix picks the wire dialect
(OpenAI chat completions, OpenAI responses, Anthropic messages, ...) while
api_base picks the endpoint; the two are independent. The provider is
constructed explicitly so the configured api_key/api_base take precedence over
environment variables.

Only the dialects whose SDK is installed can be built — pydantic-ai imports
those lazily and raises ImportError for the rest.
"""

import inspect
import platform
import typing
from functools import cache
from importlib import metadata
from typing import Optional, Sequence

from pydantic_ai.direct import model_request
from pydantic_ai.messages import ModelMessage, TextPart
from pydantic_ai.models import KnownModelName, Model, ModelRequestParameters, infer_model
from pydantic_ai.providers import infer_provider_class

from .errors import PaimonError

# The pseudo-provider served by paimon.chatgpt, which is imported on demand
# like the provider SDKs are.
CHATGPT_PROVIDER = "chatgpt"


class NoModelError(PaimonError):
    """No model is configured, so nothing can be asked of one."""


@cache
def user_agent() -> str:
    """The User-Agent sent with LLM requests, replacing pydantic-ai's default."""
    try:
        version = metadata.version("paimon")
    except metadata.PackageNotFoundError:
        version = "dev"
    return f"paimon/{version} (Python {platform.python_version()}; {platform.system()} {platform.machine()})"


def provider_class(provider_name: str):
    """Look up a provider class, turning a missing SDK into a readable error."""
    try:
        return infer_provider_class(provider_name)
    except ImportError as exc:
        raise ValueError(
            f"Provider {provider_name!r} needs a dependency Paimon does not ship: {exc}"
        ) from exc


def is_provider_available(provider_name: str) -> bool:
    """Whether this provider can be constructed with the installed dependencies."""
    if provider_name == CHATGPT_PROVIDER:
        return True
    try:
        infer_provider_class(provider_name)
    except (ImportError, ValueError):
        return False
    return True


def split_model_string(model: str) -> tuple[str, str]:
    """Split "provider:name" (or legacy "provider/name") into its two halves."""
    if ":" in model:
        provider, _, name = model.partition(":")
    else:
        provider, _, name = model.partition("/")
    if not provider or not name:
        raise ValueError(
            f"Model {model!r} must be qualified as 'provider:model' (e.g. 'zai:glm-5.2')"
        )
    return provider, name


# The models worth offering, one line of succession per row, newest first. A
# provider shows the newest member of each row its catalog has, so a model
# whose successor has not reached that provider (or this pydantic-ai) stays
# listed. Resellers name the same models their own way and get their own rows.
# Left out on purpose: dated snapshots, superseded generations, and models
# that are not general coding models (cyber, audio, deep research).
_CURRENT = (
    ("claude-fable-5-1", "claude-fable-5"),
    ("claude-opus-5-5", "claude-opus-5", "claude-opus-4-8", "claude-opus-4-6"),
    ("claude-sonnet-5-5", "claude-sonnet-5", "claude-sonnet-4-6", "claude-4-6-sonnet"),
    ("claude-haiku-4-5", "claude-4-5-haiku"),
    ("gpt-6.1-sol", "gpt-6-sol", "gpt-5.6-sol"),
    ("gpt-6-astra",),
    ("gpt-5.6-terra",),
    ("gpt-6-luna", "gpt-5.6-luna"),
    ("openai.gpt-5.6-sol",),
    ("openai.gpt-5.6-terra",),
    ("openai.gpt-5.6-luna",),
    ("openai-gpt-5-6-sol",),
    ("openai-gpt-5-6-terra",),
    ("openai-gpt-5-6-luna",),
    ("glm-5.3", "glm-5.2", "zai/GLM-5.2"),
    ("glm-5.3-flashx",),
    ("glm-5.3-flash",),
    ("glm-5-turbo",),
    ("glm-5v-turbo",),
    ("kimi-k3", "moonshotai/Kimi-K2.6", "kimi-k2-5"),
    ("kimi-k2.7-code",),
    ("kimi-k2.7-code-highspeed",),
    ("deepseek-v4-pro", "deepseek-ai/DeepSeek-V4-Pro", "deepseek-v3-2"),
    ("deepseek-flash", "deepseek-v4-flash", "deepseek-ai/Deepseek-V4-Flash"),
    ("qwen-3.8-27b", "qwen3-coder-480b"),
    ("minimax-m2-1",),
    ("gemma-4-31b", "google/gemma-4-31b-it"),
    ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",),
    ("gpt-oss-120b", "openai.gpt-oss-120b", "openai/gpt-oss-120b"),
)


def known_models() -> list[str]:
    return sorted(typing.get_args(KnownModelName.__value__))


def models_for(provider: str) -> list[str]:
    """The current models of one provider, newest generation first."""
    if provider == CHATGPT_PROVIDER:
        # A ChatGPT plan serves OpenAI's own models; which ones depends on the plan.
        provider = "openai"
    prefix = provider + ":"
    catalog = [name.removeprefix(prefix) for name in known_models() if name.startswith(prefix)]
    current = [next((name for name in line if name in catalog), None) for line in _CURRENT]
    # A provider none of the rows know is shown whole rather than empty.
    return [name for name in current if name] or catalog


def _provider_resolved_key(provider_cls) -> str:
    """The api key the provider itself resolves (its environment variables).

    Used when a custom api_base forces building the OpenAI client by hand,
    which would otherwise mask DEEPSEEK_API_KEY and friends. Providers raise
    when they find no key; fall back to a placeholder then, so endpoints that
    need no auth (local servers) keep working.
    """
    try:
        return provider_cls().client.api_key
    except Exception:
        return "unset"


def build_model(model: str, api_base: Optional[str] = None, api_key: Optional[str] = None,
                profile: Optional[str] = None) -> Model:
    """``profile`` only matters to providers whose credential is a login they
    read from the profile themselves, rather than a key passed in here."""
    provider_name, model_name = split_model_string(model)
    if provider_name == CHATGPT_PROVIDER:
        from . import chatgpt

        return chatgpt.build_model(model_name, profile)
    provider_cls = provider_class(provider_name)
    parameters = inspect.signature(provider_cls.__init__).parameters

    kwargs: dict = {}
    if api_base:
        if "base_url" in parameters:
            kwargs["base_url"] = api_base
        elif "openai_client" in parameters:
            # OpenAI-compatible providers with a fixed base_url (zai, deepseek,
            # moonshotai, ...) only take a custom endpoint via a full client.
            from openai import AsyncOpenAI

            kwargs["openai_client"] = AsyncOpenAI(
                base_url=api_base, api_key=api_key or _provider_resolved_key(provider_cls))
        else:
            raise ValueError(f"Provider {provider_name!r} does not support a custom api_base")
    if api_key and "openai_client" not in kwargs and "api_key" in parameters:
        kwargs["api_key"] = api_key

    provider = provider_cls(**kwargs)
    return infer_model(f"{provider_name}:{model_name}", provider_factory=lambda _: provider)


def request_settings(model: Model, cache_key: Optional[str] = None) -> dict:
    """The settings every request carries.

    ``cache_key`` names the conversation the request belongs to. OpenAI
    spreads requests over machines that each keep their own prompt cache, and
    without a key to route by, a request often lands where its prefix was
    never seen and is billed in full.
    """
    settings: dict = {"extra_headers": {"User-Agent": user_agent()}}
    if cache_key and model.system == "openai":
        settings["openai_prompt_cache_key"] = cache_key
    return settings


async def ask_once(model: Model, messages: list[ModelMessage], *, max_tokens: int,
                   tools: Sequence = (), cache_key: Optional[str] = None) -> str:
    """One request outside the turn loop, answered as plain text.

    Not streamed and not retried: the callers are a checkpoint summary and a
    recap, which have nobody watching the words arrive and a caller that
    decides what a failure means. A reply holding only tool calls comes back
    as the empty string.
    """
    response = await model_request(
        model,
        messages,
        model_settings={"max_tokens": max_tokens, **request_settings(model, cache_key)},
        model_request_parameters=ModelRequestParameters(
            function_tools=list(tools), allow_text_output=True),
    )
    return "".join(part.content for part in response.parts if isinstance(part, TextPart))
