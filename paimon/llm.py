"""Build a pydantic-ai Model from the configured model string.

The config keeps litellm-style strings ("zai/glm-5.2") as well as pydantic-ai
style ones ("zai:glm-5.2"). The provider is constructed explicitly so the
configured api_key/api_base take precedence over environment variables.
"""

import inspect
from typing import Optional

from pydantic_ai.models import Model, infer_model
from pydantic_ai.providers import infer_provider_class


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


def build_model(model: str, api_base: Optional[str] = None, api_key: Optional[str] = None) -> Model:
    provider_name, model_name = split_model_string(model)
    provider_cls = infer_provider_class(provider_name)
    parameters = inspect.signature(provider_cls.__init__).parameters

    kwargs: dict = {}
    if api_base:
        if "base_url" in parameters:
            kwargs["base_url"] = api_base
        elif "openai_client" in parameters:
            # OpenAI-compatible providers with a fixed base_url (zai, deepseek,
            # moonshotai, ...) only take a custom endpoint via a full client.
            from openai import AsyncOpenAI

            kwargs["openai_client"] = AsyncOpenAI(base_url=api_base, api_key=api_key or "unset")
        else:
            raise ValueError(f"Provider {provider_name!r} does not support a custom api_base")
    if api_key and "openai_client" not in kwargs and "api_key" in parameters:
        kwargs["api_key"] = api_key

    provider = provider_cls(**kwargs)
    return infer_model(f"{provider_name}:{model_name}", provider_factory=lambda _: provider)
