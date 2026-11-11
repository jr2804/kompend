"""Build a model against an endpoint an ACP client injected via ``providers/set``.

The ACP provider payload carries an ``apiType``, a ``baseUrl`` and a header map
— no model id. So an override changes *where* requests go, not *which* model
answers: the model name still comes from ``compile.llm.default_model``, and its
provider part is replaced by the injected endpoint. A session with an endpoint
but no configured model still fails with the house sentence.

The built model is a stock pydantic-ai model on a stock provider, constructed
from public API only — the same shape :func:`kompend.llm.models.build_model`
uses for a ``[providers.<name>]`` block, but with the endpoint taken from the
store instead of the config.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Final

import acp
from pydantic_ai.models import Model, ModelSelectionContext

from kompend.acp.model_selection import NO_MODEL_MESSAGE
from kompend.acp.provider_store import LlmApiType, ProviderConfig, ProviderCredentials
from kompend.core.config import resolve_config

#: Placeholder satisfies the OpenAI SDK's non-empty api_key requirement for a
#: keyless endpoint. Mirrors ``kompend.llm.models``.
_NO_KEY_PLACEHOLDER: Final[str] = "api-key-not-set"


class EndpointOverrideSelector:
    """Prefer the injected endpoint, else defer to the configured-model selector.

    Callable per the ``ModelSelector`` contract, so an instance can be handed to
    pydantic-ai's ``SelectModel``. The base selector is the session's
    :class:`kompend.acp.model_selection.ConfigModelSelector`, which raises the
    house sentence when nothing is configured.

    Args:
        base_dir: Workspace the session is rooted at.
        credentials: The process-wide ``providers/set`` store.
        fallback: The selector used when no endpoint is injected.
    """

    def __init__(
        self,
        base_dir: Path,
        credentials: ProviderCredentials,
        fallback: Callable[[ModelSelectionContext[None]], Model],
    ) -> None:
        self.base_dir = base_dir
        self.credentials = credentials
        self.fallback = fallback
        self._model: Model | None = None
        self._endpoint: ProviderConfig | None = None

    def __call__(self, ctx: ModelSelectionContext[None]) -> Model:
        """Return the model for the request step described by *ctx*."""
        provider = self.credentials.get()
        if provider is None:
            return self.fallback(ctx)
        if self._model is None or self._endpoint != provider:
            # A second ``providers/set`` must land on the next step, so the
            # cached model is keyed by the config it was built from.
            self._model = self._build(provider)
            self._endpoint = provider
        return self._model

    def _build(self, provider: ProviderConfig) -> Model:
        model_id = resolve_config(base_dir=self.base_dir).kompend.compile.llm.default_model
        if not model_id:
            raise acp.RequestError(-32603, NO_MODEL_MESSAGE)
        try:
            return build_endpoint_model(provider, model_name_of(model_id))
        except ValueError as exc:
            raise acp.RequestError(-32603, str(exc)) from exc


def model_name_of(model_id: str) -> str:
    """Return the model name of a ``provider/model`` id, or the id itself."""
    _, _, name = model_id.partition("/")
    return name or model_id


def build_endpoint_model(provider: ProviderConfig, model_name: str) -> Model:
    """Build *model_name* on the endpoint described by *provider*.

    Args:
        provider: The stored ``providers/set`` config.
        model_name: The model id to request, without a provider prefix.

    Raises:
        ValueError: the stored ``apiType`` names a protocol kompend cannot serve.
    """
    api_key = provider.api_key() or _NO_KEY_PLACEHOLDER
    if provider.api_type == LlmApiType.ANTHROPIC:
        anthropic_models = import_module("pydantic_ai.models.anthropic")
        anthropic_providers = import_module("pydantic_ai.providers.anthropic")
        anthropic_provider = anthropic_providers.AnthropicProvider(base_url=provider.base_url, api_key=api_key)
        return anthropic_models.AnthropicModel(model_name, provider=anthropic_provider)
    if provider.api_type not in {LlmApiType.OPENAI.value, LlmApiType.AZURE.value}:
        raise ValueError(
            f"ACP provider apiType {provider.api_type!r} is not supported; "
            f"supported: {LlmApiType.ANTHROPIC.value}, {LlmApiType.OPENAI.value}, {LlmApiType.AZURE.value}.",
        )
    openai_models = import_module("pydantic_ai.models.openai")
    openai_providers = import_module("pydantic_ai.providers.openai")
    openai_provider = openai_providers.OpenAIProvider(base_url=provider.base_url, api_key=api_key)
    return openai_models.OpenAIChatModel(model_name, provider=openai_provider)


__all__ = ["EndpointOverrideSelector", "build_endpoint_model", "model_name_of"]
