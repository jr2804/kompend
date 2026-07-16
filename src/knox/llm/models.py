"""Build native pydantic-ai Model instances and resolve model metadata.

KNOX uses the ``provider/model`` convention (e.g. ``ollama/qwen3``,
``openai/gpt-5.2``, ``claudinio/claude-4``). This module translates such
a string into a native pydantic-ai :class:`~pydantic_ai.models.Model`.

Resolution strategy (first match wins):

1. **Native pydantic-ai provider** — :func:`pydantic_ai.providers.infer_provider_class`
   is the smart dispatcher: it returns the matching ``Provider`` subclass
   for every provider pydantic-ai ships a class for (ollama, openai,
   anthropic, google, deepseek, groq, mistral, cohere, bedrock, …). We
   construct that provider, then :func:`~pydantic_ai.models.infer_model`
   builds the typed Model subclass (``OllamaModel``, ``AnthropicModel``,
   ``OpenAIResponsesModel``, …). No hard-coded string comparison on our
   side — pydantic-ai's dispatch resolves both the provider class and
   the model class.

2. **models-dev catalog** — for providers pydantic-ai doesn't ship a
   class for (custom OpenAI-compatible mirrors like ``claudinio``,
   ``opencode``), the catalog's ``api`` URL and advertised ``env`` var
   build a generic :class:`~pydantic_ai.models.openai.OpenAIChatModel`
   on an :class:`~pydantic_ai.providers.openai.OpenAIProvider`.

3. Neither → ``ValueError``.

API keys and endpoints are **never** injected via environment mutation.
Defaults are passed as constructor arguments; user overrides are read
from the env vars the provider/catalog advertises.
"""

from __future__ import annotations

import os

from models_dev import Provider, get_model_by_id, get_provider
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers import Provider as PaiProvider
from pydantic_ai.providers import infer_provider_class
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from knox.logging import get_logger

logger = get_logger(__name__)

# Placeholder satisfies the OpenAI SDK's non-empty api_key requirement for
# local / keyless OpenAI-compatible endpoints (mirrors pydantic-ai's own
# OllamaProvider behaviour).
_NO_KEY_PLACEHOLDER = "api-key-not-set"


# ── public API ─────────────────────────────────────────────────────────


def build_model(model_string: str) -> Model:
    """Build a native pydantic-ai Model from a KNOX ``provider/model`` string."""
    provider_id, model_name = _split(model_string)

    # 1. Native pydantic-ai provider?
    try:
        return _build_native(provider_id, model_name)
    except _NotNativeProvider:
        pass

    # 2. models-dev catalog (OpenAI-compatible custom provider)?
    try:
        return _build_catalog(provider_id, model_name)
    except _NotInCatalog:
        pass

    msg = (
        f"Unknown provider {provider_id!r}: not natively supported by "
        f"pydantic-ai and not in the models-dev catalog. "
        f"Check the provider prefix (before the first '/' or ':')."
    )
    raise ValueError(msg)


def resolve_model_limits(model_id: str) -> tuple[int, int]:
    """Resolve ``(max_input, max_output)`` token limits from the models-dev catalog.

    Raises:
        ValueError: If *model_id* has no ``/`` or ``:`` separator.
        KeyError: If the provider or model is not in the catalog.
        LookupError: If the model entry has no limit data.
    """
    provider_id, model_name = _split(model_id)
    model = get_model_by_id(provider_id, model_name)  # raises KeyError

    limit = model.limit
    if limit is None or limit.context is None:
        raise LookupError(f"No limit data for {model_id!r}")

    return limit.context, limit.output or limit.context


# ── parsing ────────────────────────────────────────────────────────────


def _split(model_string: str) -> tuple[str, str]:
    """Split ``provider/model`` or ``provider:model`` into *(provider, model_name)*.

    The first ``/`` or ``:`` (whichever comes first) is the separator; the
    model name may itself contain slashes or colons.
    """
    slash = model_string.find("/")
    colon = model_string.find(":")
    candidates = [p for p in (slash, colon) if p > 0]
    if not candidates:
        raise ValueError(
            f"Invalid model string {model_string!r}. "
            f"Use 'provider/model' or 'provider:model'."
        )
    pos = min(candidates)
    return model_string[:pos], model_string[pos + 1 :]


# ── path 1: native pydantic-ai provider ────────────────────────────────


class _NotNativeProvider(Exception):
    """Provider is not natively known to pydantic-ai."""


def _build_native(provider_id: str, model_name: str) -> Model:
    """Build via pydantic-ai's native provider/model dispatch.

    Raises :class:`_NotNativeProvider` when pydantic-ai has no class for
    *provider_id* (caller falls through to the catalog path).
    """
    try:
        provider_cls = infer_provider_class(provider_id)
    except ValueError as exc:
        raise _NotNativeProvider from exc

    provider = provider_cls(**_native_kwargs(provider_cls))
    return infer_model(
        f"{provider_id}:{model_name}",
        provider_factory=lambda _name: provider,
    )


def _native_kwargs(provider_cls: type[PaiProvider]) -> dict[str, str]:
    """Constructor kwargs a native provider needs beyond its own defaults.

    Hosted providers (OpenAI, Anthropic, Google, …) bake their endpoint
    into the client and read their own API-key env var, so they need no
    kwargs. Locally-served providers (Ollama) have no baked endpoint —
    pydantic-ai's ``OllamaProvider`` raises unless a ``base_url`` is given
    or ``OLLAMA_BASE_URL`` is set. We pass the endpoint directly (reading
    the user's override env var if present) rather than mutating the
    environment.
    """
    if provider_cls is OllamaProvider:
        return {"base_url": _ollama_base_url()}
    return {}


def _ollama_base_url() -> str:
    """Ollama OpenAI-compatible endpoint.

    pydantic-ai's ``OllamaProvider`` passes ``base_url`` verbatim to the
    OpenAI SDK, which appends ``/chat/completions`` — so the base URL must
    include the ``/v1`` segment. A user-set ``OLLAMA_BASE_URL`` (already in
    OpenAI format) wins over the standard local default.
    """
    return os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"


# ── path 2: models-dev catalog (OpenAI-compatible custom provider) ─────


class _NotInCatalog(Exception):
    """Provider is not in the models-dev catalog or has no API endpoint."""


def _build_catalog(provider_id: str, model_name: str) -> Model:
    """Build a generic OpenAI-compatible model from models-dev catalog data.

    The catalog's ``api`` field is the OpenAI-compatible base URL; the
    advertised ``env`` var carries the API key. Raises
    :class:`_NotInCatalog` when the provider is absent or has no ``api``.
    """
    try:
        catalog = get_provider(provider_id)
    except KeyError as exc:
        raise _NotInCatalog from exc

    if not catalog.api:
        raise _NotInCatalog(
            f"Provider {provider_id!r} is in the catalog but has no API URL."
        )

    base_url = _normalize_base_url(catalog.api)
    api_key = _catalog_api_key(catalog)
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
    )


def _normalize_base_url(api_url: str) -> str:
    """Ensure the catalog ``api`` URL ends with ``/v1``."""
    base = api_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _catalog_api_key(catalog: Provider) -> str:
    """API key from the first advertised env var that is set, else a placeholder."""
    for env_var in catalog.env:
        value = os.environ.get(env_var)
        if value:
            return value
    return _NO_KEY_PLACEHOLDER


__all__ = ["build_model", "resolve_model_limits"]
