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
import pkgutil

import pydantic_ai.providers
from models_dev import Provider, get_model_by_id, get_provider
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers import infer_provider_class
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from knox.logging import get_logger

logger = get_logger(__name__)

# Default base URL for local Ollama — ``ollama`` is not in the models-dev
# catalog, and OllamaProvider has no built-in default (it raises UserError
# when neither ``base_url`` nor ``OLLAMA_BASE_URL`` is given). pydantic-ai's
# OllamaModel uses the OpenAI-compatible endpoint, so ``/v1`` is required in
# the base URL (the OpenAI SDK appends ``/chat/completions`` to it).
OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"

# Placeholder satisfies the OpenAI SDK's non-empty api_key requirement for
# local / keyless OpenAI-compatible endpoints.
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

    # Build a helpful list of known providers for the error message.
    known = _discover_provider_ids()
    known_str = ", ".join(sorted(known))
    msg = (
        f"Unknown provider {provider_id!r}: not natively supported by "
        f"pydantic-ai and not in the models-dev catalog. "
        f"Model strings use the format 'provider/model' (e.g. 'ollama/{model_string}').\n"
        f"  Known providers: {known_str}\n"
        f"  Got: {model_string!r} -> provider={provider_id!r}, model={model_name!r}"
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


def _discover_provider_ids() -> set[str]:
    """Discover all native provider IDs by scanning pydantic_ai.providers submodules.

    No hard-coded provider list — the scan discovers whatever pydantic-ai
    ships in its installed version. ``sentence_transformers`` is excluded
    because it's an embedding-only backend, not an LLM provider.
    """
    return {m.name for m in pkgutil.iter_modules(pydantic_ai.providers.__path__) if not m.name.startswith("_") and m.name != "sentence_transformers"}


def _split(model_string: str) -> tuple[str, str]:
    """Split ``provider/model`` or ``provider:model`` into *(provider, model_name)*.

    The first ``/`` or ``:`` (whichever comes first) is the separator; the
    model name may itself contain slashes or colons.
    """
    slash = model_string.find("/")
    colon = model_string.find(":")
    candidates = [p for p in (slash, colon) if p > 0]
    if not candidates:
        raise ValueError(f"Invalid model string {model_string!r}. Use 'provider/model' or 'provider:model'.")
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

    # Ollama is a well-known local provider with no models-dev catalog entry
    # and no built-in default base_url. Provide the local default so KNOX
    # works out of the box; OLLAMA_BASE_URL still wins when set.
    if provider_cls is OllamaProvider:
        return _build_ollama(model_name)

    provider = provider_cls()
    return infer_model(
        f"{provider_id}:{model_name}",
        provider_factory=lambda _name: provider,
    )


def _build_ollama(model_name: str) -> OllamaModel:
    """Construct an OllamaModel with a sane local default base_url.

    ``OLLAMA_BASE_URL`` (the env var pydantic-ai reads) wins when set;
    otherwise we fall back to the local default ``http://127.0.0.1:11434``
    so a stock local install works without any configuration.
    """
    base_url = os.getenv("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_BASE_URL
    provider = OllamaProvider(base_url=base_url)
    return OllamaModel(model_name, provider=provider)


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
        raise _NotInCatalog(f"Provider {provider_id!r} is in the catalog but has no API URL.")

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
