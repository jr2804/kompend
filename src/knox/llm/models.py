"""Build native pydantic-ai Model instances and resolve model metadata.

KNOX uses the ``provider/model`` convention (e.g. ``ollama/qwen3``,
``openai/gpt-5.2``). This module translates those strings to native
pydantic-ai models.

Resolution order:
1. Look up the provider in the ``models-dev`` bundled catalog.
   * If the provider has a known ``api`` base URL (e.g. ``claudinio``,
     ``opencode-zen``, ``deepseek``), construct an OpenAI-compatible
     ``OpenAIChatModel`` with the discovered endpoint.
   * Otherwise fall through to pydantic-ai native resolution.
2. If ``models-dev`` has no entry for the provider either, try
   pydantic-ai's ``infer_model()`` which natively supports providers
   such as ``ollama``, ``openai``, ``anthropic``, ``google-gla``, etc.
3. If all resolution paths fail, raise ``ValueError``.

Model strings split at the **first** slash — the remainder is the
model name verbatim and may itself contain a colon (e.g.
``ollama/gemma4:e2b`` → provider ``ollama``, model ``gemma4:e2b``) or
further slashes (e.g. ``ollama/hf.co/qwen3.6``).
"""

from __future__ import annotations

import os

from models_dev import Limit, get_model_by_id, get_provider
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from knox.logging import get_logger

logger = get_logger(__name__)

# Provider ID aliases — KNOX prefix convention vs. models-dev naming.
_PROVIDER_ALIASES: dict[str, str] = {
    "opencode-zen": "opencode",
}

# Well-known local providers with fixed endpoints — not in models-dev catalog.
_LOCAL_PROVIDERS: dict[str, str] = {
    "ollama": "http://localhost:11434",
}


def build_model(model_string: str) -> Model:
    """Build a native pydantic-ai Model from a KNOX ``provider/model`` string.

    Resolution order: models-dev catalog → pydantic-ai native inference.
    """
    prefix, sep, name = model_string.partition("/")
    if not sep:
        raise ValueError(
            f"Invalid model string {model_string!r}. Use the format "
            f"'provider/model' (e.g. 'ollama/qwen3', 'openai/gpt-5.2')."
        )

    # 1. Try models-dev provider lookup — covers 150+ providers
    model = _from_models_dev(prefix, name)
    if model is not None:
        return model

    # 2. Fall through to pydantic-ai native inference (ollama, openai,
    #    anthropic, google-gla, groq, mistral, cohere, etc.)
    pydantic_ai_string = f"{prefix}:{name}"
    try:
        return infer_model(pydantic_ai_string)
    except Exception as exc:
        msg = (
            f"Unknown model model_string {model_string!r}. "
            f"Verify the provider name (prefix before '/') is correct. "
            f"Original error: {exc}"
        )
        raise ValueError(msg) from exc


def _from_models_dev(provider_id: str, model_name: str) -> Model | None:
    """Resolve via models-dev catalog; returns ``None`` if not found."""
    # Provider ID aliases — KNOX prefix convention vs. models-dev naming.
    resolved_id = _PROVIDER_ALIASES.get(provider_id, provider_id)

    # 1. Well-known local provider (not in models-dev catalog)
    if resolved_id in _LOCAL_PROVIDERS:
        return _local_openai_model(resolved_id, model_name)

    try:
        provider = get_provider(resolved_id)
    except KeyError:
        # Not in models-dev either — caller falls through to infer_model
        return None

    api_url = provider.api
    if not api_url:
        # Standard provider (openai, anthropic, etc.) — handled by
        # pydantic-ai native inference.
        return None

    # Custom OpenAI-compatible provider — look up the API key from
    # the env var(s) the catalog advertises.
    api_key = None
    for env_var in (provider.env or ()):
        value = os.environ.get(env_var)
        if value:
            api_key = value
            break

    base = api_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"

    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base, api_key=api_key),
    )


def _local_openai_model(provider_id: str, model_name: str) -> OpenAIChatModel:
    """Construct an OpenAI-compatible model for a well-known local provider.

    Respects provider-specific env-var overrides for the base URL:
    - ollama: ``OLLAMA_HOST``, then ``OLLAMA_BASE_URL``, then default
      ``http://localhost:11434``.
    """
    default_base = _LOCAL_PROVIDERS[provider_id]
    base_url = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_BASE_URL") or default_base if provider_id == "ollama" else default_base

    api = f"{base_url.rstrip('/')}/v1"
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=api, api_key=None),
    )


# ── models.dev catalog limits lookup ───────────────────────────────────


def resolve_model_limits(model_id: str) -> tuple[int, int] | None:
    """Resolve ``(max_input, max_output)`` token limits from the models.dev catalog.

    *model_id* is a KNOX ``provider/model`` string. Returns ``None`` when
    the model is not present in the catalog. The caller
    (``discover_context_window``) decides how to handle the miss.
    """
    provider, sep, model_name = model_id.partition("/")
    if not sep:
        return None

    resolved = _PROVIDER_ALIASES.get(provider, provider)
    # Local providers won't be in the catalog
    if resolved in _LOCAL_PROVIDERS:
        return None
    try:
        model = get_model_by_id(resolved, model_name)
    except KeyError:
        return None

    limit: Limit | None = model.limit
    if limit is None or limit.context is None:
        return None

    return limit.context, limit.output or limit.context


__all__ = ["build_model", "resolve_model_limits"]
