"""Build native pydantic-ai Model instances and resolve model metadata.

KNOX uses the ``provider/model`` convention (e.g. ``ollama/qwen3``,
``openai/gpt-5.2``). This module translates those strings to native
pydantic-ai models.

Resolution order:
1. Look up the provider in the ``models-dev`` bundled catalog.
   * If the provider has a known ``api`` base URL, construct an
     OpenAI-compatible ``OpenAIChatModel`` with the discovered endpoint
     and API key from the catalog-advertised env var.
   * Otherwise fall through to pydantic-ai native resolution.
2. If ``models-dev`` has no entry for the provider, try pydantic-ai's
   ``infer_model()`` which supports ``ollama``, ``openai``,
   ``anthropic``, ``google-gla``, ``groq``, ``mistral``, ``cohere``,
   etc.
3. If all resolution paths fail, raise ``ValueError``.
"""

from __future__ import annotations

import functools
import os

from models_dev import Limit, Provider, get_model_by_id, get_provider
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from knox.logging import get_logger

logger = get_logger(__name__)

logger = get_logger(__name__)


# ── models-dev catalog (cached) ────────────────────────────────────────


@functools.cache
def _get_provider_cached(provider_id: str) -> Provider | None:
    """Cached wrapper around ``models_dev.get_provider()``.

    Repeated lookups for the same provider avoid re-querying the catalog.
    """
    try:
        return get_provider(provider_id)
    except KeyError:
        return None


# ── public API ─────────────────────────────────────────────────────────


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

    # 1. Try models-dev catalog (150+ providers)
    model = _from_models_dev(prefix, name)
    if model is not None:
        return model

    # 2. Fall through to pydantic-ai native inference (ollama, openai,
    #    anthropic, google-gla, groq, mistral, cohere, etc.)
    try:
        return infer_model(f"{prefix}:{name}")
    except Exception as exc:
        msg = (
            f"Unknown model model_string {model_string!r}. "
            f"Verify the provider name (prefix before '/') is correct. "
            f"Original error: {exc}"
        )
        raise ValueError(msg) from exc


def resolve_model_limits(model_id: str) -> tuple[int, int] | None:
    """Resolve ``(max_input, max_output)`` token limits from models-dev catalog.

    Returns ``None`` when the model is not present in the catalog.
    """
    provider, sep, model_name = model_id.partition("/")
    if not sep:
        return None
    try:
        model = get_model_by_id(provider, model_name)
    except KeyError:
        return None

    limit: Limit | None = model.limit
    if limit is None or limit.context is None:
        return None

    return limit.context, limit.output or limit.context


# ── internal helpers ───────────────────────────────────────────────────


def _from_models_dev(provider_id: str, model_name: str) -> Model | None:
    """Resolve via models-dev catalog; returns ``None`` if not found."""
    provider = _get_provider_cached(provider_id)
    if provider is None:
        return None

    api_url = provider.api
    if not api_url:
        # Standard provider (openai, anthropic, etc.) — handled by
        # pydantic-ai native inference.
        return None

    # Custom OpenAI-compatible provider — look up the API key from
    # the env var(s) the catalog advertises.
    api_key: str | None = None
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


__all__ = ["build_model", "resolve_model_limits"]
