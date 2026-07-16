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

import os

from models_dev import get_model_by_id, get_provider
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from knox.logging import get_logger

logger = get_logger(__name__)


# ── public API ─────────────────────────────────────────────────────────


def _parse_model_string(model_string: str) -> tuple[str, str]:
    """Split ``provider/model`` or ``provider:model`` into *(provider, model_name)*.

    Splits on the **first** ``/`` or ``:`` — whichever comes first.
    The remainder (model_name) may itself contain further slashes or colons.
    """
    slash_pos = model_string.find("/")
    colon_pos = model_string.find(":")

    # First separator wins
    if slash_pos == -1:
        sep_pos = colon_pos
    elif colon_pos == -1:
        sep_pos = slash_pos
    else:
        sep_pos = min(slash_pos, colon_pos)

    if sep_pos <= 0:
        raise ValueError(
            f"Invalid model string {model_string!r}. "
            f"Use 'provider/model' or 'provider:model' format "
            f"(e.g. 'ollama/qwen3', 'openai:gpt-5.2')."
        )

    return model_string[:sep_pos], model_string[sep_pos + 1 :]


def build_model(model_string: str) -> Model:
    """Build a native pydantic-ai Model from a KNOX model string.

    Resolution order: models-dev catalog → pydantic-ai native inference.
    """
    prefix, name = _parse_model_string(model_string)

    # 1. Try models-dev catalog (150+ providers)
    try:
        return _from_models_dev(prefix, name)
    except _ProviderUnsupported:
        pass

    # 2. Fall through to pydantic-ai native inference (ollama, openai,
    #    anthropic, google-gla, groq, mistral, cohere, etc.)
    try:
        return infer_model(f"{prefix}:{name}")
    except Exception as exc:
        msg = (
            f"Unknown model string {model_string!r}. "
            f"Verify the provider name is correct. "
            f"Original error: {exc}"
        )
        raise ValueError(msg) from exc


def resolve_model_limits(model_id: str) -> tuple[int, int]:
    """Resolve ``(max_input, max_output)`` token limits from models-dev catalog.

    Raises:
        ValueError: If *model_id* has no ``/`` or ``:`` separator.
        KeyError: If the provider or model is not in the catalog.
        LookupError: If the model entry has no limit data.
    """
    provider, model_name = _parse_model_string(model_id)

    model = get_model_by_id(provider, model_name)  # raises KeyError

    limit = model.limit
    if limit is None or limit.context is None:
        raise LookupError(f"No limit data for {model_id!r}")

    return limit.context, limit.output or limit.context


# ── internal helpers ───────────────────────────────────────────────────


class _ProviderUnsupported(Exception):
    """Provider not in models-dev catalog or has no API endpoint."""


def _from_models_dev(provider_id: str, model_name: str) -> Model:
    """Resolve via models-dev catalog; raises ``_ProviderUnsupported`` if not found."""
    try:
        provider = get_provider(provider_id)
    except KeyError:
        raise _ProviderUnsupported(provider_id) from None

    if not provider.api:
        raise _ProviderUnsupported(provider_id)

    # Use the env var(s) the catalog advertises for the API key.
    api_key = None
    for env_var in (provider.env or ()):
        value = os.environ.get(env_var)
        if value:
            api_key = value
            break

    base = provider.api.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"

    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base, api_key=api_key),
    )


__all__ = ["build_model", "resolve_model_limits"]
