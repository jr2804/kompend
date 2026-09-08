"""User-defined LLM provider registry.

Holds the ``[providers.<name>]`` config section in a lightweight module with no
heavy imports (pydantic-ai, SDKs), so the config engine can register providers
without paying ``knox.llm.models``' import cost (which pulls pydantic-ai,
openai, ollama, …) on every ``knox`` invocation — including ``--help``.
"""

from __future__ import annotations

from typing import Any

from knox.logging import get_logger

logger = get_logger(__name__)

__all__ = ["_user_providers", "effective_workers", "provider_max_connections", "set_user_providers"]

#: Module-level store for user-defined providers. Maps provider_id -> dict with
#: keys: base_url, api_key (optional), api_type (optional), context_window
#: (optional; deprecated alias context_window_cap), models (optional; per-model
#: settings keyed by model name, each with context_window).
_user_providers: dict[str, dict[str, Any]] = {}


def set_user_providers(providers: dict[str, dict[str, Any]]) -> None:
    """Register user-defined providers for ``build_model()`` resolution.

    Called during config resolution from the merged config. Each entry is a
    ``[providers.<name>]`` TOML section as a dict (base_url, api_key,
    api_type, context_window, models, max_connections).
    """
    _user_providers.clear()
    _user_providers.update(providers)


def provider_max_connections(provider_id: str) -> int | None:
    """Client-side connection limit for *provider_id* (None = unlimited).

    Reads ``[providers.<name>].max_connections`` — works for both
    custom-defined endpoints and override-only blocks against integrated
    providers (``[providers.ollama]`` with just ``max_connections``).
    """
    return _user_providers.get(provider_id, {}).get("max_connections")


def effective_workers(model_string: str, requested: int) -> int:
    """Clip a worker count to the provider's ``max_connections`` limit.

    The provider part of *model_string* (``provider/model``) is looked up in
    the registry; when a limit is set, returns ``min(requested, limit)`` and
    logs a warning naming both values so silent throttling never happens.
    No limit (None) or unknown provider → *requested* unchanged.

    This is the single place workers are clipped against provider
    connection budgets — every parallel stage (parse today, future
    parallel LLM stages) calls it.
    """
    provider_id = model_string.split("/", 1)[0] if "/" in model_string else model_string
    limit = provider_max_connections(provider_id)
    if limit is None or requested <= limit:
        return requested
    logger.warning(
        "Worker count clipped: %d requested, %s.max_connections=%d — using %d",
        requested,
        provider_id,
        limit,
        limit,
    )
    return limit
