"""User-defined LLM provider registry.

Holds the ``[providers.<name>]`` config section in a lightweight module with no
heavy imports (pydantic-ai, SDKs), so the config engine can register providers
without paying ``kompend.llm.models``' import cost (which pulls pydantic-ai,
openai, ollama, …) on every ``kompend`` invocation — including ``--help``.
"""

from __future__ import annotations

from typing import Any

from kompend.logging import get_logger

logger = get_logger(__name__)

__all__ = ["_user_providers", "effective_workers", "provider_max_connections", "set_user_providers"]


#: Provider-level keys that left the schema and now migrate to a canonical key.
#: ``context_window_cap`` → ``context_window``.
_DEPRECATED_PROVIDER_KEYS: dict[str, str] = {"context_window_cap": "context_window"}

#: Module-level store for user-defined providers. Maps provider_id -> dict with
#: keys: base_url, api_key (optional), api_type (optional), context_window
#: (optional), models (optional; per-model settings keyed by model name, each
#: with context_window).
_user_providers: dict[str, dict[str, Any]] = {}


def set_user_providers(providers: dict[str, dict[str, Any]]) -> None:
    """Register user-defined providers for ``build_model()`` resolution.

    Called during config resolution from the merged config. Each entry is a
    ``[providers.<name>]`` TOML section as a dict (base_url, api_key,
    api_type, context_window, models, max_connections).

    Deprecated keys are migrated **in place**: the caller's nested dict is the
    same object ``KompendConfig`` later validates, so the old key has to be gone
    by then.
    """
    _user_providers.clear()
    for name, provider in providers.items():
        _migrate_deprecated_provider_keys(name, provider)
        _user_providers[name] = provider


def _migrate_deprecated_provider_keys(name: str, provider: dict[str, Any]) -> None:
    """Move deprecated provider keys onto their canonical names, in place."""
    for old_key, new_key in _DEPRECATED_PROVIDER_KEYS.items():
        if old_key not in provider:
            continue
        value = provider.pop(old_key)
        if provider.get(new_key) is None:
            provider[new_key] = value
            logger.warning(
                "Config key 'providers.%s.%s' is deprecated — use '%s' instead. The value has been migrated; please update your config.",
                name,
                old_key,
                new_key,
            )
        else:
            logger.warning(
                "Config key 'providers.%s.%s' is deprecated and was ignored — '%s' is already set. Please delete it from your config.",
                name,
                old_key,
                new_key,
            )


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


def provider_max_connections(provider_id: str) -> int | None:
    """Client-side connection limit for *provider_id* (None = unlimited).

    Reads ``[providers.<name>].max_connections`` — works for both
    custom-defined endpoints and override-only blocks against integrated
    providers (``[providers.ollama]`` with just ``max_connections``).
    """
    return _user_providers.get(provider_id, {}).get("max_connections")
