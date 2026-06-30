"""models.dev catalog lookup for LLM context-window discovery.

Fetches the `models.dev <https://github.com/anomalyco/models.dev>`_ catalog
(a community-maintained JSON DB of AI model metadata) once per process and
resolves a KNOX/litellm model string to its ``(max_input, max_output)``
token limits.

All network access is optional: on any fetch/parse failure the lookup
returns ``None`` and the caller falls through to its existing fallback
path. This module never raises for "model not found" or "network down".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import httpx

from knox.logging import get_logger

logger = get_logger(__name__)

# Catalog endpoint (combined providers + model metadata in one response).
CATALOG_URL: Final[str] = "https://models.dev/catalog.json"
# Short timeout — discovery must not stall a pipeline build when the host
# is unreachable; a miss falls through to the fallback path.
_FETCH_TIMEOUT_S: Final[float] = 5.0

# Top-level keys of the catalog response (external API contract, read as literals).
_MODELS_KEY: Final[str] = "models"
_PROVIDERS_KEY: Final[str] = "providers"

# Field names inside a model's ``limit`` object (external API contract).
_LIMIT_KEY: Final[str] = "limit"
_LIMIT_CONTEXT: Final[str] = "context"
_LIMIT_INPUT: Final[str] = "input"
_LIMIT_OUTPUT: Final[str] = "output"

# In-memory cache: the catalog is static between releases, so fetch once
# per process. A second concurrent first-call would at worst double-fetch
# the same payload (benign); no lock is warranted.
_catalog: Mapping[str, Any] | None = None


def resolve_model_limits(
    model_id: str,
    *,
    catalog: Mapping[str, Any] | None = None,
) -> tuple[int, int] | None:
    """Resolve ``(max_input, max_output)`` token limits for *model_id*.

    *model_id* is a KNOX/litellm-style identifier (``"provider/model[:tag]"``).
    The provider prefix is matched against the catalog's ``providers`` table
    first (provider-local limits, e.g. an Ollama-Cloud serving cap), then
    against the global ``models`` table keyed by ``lab/model``.

    Ollama-style tags (``":cloud"``, ``":31b"``, ``":Q4_0"``) are stripped
    for matching, so ``"ollama-cloud/gemma4:cloud"`` resolves to the
    ``"gemma4:31b"`` entry.

    Returns ``None`` when the model is absent from the catalog, the catalog
    could not be loaded, or the entry lacks usable limit values. Callers
    MUST treat ``None`` as "unknown" and fall through to their own fallback.
    """
    data = catalog if catalog is not None else _load_catalog()
    if not data:
        return None
    entry = _lookup(data, model_id)
    if entry is None:
        return None
    limit = entry.get(_LIMIT_KEY)
    if not isinstance(limit, Mapping):
        return None
    # Prefer an explicit input cap when present (some models cap input below
    # the full context window); otherwise the context window is the input limit.
    max_input = _as_int(limit.get(_LIMIT_INPUT)) or _as_int(limit.get(_LIMIT_CONTEXT))
    max_output = _as_int(limit.get(_LIMIT_OUTPUT))
    if max_input is None or max_output is None:
        return None
    return max_input, max_output


def _lookup(data: Mapping[str, Any], model_id: str) -> Mapping[str, Any] | None:
    """Find the catalog entry for *model_id*.

    Provider-local lookup first (exact stem, then base, then tag-prefix
    match), global ``lab/model`` lookup second.
    """
    prefix, sep, stem = model_id.partition("/")
    base = (stem if sep else model_id).split(":", 1)[0]

    # 1. Provider-local entry — e.g. providers["ollama-cloud"]["models"]["deepseek-v4-flash"].
    if sep:
        providers = data.get(_PROVIDERS_KEY)
        if isinstance(providers, Mapping):
            prov = providers.get(prefix)
            if isinstance(prov, Mapping):
                found = _match_provider_model(prov.get(_MODELS_KEY), stem, base)
                if found is not None:
                    return found

    # 2. Global lab entry — e.g. models["deepseek/deepseek-v4-flash"].
    models = data.get(_MODELS_KEY)
    if isinstance(models, Mapping):
        for key_form in (model_id, stem, base, f"{prefix}/{base}" if sep else base):
            hit = models.get(key_form)
            if isinstance(hit, Mapping):
                return hit
        # Stem/family scan: match by the part after the lab "/" and before a tag.
        for mid, m in models.items():
            if isinstance(m, Mapping) and mid.split("/", 1)[-1].split(":", 1)[0] == base:
                return m
    return None


def _match_provider_model(
    provider_models: Any,
    stem: str,
    base: str,
) -> Mapping[str, Any] | None:
    """Match a stem/base against a provider's model table.

    Tries exact stem, exact base, then any key whose pre-tag name equals
    *base* (so ``"gemma4"`` matches ``"gemma4:31b"``).
    """
    if not isinstance(provider_models, Mapping):
        return None
    for key_form in (stem, base):
        hit = provider_models.get(key_form)
        if isinstance(hit, Mapping):
            return hit
    if base:
        for key, m in provider_models.items():
            if key.split(":", 1)[0] == base and isinstance(m, Mapping):
                return m
    return None


def _load_catalog() -> Mapping[str, Any]:
    """Return the cached catalog, fetching it on first use.

    On any network/parse failure the cache is set to an empty mapping so
    subsequent lookups short-circuit to "not found" without retrying.
    """
    global _catalog  # noqa: PLW0603 — process-wide memo of a static catalog
    if _catalog is not None:
        return _catalog
    try:
        resp = httpx.get(CATALOG_URL, timeout=_FETCH_TIMEOUT_S, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        # Optional-feature probe: a missing/offline catalog is not fatal —
        # callers fall through to their existing fallback. DEBUG keeps it
        # visible for diagnosis without crying wolf at WARNING.
        logger.debug("models.dev catalog unavailable — context lookup disabled", exc_info=True)
        _catalog = {}
        return _catalog
    if not isinstance(data, Mapping):
        _catalog = {}
        return _catalog
    _catalog = data
    return _catalog


def _as_int(value: Any) -> int | None:
    """Coerce a catalog value to ``int``, rejecting bools and junk."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
