"""Model context-window discovery via the models.dev catalog.

Fetches ``https://models.dev/catalog.json`` and caches it both in-memory
(per process) and on disk (``~/.config/knox/cache/models-dev-catalog.json``,
shared across invocations). Disk cache freshness is governed by TTL.

Matching strategy (first hit wins):

1. ``litellm.get_model_info()`` — static tables for major providers.
2. models.dev catalog — exact key, then provider-local stem, then
   normalized base + size fuzzy match (bridges ``ollama-cloud/gemma4:31b``
   to ``google/gemma-4-31b-it``).

Returns ``None`` when unknown — the caller decides how to handle the miss
(warn + fallback, or fail). Network failures are caught narrowly
(``httpx.HTTPError``, ``ValueError``) and logged at DEBUG — never raises.

Cache control:

- ``KNOX_MODELS_CACHE_TTL`` — env var, cache lifetime in **hours**
  (default 168 = 7 days). Set to ``0`` to force a fresh fetch every call.
- :func:`refresh_catalog` — programmatic force-refresh (clears in-memory
  and deletes the disk cache file).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
import litellm

from knox.logging import get_logger

logger = get_logger(__name__)

_CATALOG_URL = "https://models.dev/catalog.json"
_CACHE_FILE = "models-dev-catalog.json"
_TTL_DEFAULT_HOURS = 168.0
# Mutable holder avoids `global`: {"catalog": dict|None}. None = not fetched yet.
_CACHE: dict[str, object] = {"catalog": None}


# ── Cache location & TTL ──────────────────────────────────────────────


def _global_cache_dir() -> Path:
    """Return the user-global knox cache directory (XDG_CONFIG_HOME/knox/cache)."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "knox" / "cache"


def _cache_path() -> Path:
    return _global_cache_dir() / _CACHE_FILE


def _cache_ttl_seconds() -> float:
    """TTL in seconds. ``KNOX_MODELS_CACHE_TTL`` (hours); 0 = always refresh."""
    raw = os.environ.get("KNOX_MODELS_CACHE_TTL")
    if not raw:
        return _TTL_DEFAULT_HOURS * 3600
    try:
        return float(raw) * 3600
    except ValueError:
        logger.debug("KNOX_MODELS_CACHE_TTL=%r is not a number; using default", raw)
        return _TTL_DEFAULT_HOURS * 3600


# ── Disk + remote fetch ───────────────────────────────────────────────


def _load_disk_cache(path: Path, ttl: float) -> dict[str, dict[str, Any]] | None:
    """Load the disk cache if it exists and is fresher than *ttl* seconds."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = data.get("fetched_at")
        if isinstance(fetched_at, (int, float)) and (time.time() - fetched_at) < ttl:
            models = data.get("models")
            if isinstance(models, dict):
                return models
    except (OSError, ValueError, TypeError):
        logger.debug("models.dev disk cache read failed", exc_info=True)
    return None


def _save_disk_cache(path: Path, models: dict[str, dict[str, Any]]) -> None:
    """Persist the catalog to disk (best-effort)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fetched_at": time.time(), "models": models}
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        logger.debug("models.dev disk cache write failed", exc_info=True)


def _fetch_remote() -> dict[str, dict[str, Any]]:
    """Fetch the catalog from models.dev. Returns ``{}`` on failure."""
    try:
        resp = httpx.get(_CATALOG_URL, timeout=10, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models") if isinstance(data, dict) else None
        return models if isinstance(models, dict) else {}
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("models.dev catalog fetch failed: %s", exc)
        return {}


def _fetch_catalog() -> dict[str, dict[str, Any]]:
    """Resolve the catalog: in-memory → disk (if fresh) → remote → disk write."""
    cached = _CACHE["catalog"]
    if cached is not None:
        return cached  # type: ignore[return-value]
    ttl = _cache_ttl_seconds()
    path = _cache_path()
    if ttl > 0:
        disk = _load_disk_cache(path, ttl)
        if disk is not None:
            _CACHE["catalog"] = disk
            return disk
    fresh = _fetch_remote()
    _CACHE["catalog"] = fresh
    if fresh:
        _save_disk_cache(path, fresh)
    return fresh


def refresh_catalog() -> None:
    """Force a fresh fetch on the next access (clears in-memory + disk cache)."""
    _CACHE["catalog"] = None
    try:
        _cache_path().unlink(missing_ok=True)
    except OSError:
        logger.debug("models.dev disk cache delete failed", exc_info=True)


# ── Matching ──────────────────────────────────────────────────────────


def _normalize_stem(name: str) -> str:
    """Normalize a model stem for fuzzy matching.

    ``gemma4`` → ``gemma-4``; separators collapsed; tag stripped.
    """
    s = name.lower().split(":", 1)[0]
    s = re.sub(r"([a-z])(\d)", r"\1-\2", s)
    return re.sub(r"[-_.]+", "-", s).strip("-")


def _extract_size(model_id: str) -> str | None:
    """Extract a size token (``31b``, ``7b``, ``70b``) from *model_id*."""
    m = re.search(r"(\d+\s*b\b)", model_id.lower())
    return m.group(1).replace(" ", "") if m else None


def _catalog_limits(entry: dict[str, Any]) -> tuple[int, int] | None:
    """Extract ``(max_input, max_output)`` from a catalog entry."""
    limit = entry.get("limit")
    if not isinstance(limit, dict):
        return None
    ctx = limit.get("context")
    out = limit.get("output") or ctx
    if isinstance(ctx, int) and isinstance(out, int):
        return ctx, out
    return None


def _match_catalog(model_id: str, catalog: dict[str, dict[str, Any]]) -> tuple[int, int] | None:
    """Match *model_id* against the catalog. Returns limits or ``None``."""
    # 1. Exact key (e.g. openai/gpt-4o)
    if model_id in catalog:
        return _catalog_limits(catalog[model_id])
    # 2. Provider-local stem exact (ollama-cloud/gemma4:31b → gemma4:31b / gemma4)
    _, _, stem = model_id.partition("/")
    base = stem.split(":", 1)[0] if stem else model_id.split(":", 1)[0]
    if stem and stem in catalog:
        return _catalog_limits(catalog[stem])
    if base and base in catalog:
        return _catalog_limits(catalog[base])
    # 3. Fuzzy: normalized base prefix + size token
    q_norm = _normalize_stem(base or model_id)
    q_size = _extract_size(model_id)
    if not q_norm:
        return None
    for key, entry in catalog.items():
        _, _, k_stem = key.partition("/")
        k_base = k_stem.split(":", 1)[0]
        k_norm = _normalize_stem(k_base)
        if k_norm != q_norm and not k_norm.startswith(q_norm + "-"):
            continue
        if q_size and q_size not in k_stem.lower():
            continue
        limits = _catalog_limits(entry)
        if limits:
            return limits
    return None


def resolve_model_limits(
    model_id: str,
    **kwargs: Any,  # noqa: ARG001 — accept `catalog` param for backward compat
) -> tuple[int, int] | None:
    """Resolve ``(max_input, max_output)`` token limits for *model_id*.

    Tries litellm, then the models.dev catalog. Returns ``None`` when
    unknown — no guessed values.
    """
    _, _, stem = model_id.partition("/")
    lookup = stem or model_id

    # 1. litellm static tables
    try:
        info = litellm.get_model_info(lookup)
        if isinstance(info, dict):
            max_in = info.get("max_input_tokens") or info.get("max_tokens")
            max_out = info.get("max_output_tokens") or max_in
            if max_in and max_out:
                return int(max_in), int(max_out)
    except Exception:  # noqa: BLE001 — optional probe, never fatal
        logger.debug("litellm.get_model_info(%r) failed", lookup, exc_info=True)

    # 2. models.dev catalog
    catalog = _fetch_catalog()
    if catalog:
        limits = _match_catalog(model_id, catalog)
        if limits:
            return limits

    return None
