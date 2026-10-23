"""models.dev catalogue access with a local snapshot.

``llmdb`` resolves the same metadata the old bundled ``models-dev`` package
carried (provider API base URLs, env var names, model limits, modalities), and
adds an explicit ``refresh()``. That refresh downloads the whole payload
(≈6 MB) and ``llmdb`` keeps it in memory only, so a CLI that resolved a model
would pay the download on every invocation.

kompend therefore keeps the payload in a per-user cache file:

1. first use downloads once through ``llmdb.db.refresh()`` and writes the file;
2. later runs read the file, with no network access at all;
3. ``kompend doctor`` refreshes it when it is older than
   :data:`CACHE_TTL_HOURS`, so a new model shows up without waiting for a
   package release.

A failed download never breaks a run: a stale file stays in use, and with no
file at all the callers fall back to ``[providers.<name>]`` configuration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import llmdb

from kompend.logging import get_logger

#: Hours a cached snapshot stays usable before ``kompend doctor`` refreshes it.
CACHE_TTL_HOURS: Final[int] = 24 * 7

#: Cache file name inside the per-user cache directory.
_CACHE_NAME: Final[str] = "models.json"

#: Application directory under the cache root.
_APP_DIR: Final[str] = "kompend"

_FETCHED_KEY: Final[str] = "fetched_at"
_PROVIDERS_KEY: Final[str] = "providers"


#: Module-level holder for the in-process snapshot. A one-key dict keeps the
#: module free of ``global`` statements (ruff PLW0603) while still caching the
#: payload across calls.
_SNAPSHOT: dict[str, CatalogSnapshot] = {}

logger = get_logger(__name__)


@dataclass(frozen=True)
class CatalogSnapshot:
    """One models.dev payload: providers keyed by id, each with its models."""

    fetched_at: datetime
    providers: dict[str, llmdb.Provider]

    def provider(self, provider_id: str) -> llmdb.Provider | None:
        """Return the provider entry, or ``None`` when the id is unknown."""
        return self.providers.get(provider_id)

    def model(self, provider_id: str, model_id: str) -> llmdb.Model | None:
        """Return one model entry, or ``None`` when it is not listed."""
        provider = self.provider(provider_id)
        if provider is None:
            return None
        return provider.models.get(model_id)


def model(provider_id: str, model_id: str) -> llmdb.Model | None:
    """Return the catalogue entry for one model, or ``None``."""
    current = snapshot()
    return None if current is None else current.model(provider_id, model_id)


def ensure_fresh(ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    """Refresh when the cache is stale; returns ``True`` when a download happened."""
    if not is_stale(ttl_hours):
        return False
    return refresh()


def is_stale(ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    """True when the payload is missing or older than *ttl_hours*."""
    current_age = age()
    return current_age is None or current_age > timedelta(hours=ttl_hours)


def age() -> timedelta | None:
    """Age of the cached payload, or ``None`` when there is none."""
    stamp = fetched_at()
    return None if stamp is None else datetime.now(UTC) - stamp


def reset() -> None:
    """Drop the in-process snapshot (tests and long-lived processes)."""
    _install(None)


def _current() -> CatalogSnapshot | None:
    """Return the in-process snapshot, or ``None`` when none is loaded."""
    return _SNAPSHOT.get("value")


def _install(snapshot: CatalogSnapshot | None) -> None:
    """Replace the in-process snapshot (``None`` clears it)."""
    if snapshot is None:
        _SNAPSHOT.pop("value", None)
    else:
        _SNAPSHOT["value"] = snapshot


def cache_path() -> Path:
    """Location of the cached payload (``$XDG_CACHE_HOME/kompend/models.json``)."""
    root = os.environ.get("XDG_CACHE_HOME")
    base = Path(root) if root else Path.home() / ".cache"
    return base / _APP_DIR / _CACHE_NAME


def _read_cache(path: Path) -> CatalogSnapshot | None:
    """Load the cached payload, or ``None`` when it is missing or unreadable."""
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(str(raw[_FETCHED_KEY]))
        providers = {provider_id: llmdb.Provider.model_validate(entry) for provider_id, entry in raw[_PROVIDERS_KEY].items()}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Model catalogue cache is unreadable (%s) - refetching", exc)
        return None
    return CatalogSnapshot(fetched_at=fetched_at, providers=providers)


def _write_cache(path: Path, snapshot: CatalogSnapshot) -> None:
    """Persist *snapshot*; a write failure only costs the next download."""
    payload = {
        _FETCHED_KEY: snapshot.fetched_at.isoformat(),
        _PROVIDERS_KEY: {provider_id: provider.model_dump(mode="json") for provider_id, provider in snapshot.providers.items()},
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot write the model catalogue cache to %s: %s", path, exc)


def _fetch() -> CatalogSnapshot | None:
    """Download the payload through ``llmdb``; ``None`` when the API is unreachable."""
    try:
        llmdb.db.refresh()
    except Exception as exc:  # noqa: BLE001 - any transport/parse failure degrades to "no catalogue"
        logger.warning("Model catalogue refresh failed: %s", exc)
        return None
    providers = {provider.id: provider for provider in llmdb.db.providers.all()}
    return CatalogSnapshot(fetched_at=datetime.now(UTC), providers=providers)


def snapshot(*, refresh: bool = False) -> CatalogSnapshot | None:
    """Return the catalogue snapshot, downloading it only when unavoidable.

    Parameters
    ----------
    refresh:
        Ignore the cached payload and download a fresh one. Without it the
        cache is used as-is (however old), and a download happens only when no
        cache exists at all.
    """
    path = cache_path()
    current = _current()

    if refresh:
        fetched = _fetch()
        if fetched is not None:
            _install(fetched)
            _write_cache(path, fetched)
            return fetched
        if current is None:
            cached = _read_cache(path)
            _install(cached)
            return cached
        return current

    if current is not None:
        return current
    cached = _read_cache(path)
    if cached is not None:
        _install(cached)
        return cached
    fetched = _fetch()
    if fetched is not None:
        _install(fetched)
        _write_cache(path, fetched)
    return fetched


def provider(provider_id: str) -> llmdb.Provider | None:
    """Return the catalogue entry for *provider_id*, or ``None``."""
    current = snapshot()
    return None if current is None else current.provider(provider_id)


def fetched_at() -> datetime | None:
    """Timestamp of the cached payload, or ``None`` when there is none."""
    current = snapshot()
    return None if current is None else current.fetched_at


def refresh() -> bool:
    """Download a fresh payload and cache it; ``False`` when that failed."""
    before = _current()
    result = snapshot(refresh=True)
    return result is not None and (before is None or result.fetched_at > before.fetched_at)
