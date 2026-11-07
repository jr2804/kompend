"""The one vector-store call behind every search.

Hits are raw ``dict[str, object]`` rows: everything downstream (assembly,
ranking, graph expansion) works from those rows, and only this module talks
to the store.
"""

from __future__ import annotations

from collections.abc import Sequence

from kompend.agents.failure import record_unit_failure
from kompend.rag.vector_store import VectorStoreLike


def vector_search(
    store: VectorStoreLike | None,
    query: str,
    *,
    top_k: int,
    tags_all: Sequence[str] | None = None,
    tags_dnf: Sequence[Sequence[str]] | None = None,
    attributes_all: Sequence[str] | None = None,
    attribute_keys: Sequence[str] | None = None,
) -> list[dict[str, object]] | None:
    """Ask the store for raw hits; never raise.

    ``None`` means the store could not be asked at all (absent, unavailable)
    or the call failed — the caller returns no results rather than falling
    back, because a fallback would mask a broken store behind plausible
    keyword hits. An empty list means the store answered and found nothing;
    only that case is a fallback. A call error is recorded as a unit failure
    before it is swallowed.
    """
    if store is None or not store.available:
        return None
    try:
        return store.search(
            query,
            top_k=top_k,
            tags_all=tags_all,
            tags_dnf=tags_dnf,
            attributes_all=attributes_all,
            attribute_keys=attribute_keys,
        )
    except Exception as exc:
        record_unit_failure(stage="search.vector-search", unit=query, exc=exc)
        return None


def hit_str(hit: dict[str, object], key: str, default: str = "") -> str:
    """Read a string field from a vector-store hit, falling back to *default*."""
    value = hit.get(key)
    return value if isinstance(value, str) else default


def hit_opt_str(hit: dict[str, object], key: str) -> str | None:
    """Read an optional string field from a vector-store hit."""
    value = hit.get(key)
    return value if isinstance(value, str) else None


def hit_float(hit: dict[str, object], key: str, default: float = 0.0) -> float:
    """Read a float field from a vector-store hit."""
    value = hit.get(key)
    return value if isinstance(value, int | float) else default
