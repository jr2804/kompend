"""Prompt assembly for the generation pass.

Context blocks are rendered from configured categories and the on-disk wiki
cache; templates stay resources, never Python literals.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from kompend.compilation.two_phase import TwoPhaseRuntime
from kompend.core.models import load_cached_wiki
from kompend.core.models.enums import ExtractionField, MetadataField
from kompend.logging import get_logger

#: SQLite codes that mean the cache is busy, not broken: another process holds it.
_TRANSIENT_SQLITE_CODES = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})

logger = get_logger(__name__)


def _categories_block(extras: dict[str, str]) -> str:
    """Render configured extra categories as prompt instruction lines."""
    return "\n".join(f'- "{name}" — {desc}'.rstrip(".") for name, desc in extras.items())


def build_known_titles(compiler: TwoPhaseRuntime, concepts: list[dict[str, Any]]) -> list[str]:
    """Canonical page titles the generation LLM should link to verbatim.

    Combines the merged concept names from this compile with the canonical
    titles learned by the *previous* compile (its
    ``MetadataField.WIKILINK_ALIASES`` map, read from the on-disk wiki cache
    — still the previous compile's output at generation time, since the
    cache is written after generation). Deduplicated, order-preserving.
    """
    titles: list[str] = []
    seen: set[str] = set()

    def _add(title: str) -> None:
        if title and title not in seen:
            seen.add(title)
            titles.append(title)

    for c in concepts:
        _add(str(c.get(ExtractionField.NAME, "")))

    try:
        previous = load_cached_wiki(compiler._config.paths.base_dir)
    except sqlite3.OperationalError as exc:
        if getattr(exc, "sqlite_errorcode", None) not in _TRANSIENT_SQLITE_CODES:
            raise
        # A locked cache is environmental, not corrupt: the previous compile's
        # titles are lost for this run, so it must stay visible.
        logger.warning("Wiki cache is locked; known titles from the previous compile are unavailable: %s", exc)
        previous = None
    except OSError as exc:
        # The cache is best-effort context for the prompt, never a reason to fail a compile.
        logger.warning("Wiki cache is unreadable; known titles from the previous compile are unavailable: %s", exc)
        previous = None
    if previous is not None:
        learned = previous.metadata.get(MetadataField.WIKILINK_ALIASES) or {}
        if isinstance(learned, dict):
            for canonical in learned.values():
                _add(str(canonical))

    return titles
