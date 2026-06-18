"""Standalone tier-1 ingestion: parse sources and cache structured documents.

This module performs only the parsing/caching tier (tier 1) of the KNOX
pipeline. It is used by ``knox ingest`` so users can pre-populate the
cache without running the full compile pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from knox.core.paths import CACHE_SUBDIR, COMPILE_STATE_FILE, KNOX_DIR
from knox.core.tier_state import TierStateStore
from knox.preprocessing.parsers.base import ParserRouter

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of an ingestion pass."""

    count: int
    skipped: int
    failed: int
    cache_dir: Path


def ingest_documents(
    files: Sequence[Path],
    sources_dir: Path,
    router: ParserRouter,
    progress_fn: ProgressFn | None = None,
) -> IngestResult:
    """Parse every file and cache the structured documents.

    Each file is routed to a parser, parsed into a :class:`Document`, and
    serialised to ``sources_dir/.knox/cache/<stem>_<sha256[:12]>.json``.
    Tier-1 state is recorded so ``knox compile --start-at-tier chunk`` can
    verify prerequisites.

    Args:
        files: Source files to ingest (already resolved/copied).
        sources_dir: Project sources directory (cache and state live
            under ``sources_dir / .knox``).
        router: Parser router used to select a parser per file.
        progress_fn: Optional callback ``(current, total, detail)``
            invoked before each file is processed.

    Returns:
        :class:`IngestResult` with counts and the cache directory path.
    """
    total = len(files)
    cache_dir = sources_dir / KNOX_DIR / CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    state_store = TierStateStore(sources_dir / KNOX_DIR / COMPILE_STATE_FILE)
    state = state_store.load()

    count = skipped = failed = 0

    for idx, path in enumerate(files):
        if progress_fn is not None:
            progress_fn(idx, total, path.name)

        parser = router.route(path)
        if parser is None:
            logger.info("Skipping %s — no parser available", path.name)
            skipped += 1
            continue

        try:
            doc = parser.parse(path)
        except Exception:
            logger.debug("Failed to parse %s", path, exc_info=True)
            failed += 1
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        cache_key = f"{path.stem}_{digest[:12]}"
        out_path = cache_dir / f"{cache_key}.json"
        out_path.write_text(
            json.dumps(asdict(doc), indent=2, default=str),
            encoding="utf-8",
        )
        count += 1
        state_store.record(state, path.stem, path, digest, tier=1)

    if progress_fn is not None:
        progress_fn(total, total, "done")

    state_store.save(state)
    return IngestResult(count=count, skipped=skipped, failed=failed, cache_dir=cache_dir)
