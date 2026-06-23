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
from typing import Any

from knox.core.paths import VaultPaths
from knox.core.tier_state import TierStateStore
from knox.preprocessing.dedup import DedupOrchestrator
from knox.preprocessing.parsers.base import ParserRouter

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]


def _extract_text_for_dedup(doc: Any, source_file: Path) -> str:
    """Extract plain text from a parsed document for dedup fingerprinting.

    Concatenates all element text. Falls back to raw file contents when the
    document has no extractable elements.
    """
    text = ""
    elements = getattr(doc, "elements", None)
    if elements:
        for element in elements:
            element_text = getattr(element, "text", None)
            if element_text:
                text += element_text + "\n\n"
    return text or source_file.read_text(errors="replace")


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of an ingestion pass."""

    count: int
    cached: int = 0
    skipped: int = 0
    failed: int = 0
    dedup_skipped: int = 0
    cache_dir: Path = Path()


def ingest_documents(  # noqa: PLR0915 — many statements for structured ingest flow
    files: Sequence[Path],
    base_dir: Path,
    router: ParserRouter,
    progress_fn: ProgressFn | None = None,
    *,
    no_cache: bool = False,
    config: dict[str, Any] | None = None,
    ingested_by: str = "",
) -> IngestResult:
    """Parse every file and cache the structured documents.

    Each file is routed to a parser, parsed into a :class:`Document`, and
    serialised to ``base_dir/.knox/cache/<stem>.json``.
    Tier-1 state is recorded so ``knox compile --start-at-tier chunk`` can
    verify prerequisites.

    When *no_cache* is ``False`` (default), files whose content hash matches
    an existing cache entry are skipped — the cached JSON is reused and the
    file is not re-parsed.

    When *config* contains ``dedup.enabled = True``, content-level duplicate
    detection runs alongside extraction. Fingerprints (including embedding
    network calls) are computed per-document during the parse loop — if the
    embedding model is misconfigured, the error is fatal and aborts ingest
    immediately. Similarity comparison runs at end of ingest using the
    pre-computed fingerprints. Documents identified as ``identical`` are
    counted in ``dedup_skipped`` (they are still cached, but flagged for
    the compile pipeline to skip).

    Args:
        files: Source files to ingest (already resolved/copied).
        base_dir: Project root directory (the ``--wiki-dir``). Cache and
            state live under ``base_dir / .knox`` — the same location the
            pipeline reads from.
        router: Parser router used to select a parser per file.
        progress_fn: Optional callback ``(current, total, detail)``
            invoked before each file is processed.
        no_cache: When ``True``, re-parse every file even if a cache
            entry with a matching hash already exists.
        config: Resolved config dict (or ``None`` to skip dedup). Used to
            configure content-level duplicate detection.
        ingested_by: OS user to record as the ingest actor in source state
            (``compile_state.json``). Empty string records no identity.

    Returns:
        :class:`IngestResult` with counts and the cache directory source_file.
    """
    total = len(files)
    paths = VaultPaths(base_dir)
    cache_dir = paths.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    state_store = TierStateStore(paths.compile_state_path)
    state = state_store.load()

    count = cached = skipped = failed = 0

    # ── Dedup orchestrator (created once, used per-document) ────────
    # When enabled, fingerprints (including embedding network calls) are
    # computed DURING extraction — not deferred to the end. This makes
    # embedding errors fatal and attributable to the specific document.
    dedup_enabled = bool((config or {}).get("dedup.enabled"))
    dedup_orch: DedupOrchestrator | None = DedupOrchestrator(base_dir, config=config or {}) if dedup_enabled else None
    dedup_fps: dict[str, dict[str, Any]] = {}

    for idx, source_file in enumerate(files):
        if progress_fn is not None:
            progress_fn(idx, total, source_file.name)

        # Cache key is the file stem — stable even when raw file bytes
        # change (e.g. URL downloads with dynamic HTML). One entry per
        # source file; re-parsing overwrites, never orphans.
        # NOTE: the cache check MUST run before parser.parse() — Docling
        # extraction is expensive and must not repeat for cached files.
        out_file = cache_dir / f"{source_file.stem}.json"

        if not no_cache and out_file.exists():
            logger.info("Skipping %s — already cached", source_file.name)
            cached += 1
            count += 1
            state_store.record(state, source_file.stem, source_file, hashlib.sha256(source_file.read_bytes()).hexdigest(), tier=1, ingested_by=ingested_by)
            continue

        parser = router.route(source_file)
        if parser is None:
            logger.info("Skipping %s — no parser available", source_file.name)
            skipped += 1
            continue

        try:
            doc = parser.parse(source_file)
        except Exception:
            logger.debug("Failed to parse %s", source_file, exc_info=True)
            failed += 1
            continue

        out_file.write_text(
            json.dumps(asdict(doc), indent=2, default=str),
            encoding="utf-8",
        )
        count += 1
        state_store.record(state, source_file.stem, source_file, hashlib.sha256(source_file.read_bytes()).hexdigest(), tier=1, ingested_by=ingested_by)

        # ── Compute dedup fingerprint NOW (during extraction) ────────
        # Embedding network calls happen here. If the model is misconfigured
        # or the provider is unreachable, the exception propagates — the
        # ingest ABORTS with a clear per-document error. This is intentional:
        # dedup was explicitly enabled, so silent degradation is worse than
        # crashing.
        if dedup_orch is not None:
            slug = source_file.stem
            text = _extract_text_for_dedup(doc, source_file)
            dedup_fps[slug] = dedup_orch.fingerprint(slug, text)

    # ── Content-level duplicate comparison (end of ingest) ──────────
    # Pure CPU — no network calls. Uses pre-computed fingerprints.
    dedup_skipped = 0
    if dedup_orch is not None and dedup_fps:

        def _dedup_progress(completed: int, total_cmp: int) -> None:
            if progress_fn is not None:
                progress_fn(completed, max(total_cmp, 1), f"dedup {completed}/{total_cmp}")

        results = dedup_orch.compare(dedup_fps, progress_fn=_dedup_progress)
        for slug, result in results.items():
            if result.verdict.value == "identical":
                dedup_skipped += 1
                logger.info("Dedup: %s is identical to %s (score=%.3f) — skipping compile", slug, result.closest_match, result.closest_score)
            elif result.verdict.value == "similar":
                logger.info("Dedup: %s is similar to %s (score=%.3f) — will add warning", slug, result.closest_match, result.closest_score)
        if dedup_skipped:
            logger.info("Dedup: %d document(s) identified as identical and will be skipped during compile", dedup_skipped)

    if progress_fn is not None:
        progress_fn(total, total, "done")

    state_store.save(state)
    return IngestResult(
        count=count,
        cached=cached,
        skipped=skipped,
        failed=failed,
        dedup_skipped=dedup_skipped,
        cache_dir=cache_dir,
    )
