"""Standalone tier-1 ingestion: parse sources and cache structured documents.

This module performs only the parsing/caching tier (tier 1) of the KNOX
pipeline. It is used by ``knox ingest`` so users can pre-populate the
cache without running the full compile pipeline.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from knox.core.models import MetadataField
from knox.core.paths import VaultPaths
from knox.core.tier_state import TierStateStore
from knox.logging import get_logger
from knox.preprocessing.dedup import DedupOrchestrator
from knox.preprocessing.parsers.base import ParserRouter

logger = get_logger(__name__)

ProgressFn = Callable[[int, int, str], None]

# Length of the content-hash suffix appended to parse-cache filenames.
# 8 hex chars (32 bits) plus the stem prefix makes collisions astronomically
# unlikely within a single corpus while keeping filenames human-readable.
_CACHE_HASH_LEN = 8


def _content_hash(source_file: Path) -> str:
    """SHA-256 prefix of file contents, used to discriminate same-stem files."""
    return hashlib.sha256(source_file.read_bytes()).hexdigest()[:_CACHE_HASH_LEN]


def parse_cache_name(source_file: Path) -> str:
    """Content-discriminated parse-cache filename for *source_file*.

    Format: ``<stem>-<content-hash-prefix>.json``. Two files with the same
    stem but different content get different cache entries; the same file
    moved to a different subdirectory keeps its entry (content unchanged).
    """
    return f"{source_file.stem}-{_content_hash(source_file)}.json"


def parse_cache_glob(stem: str) -> str:
    """Glob pattern matching any parse-cache entry for files named ``<stem>``."""
    return f"{stem}-*.json"


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


def _extract_text_from_cached(json_path: Path) -> str:
    """Extract plain text from a cached parsed JSON document.

    Concatenates string-type element content. Falls back to reading the
    cached file's raw bytes when no elements are found.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Failed to read cached %s", json_path.name, exc_info=True)
        return json_path.read_text(errors="replace")
    text = ""
    for element in data.get("elements", []):
        content = element.get("content")
        if isinstance(content, str) and content.strip():
            text += content + "\n\n"
    return text or json_path.read_text(errors="replace")


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
    serialised to ``base_dir/.knox/cache/<stem>-<hash>.json``.
    Tier-1 state is recorded so ``knox compile --start-at-tier chunk`` can
    verify prerequisites.

    When *no_cache* is ``False`` (default), files whose content hash matches
    an existing cache entry are skipped — the cached JSON is reused and the
    file is not re-parsed.

    When *config* contains ``dedup.enabled = True``, content-level duplicate
    detection runs in two passes:

    1. **Parse pass** (GPU-bound parsers, one model at a time): every file
       is routed to a parser and the structured document is cached.
    2. **Embedding pass** (dedup only, GPU if configured): fingerprints
       (including embedding network calls) are computed from cached JSON
       documents. If the embedding model is misconfigured or the provider
       is unreachable, the error is fatal and aborts ingest immediately.

    Similarity comparison (pure CPU — no network calls) runs at end of
    ingest using the pre-computed fingerprints. Documents identified as
    ``identical`` are counted in ``dedup_skipped``.

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

    def _rel_path(p: Path) -> Path:
        try:
            return Path(str(p.relative_to(base_dir)))
        except ValueError:
            return p

    cache_dir = paths.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    state_store = TierStateStore(paths.compile_state_path)
    state = state_store.load()

    count = cached = skipped = failed = 0

    # ── Dedup orchestrator (created once, used in pass 2) ──────────
    # Fingerprints are computed AFTER all files are parsed so that GPU-bound
    # parsers and the embedding model don't compete for GPU memory.
    dedup_enabled = bool((config or {}).get("dedup.enabled"))
    dedup_orch: DedupOrchestrator | None = DedupOrchestrator(base_dir, config=config or {}) if dedup_enabled else None
    dedup_fps: dict[str, dict[str, Any]] = {}

    for idx, source_file in enumerate(files):
        if progress_fn is not None:
            progress_fn(idx, total, source_file.name)

        # Cache key is stem + content-hash prefix: collision-proof across
        # same-stem files (e.g. numbered proceedings papers) yet stable when a
        # file is moved by `knox organize`. One entry per distinct content.
        # NOTE: the cache check MUST run before parser.parse() — Docling
        # extraction is expensive and must not repeat for cached files.
        out_file = cache_dir / parse_cache_name(source_file)

        if not no_cache and out_file.exists():
            logger.info("Skipping %s — already cached", source_file.name)
            cached += 1
            count += 1
            content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
            state_store.record(state, source_file.stem, _rel_path(source_file), content_hash, tier=1, ingested_by=ingested_by)
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

        data = asdict(doc)
        data[MetadataField.SOURCE_FILE] = str(_rel_path(Path(data[MetadataField.SOURCE_FILE])))
        out_file.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
        count += 1
        content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
        state_store.record(state, source_file.stem, _rel_path(source_file), content_hash, tier=1, ingested_by=ingested_by)

    # ── Pass 2: Dedup fingerprints (separate from parsing) ─────────
    # Fingerprints are computed AFTER all files are parsed so that GPU-bound
    # parsers (Docling) and the embedding model don't compete for GPU memory.
    # The embedding model loads once, fingerprints all documents, then unloads.
    if dedup_orch is not None:
        for idx, source_file in enumerate(files):
            if progress_fn is not None:
                progress_fn(idx, total, f"embedding {source_file.name}")
            slug = source_file.stem
            cached_file = cache_dir / parse_cache_name(source_file)
            if not cached_file.exists():
                continue
            text = _extract_text_from_cached(cached_file)
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
