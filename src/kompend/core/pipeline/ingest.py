"""Standalone tier-1 ingestion: parse sources and cache structured documents.

This module performs only the parsing/caching tier (tier 1) of the kompend
pipeline. It is used by ``kompend ingest`` so users can pre-populate the
cache without running the full compile pipeline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from kompend.core.cancel import CancelState, KompendCancelled
from kompend.core.config.rumdl import resolve_rumdl_config
from kompend.core.keys import (
    DEDUP_ENABLED,
)
from kompend.core.models import ElementType, MetadataField, ParsedDocument
from kompend.core.paths import VaultPaths
from kompend.core.pipeline.parse_cache import (
    PARSE_CACHE_PARSER_KEY,
    extract_text_from_cached,  # noqa: F401 — re-exported for capture.py
    extract_text_from_data,
    read_document,
    write_document,
)
from kompend.core.pipeline.parse_cache import (
    cache_key as document_cache_key,
)
from kompend.core.pipeline.state import TierStateStore
from kompend.core.store import vault_db_path
from kompend.core.text.encoding import normalize_to_utf8_path
from kompend.core.text.rumdl import normalize_element_markdown
from kompend.core.verbosity import is_verbose
from kompend.logging import get_logger
from kompend.preprocessing.dedup import DedupOrchestrator
from kompend.preprocessing.parsers.base import ParserRouter, known_extension
from kompend.types import JSONDict

# Length of the content-hash suffix appended to parse-cache keys.
_CACHE_HASH_LEN = 8

logger = get_logger(__name__)

ProgressFn = Callable[[int, int, str], None]


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
    tags: list[str] | None = None,
    attributes: dict[str, str] | None = None,
    rumdl_config: Path | None = None,
    flavor: str | None = None,
    cancel: CancelState | None = None,
) -> IngestResult:
    """Parse every file and cache the structured documents.

    Each file is routed to a parser, parsed into a :class:`Document`, and
    serialised to ``base_dir/.kompend/cache/<stem>-<hash>.json``.
    Tier-1 state is recorded so ``kompend compile --start-at-tier chunk`` can
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
            state live under ``base_dir / .kompend`` — the same location the
            pipeline reads from.
        router: Parser router used to select a parser per file.
        progress_fn: Optional callback ``(current, total, detail)``
            invoked before each file is processed.
        no_cache: When ``True``, re-parse every file even if a cache
            entry with a matching hash already exists.
        config: Resolved config dict (or ``None`` to skip dedup). Used to
            configure content-level duplicate detection.
        ingested_by: OS user to record as the ingest actor in source state
            (``tier_state`` table in ``.kompend/kompend.db``). Empty string records no identity.
        tags: Optional list of source tags to attach to each parsed document's
            metadata (stored under ``MetadataField.TAGS``).
        attributes: Optional key=value source attributes to attach to each
            parsed document's metadata (stored under
            ``MetadataField.ATTRIBUTES``).
        rumdl_config: Optional rumdl config path for the parse-time markdown
            normalisation pass (defaults to the vault's ``.kompend/rumdl.toml``).
        flavor: Optional rumdl flavor for that same pass.
        cancel: Optional cooperative cancellation token, checked between
            files; on cancel the accumulated tier-1 state is saved before
            ``KompendCancelled`` propagates.

    Returns:
        :class:`IngestResult` with counts and the cache directory source_file.
    """
    total = len(files)
    paths = VaultPaths(base_dir)

    def _relative(p: Path) -> Path:
        try:
            return Path(str(p.relative_to(base_dir)))
        except ValueError:
            return p

    cache_dir = paths.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = vault_db_path(paths.cache_dir)

    state_store = TierStateStore(paths.compile_state_file)
    state = state_store.load()

    count = cached = skipped = failed = 0

    def _check_cancel() -> None:
        """Stop between files on a graceful cancel, persisting tier-1 records."""
        if cancel is not None and cancel.requested():
            state_store.save(state)
            raise KompendCancelled(cancel.reason)

    # ── Dedup orchestrator (created once, used in pass 2) ──────────
    # Fingerprints are computed AFTER all files are parsed so that GPU-bound
    # parsers and the embedding model don't compete for GPU memory.
    dedup_enabled = bool((config or {}).get(DEDUP_ENABLED))
    dedup_orch: DedupOrchestrator | None = DedupOrchestrator(base_dir, config=config or {}) if dedup_enabled else None
    dedup_fps: dict[str, dict[str, Any]] = {}

    # Identity of the parser chain that will produce these entries. Switching
    # ``[ingest] parser`` changes it, which makes stale entries a cache miss.
    parser_fingerprint = router.fingerprint()

    for idx, source_file in enumerate(files):
        _check_cancel()
        if progress_fn is not None:
            progress_fn(idx, total, source_file.name)

        # Compute content hash once, reuse for cache key AND state store.
        # Avoids reading the entire file twice on cache-hit paths.
        content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
        doc_key = document_cache_key(source_file, content_hash)
        prior_data = read_document(db_path, doc_key)

        if not no_cache and prior_data is not None and parse_cache_is_current(prior_data, parser_fingerprint):
            logger.info("Skipping %s — already cached", source_file.name)
            cached += 1
            count += 1
            state_store.record(state, source_file.stem, _relative(source_file), content_hash, tier=1, ingested_by=ingested_by)
            continue

        parser = router.route(source_file)
        if parser is None:
            logger.info("Skipping %s — no parser available", source_file.name)
            skipped += 1
            continue

        try:
            parse_target = parse_target_path(source_file, cache_dir)
            doc = parser.parse(parse_target)
            restore_source_path(doc, parse_target=parse_target, source_file=source_file)
            normalize_element_markdown(
                doc.elements,
                rumdl_config=resolve_rumdl_config(base_dir, rumdl_config),
                flavor=flavor,
            )
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", source_file, exc, exc_info=is_verbose())
            failed += 1
            continue

        # Preserve user-set tags/attributes before the entry is overwritten.
        # A re-parse (forced, or after a parser change) must not drop what
        # ``kompend sources tag`` / ``attr`` wrote into the previous entry.
        preserve_cached_metadata(doc, prior_data)

        # Attach CLI-provided tags — union with any preserved tags.
        if tags:
            if doc.metadata is None:
                doc.metadata = {}
            existing_tags = list(doc.metadata.get(MetadataField.TAGS) or [])
            for tag in tags:
                if tag not in existing_tags:
                    existing_tags.append(tag)
            doc.metadata[MetadataField.TAGS] = existing_tags
        if attributes:
            if doc.metadata is None:
                doc.metadata = {}
            merged = dict(doc.metadata.get(MetadataField.ATTRIBUTES) or {})
            merged.update(attributes)
            doc.metadata[MetadataField.ATTRIBUTES] = merged

        # Portable cache: image paths relative to the vault, like source_file.
        relativize_image_paths(doc, base_dir)

        data = asdict(doc)
        data[MetadataField.SOURCE_FILE] = str(_relative(Path(data[MetadataField.SOURCE_FILE])))
        data[PARSE_CACHE_PARSER_KEY] = parser_fingerprint
        write_document(db_path, doc_key, data)
        count += 1
        state_store.record(state, source_file.stem, _relative(source_file), content_hash, tier=1, ingested_by=ingested_by)

    # ── Pass 2: Dedup fingerprints (separate from parsing) ─────────
    # Fingerprints are computed AFTER all files are parsed so that GPU-bound
    # parsers (Docling) and the embedding model don't compete for GPU memory.
    # The embedding model loads once, fingerprints all documents, then unloads.
    if dedup_orch is not None:
        # Load the store early so we can skip fingerprinting unchanged docs.
        dedup_orch._store.load()  # noqa: SLF001 — intentional access to load cached fingerprints
        for idx, source_file in enumerate(files):
            _check_cancel()
            slug = source_file.stem
            # Skip files already fingerprinted in a previous run.
            if slug in dedup_orch._store:  # noqa: SLF001
                continue
            if progress_fn is not None:
                progress_fn(idx, total, f"embedding {source_file.name}")
            key = document_cache_key(source_file, _content_hash(source_file))
            data = read_document(db_path, key)
            if data is None:
                continue
            text = extract_text_from_data(data)
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


def parse_target_path(source_file: Path, cache_dir: Path) -> Path:
    """The path a parser must read for *source_file*.

    The router has already decided that a backend claims the file, so this
    never rejects a source — it only decides *which bytes* the parser
    receives:

    - A format the table knows as text (``.md``, ``.txt``, ...) is handed
      over as UTF-8: the source itself when it already is UTF-8, otherwise
      the normalized sibling in the encoding cache.
    - A format the table knows as a binary container (``.pdf``, ``.docx``,
      raster) is handed over byte-wise — charset detection would reject it.
    - A suffix the table does not know (source code: ``.c``, ``.h``,
      ``.py``) is handed over byte-wise as well: the backend that claimed
      it reads the bytes and decides. The format table classifies sources,
      it never gates parsing.

    Shared by the ingest tier and the tier-1 parse stage so the two cannot
    drift apart.
    """
    extension = known_extension(source_file.suffix)
    if extension is None or not extension.is_text:
        return source_file
    normalized, _encoding = normalize_to_utf8_path(source_file, cache_dir)
    return normalized


def restore_source_path(doc: ParsedDocument, *, parse_target: Path, source_file: Path) -> None:
    """Point a parsed document back at the real source after an encoding sibling.

    ``parse_target_path`` may hand the parser a normalized copy in the
    encoding cache. The parser reports whatever path it read, so citations,
    cache entries and the source tree would otherwise reference
    ``.kompend/cache/encoding/<hash>.md``. Rewrite the document and every
    element that read *parse_target* to *source_file*.

    No-op when the parser already read the original file.
    """
    if parse_target == source_file:
        return
    doc.source_file = source_file
    for element in doc.elements:
        if element.source_file == parse_target:
            element.source_file = source_file


def parse_cache_is_current(data: JSONDict, parser_fingerprint: str) -> bool:
    """True when a parse-cache entry was produced by the current parser chain.

    Entries written before this key existed return ``False`` — a one-time
    re-parse that self-heals the cache.
    """
    return data.get(PARSE_CACHE_PARSER_KEY) == parser_fingerprint


def preserve_cached_metadata(doc: ParsedDocument, prior_data: JSONDict | None) -> None:
    """Carry user-set tags/attributes from the prior entry into *doc*.

    Called before every parse-cache rewrite. The parser produces content,
    never user metadata; without this a forced re-parse (``kompend ingest
    --no-cache``) or an automatic re-parse after a parser change would
    silently drop what ``kompend sources tag`` / ``attr`` wrote into the
    previous entry.

    *prior_data* is the entry being overwritten (read from the store before
    the write) — same content hash, so a re-parse of unchanged content
    preserves its metadata exactly. Merges are additive: tags are unioned,
    attributes kept with *doc*'s existing values winning on key conflict.
    No-op when *prior_data* is ``None``.
    """
    if prior_data is None:
        return
    prior_meta = prior_data.get("metadata") or {}
    if not isinstance(prior_meta, dict):
        return

    if doc.metadata is None:
        doc.metadata = {}

    prior_tags = cast("list[str]", prior_meta.get(MetadataField.TAGS) or [])
    if prior_tags:
        merged_tags = list(doc.metadata.get(MetadataField.TAGS) or [])
        for tag in prior_tags:
            if tag not in merged_tags:
                merged_tags.append(tag)
        doc.metadata[MetadataField.TAGS] = merged_tags

    prior_attributes = cast("dict[str, str]", prior_meta.get(MetadataField.ATTRIBUTES) or {})
    if prior_attributes:
        merged_attributes = dict(prior_attributes)
        merged_attributes.update(doc.metadata.get(MetadataField.ATTRIBUTES) or {})
        doc.metadata[MetadataField.ATTRIBUTES] = merged_attributes


def relativize_image_paths(doc: ParsedDocument, base_dir: Path) -> None:
    """Rewrite absolute image element content to paths relative to *base_dir*.

    Keeps the parse cache portable (no machine-specific absolute paths). Data
    URIs and ``[Image: ...]`` placeholders are left untouched. Idempotent:
    already-relative paths pass through unchanged (cache-hit path).
    """
    for el in doc.elements:
        if el.type is ElementType.IMAGE and isinstance(el.content, str):
            image_path = Path(el.content)
            if image_path.is_absolute():
                # Outside the vault: leave as-is. Within it: relativize.
                with suppress(ValueError):
                    el.content = image_path.relative_to(base_dir).as_posix()


def _content_hash(source_file: Path) -> str:
    """SHA-256 prefix of file contents, used to discriminate same-stem files."""
    return hashlib.sha256(source_file.read_bytes()).hexdigest()[:_CACHE_HASH_LEN]
