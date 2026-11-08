"""Standalone tier-1 ingestion: parse sources and cache structured documents.

This module performs only the parsing/caching tier (tier 1) of the kompend
pipeline. It is used by ``kompend ingest`` so users can pre-populate the
cache without running the full compile pipeline.
"""

from __future__ import annotations

import hashlib
import time
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
from kompend.core.pipeline.bounded_parse import DeadlineWorker, ParseDeadlineExceeded
from kompend.core.pipeline.failure_bundle import write_failure_bundle
from kompend.core.pipeline.parse_cache import (
    PARSE_CACHE_PARSER_KEY,
    extract_text_from_cached,  # noqa: F401 — re-exported for capture.py
    extract_text_from_data,
    parse_cache_is_current,
    read_document,
    write_document,
)
from kompend.core.pipeline.parse_cache import (
    cache_key as document_cache_key,
)
from kompend.core.pipeline.reasons import reason_code_for
from kompend.core.pipeline.stall_watch import DEFAULT_STALL_WARN_S, StallWatchdog, WatchStage
from kompend.core.pipeline.state import TierStateStore
from kompend.core.reporting.ledger import StageLedger
from kompend.core.reporting.parse_quality import measure_parse_quality
from kompend.core.store import vault_db_path
from kompend.core.text.encoding import normalize_to_utf8_path
from kompend.core.text.rumdl import normalize_element_markdown
from kompend.core.types import ReasonCode, StageOutcome
from kompend.core.verbosity import is_verbose
from kompend.logging import get_logger
from kompend.preprocessing.dedup import DedupOrchestrator
from kompend.preprocessing.parsers.base import (
    DocumentParser,
    assert_text_source,
    known_extension,
)
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
    parser: DocumentParser,
    progress_fn: ProgressFn | None = None,
    *,
    reparse: bool = False,
    config: dict[str, Any] | None = None,
    ingested_by: str = "",
    tags: list[str] | None = None,
    attributes: dict[str, str] | None = None,
    rumdl_config: Path | None = None,
    flavor: str | None = None,
    cancel: CancelState | None = None,
    parse_timeout_s: float = 0.0,
    stall_warn_s: float | None = None,
) -> IngestResult:
    """Parse every file and cache the structured documents.

    Each file is routed to a parser, parsed into a :class:`Document`, and
    serialised to ``base_dir/.kompend/cache/<stem>-<hash>.json``.
    Tier-1 state is recorded so ``kompend compile --start-at-tier chunk`` can
    verify prerequisites.

    When *reparse* is ``False`` (default), files whose content hash matches
    an existing cache entry produced by the current backend identity are
    skipped — the cached document is reused and the file is not re-parsed.
    When the backend identity changes (backends available on the host, the
    extras installed, or the ``[ingest]`` backend restrictions), the entry is
    stale and the file is re-parsed even though its content did not change.

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
        parser: Document parser used to read each file.
        progress_fn: Optional callback ``(current, total, detail)``
            invoked before each file is processed.
        reparse: When ``True``, re-parse every file even if a cache
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
        parse_timeout_s: Wall-clock budget per file, in seconds. ``0``
            (default) parses in-process; a positive value parses each file in a
            child process that is killed on expiry, so a wedged backend costs
            one file instead of the run (see
            :mod:`kompend.core.pipeline.bounded_parse`).
        stall_warn_s: Seconds without a per-file heartbeat before the stall
            watchdog logs a one-time warning naming the file and stage (see
            :mod:`kompend.core.pipeline.stall_watch`). ``None`` uses the
            module default; ``0`` disables the watchdog.

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
    with StageLedger.open(db_path) as ledger:
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

        # One deadline worker for the whole run: a child per file would pay the
        # parser stack's import cost (seconds) on every source. It is a no-op when
        # no budget is configured.
        worker = DeadlineWorker(parse_timeout_s)

        # One stall watchdog for the whole run (knox-ty8f.8), beside the
        # deadline worker: the deadline bounds a hung backend, this names it
        # while it hangs — one warning per file whose heartbeat has been
        # silent for ``stall_warn_s`` (default 30 s, ``0`` disables). The
        # monitor thread is a daemon and never kills anything.
        watchdog = StallWatchdog(DEFAULT_STALL_WARN_S if stall_warn_s is None else stall_warn_s)
        watchdog.start()

        # One probe per run: the backend identity decides whether a cached parse
        # is still current for the parser that is running now.
        backend_identity = parser.backend_identity()

        for idx, source_file in enumerate(files):
            _check_cancel()
            if progress_fn is not None:
                progress_fn(idx, total, source_file.name)

            # Compute content hash once, reuse for cache key AND state store.
            # Avoids reading the entire file twice on cache-hit paths.
            raw = source_file.read_bytes()
            size_bytes = len(raw)
            content_hash = hashlib.sha256(raw).hexdigest()
            doc_key = document_cache_key(source_file, content_hash)
            prior_data = read_document(db_path, doc_key)

            if not reparse and prior_data is not None and parse_cache_is_current(prior_data, backend_identity):
                logger.info("Skipping %s — already cached", source_file.name)
                ledger.record(
                    str(_relative(source_file)),
                    "parse",
                    outcome=StageOutcome.CACHED,
                    parser=backend_identity,
                    cache_key=doc_key,
                    sha=content_hash,
                    size_bytes=size_bytes,
                )
                cached += 1
                count += 1
                state_store.record(state, source_file.stem, _relative(source_file), content_hash, tier=1, ingested_by=ingested_by)
                continue

            backend = parser.route(source_file)
            if backend is None:
                logger.info("Skipping %s — no parser available", source_file.name)
                ledger.record(
                    str(_relative(source_file)),
                    "parse",
                    outcome=StageOutcome.SKIPPED,
                    reason_code=ReasonCode.NO_PARSER,
                    cache_key=doc_key,
                    sha=content_hash,
                    size_bytes=size_bytes,
                )
                skipped += 1
                continue

            # Heartbeat keys match the ledger's source spelling exactly.
            source_key = str(_relative(source_file))
            watchdog.watch(source_key, stage=WatchStage.PARSE)
            attempt_start = time.perf_counter()
            parse_target: Path | None = None
            try:
                parse_target = parse_target_path(source_file, cache_dir)
                doc = worker.parse(backend, parse_target)
                watchdog.beat(source_key, stage=WatchStage.POSTPROCESS)
                restore_source_path(doc, parse_target=parse_target, source_file=source_file)
                normalize_element_markdown(
                    doc.elements,
                    rumdl_config=resolve_rumdl_config(base_dir, rumdl_config),
                    flavor=flavor,
                )
            except Exception as exc:
                reason = reason_code_for(exc)
                logger.warning(
                    "Failed to parse %s: %s",
                    source_file,
                    exc,
                    exc_info=is_verbose(),
                    extra={"reason_code": reason},
                )
                outcome = StageOutcome.TIMEOUT if isinstance(exc, ParseDeadlineExceeded) else StageOutcome.FAILED
                detail = str(exc)
                if isinstance(exc, ParseDeadlineExceeded):
                    # The deadline fires inside worker.parse, before any beat
                    # could follow registration, so the last stage is the
                    # backend call itself — name it in the row.
                    detail = f"{detail} Last progress: stage {watchdog.last_stage(source_key) or WatchStage.PARSE.value}."
                watchdog.unwatch(source_key)
                write_failure_bundle(
                    base_dir=base_dir,
                    source=str(_relative(source_file)),
                    source_file=source_file,
                    error=exc,
                    reason_code=reason,
                    stage="parse",
                    parse_target=parse_target,
                    parser=backend_identity,
                    backend=type(backend).__name__,
                    cache_key=doc_key,
                    sha=content_hash,
                    size_bytes=size_bytes,
                    detail=detail,
                )
                ledger.record(
                    str(_relative(source_file)),
                    "parse",
                    outcome=outcome,
                    duration_ms=(time.perf_counter() - attempt_start) * 1000.0,
                    reason_code=reason,
                    detail=detail,
                    parser=backend_identity,
                    backend=type(backend).__name__,
                    cache_key=doc_key,
                    sha=content_hash,
                    size_bytes=size_bytes,
                )
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

            ledger.record_parse_quality(
                measure_parse_quality(
                    doc,
                    source=str(_relative(source_file)),
                    sha=content_hash,
                    parser=backend_identity,
                    backend=type(backend).__name__,
                )
            )

            data = asdict(doc)
            data[MetadataField.SOURCE_FILE] = str(_relative(Path(data[MetadataField.SOURCE_FILE])))
            data[PARSE_CACHE_PARSER_KEY] = backend_identity
            write_document(db_path, doc_key, data)
            ledger.record(
                str(_relative(source_file)),
                "parse",
                outcome=StageOutcome.OK,
                duration_ms=(time.perf_counter() - attempt_start) * 1000.0,
                parser=backend_identity,
                backend=type(backend).__name__,
                cache_key=doc_key,
                sha=content_hash,
                size_bytes=size_bytes,
            )
            count += 1
            state_store.record(state, source_file.stem, _relative(source_file), content_hash, tier=1, ingested_by=ingested_by)
            watchdog.unwatch(source_key)

        # Parsing is over; nothing below needs the worker. (A worker left behind by
        # an abnormal exit is a daemon process and dies with the parent.)
        worker.close()
        watchdog.close()

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
                    ledger.record(
                        slug,
                        "dedup",
                        outcome=StageOutcome.SKIPPED,
                        detail=f"identical to {result.closest_match} (score={result.closest_score:.3f})",
                    )
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

    The parser has already decided that it claims the file, so this
    only decides *which bytes* the parser receives:

    - A format the table knows as text (``.md``, ``.txt``, ...) is handed
      over as UTF-8: the source itself when it already is UTF-8, otherwise
      the normalized sibling in the encoding cache. One refusal: bytes that
      sniff as binary are rejected via :func:`assert_text_source` — a
      mislabelled file (a PDF named ``.md``) must not reach the text path
      where lenient decoding turns it into a successful garbage parse
      (knox-rtmn).
    - A format the table knows as a binary container (``.pdf``, ``.docx``,
      raster) is handed over byte-wise — charset detection would reject it.
    - A suffix the table does not know (source code: ``.c``, ``.h``,
      ``.py``) is handed over byte-wise as well: the backend that claimed
      it reads the bytes and decides. The format table classifies sources,
      it never gates parsing beyond that one mislabel refusal.

    Shared by the ingest tier and the tier-1 parse stage so the two cannot
    drift apart.
    """
    extension = known_extension(source_file.suffix)
    if extension is None or not extension.is_text:
        return source_file
    # The bytes check rides the text branch only: a source the table knows as
    # text whose bytes sniff as binary is a mislabelled file (a PDF named
    # ``.md``) — refuse it instead of normalizing U+FFFD soup into the parse
    # (knox-rtmn). Binary containers never reach here.
    assert_text_source(source_file)
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


def preserve_cached_metadata(doc: ParsedDocument, prior_data: JSONDict | None) -> None:
    """Carry user-set tags/attributes from the prior entry into *doc*.

    Called before every parse-cache rewrite. The parser produces content,
    never user metadata; without this a forced re-parse (``kompend ingest
    --reparse``) or an automatic re-parse after a parser change would
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
