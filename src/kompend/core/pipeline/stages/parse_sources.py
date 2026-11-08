"""One source through Tier 1: parse, normalise, chunk, enrich.

:func:`_dispatch_single_source` is the unit the tier dispatches per file —
watchdog registration wrapped around :func:`_process_single_source`, which
beats at every phase transition; the cache halves live in
:mod:`kompend.core.pipeline.stages.parse_store`.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from kompend.core.config.rumdl import resolve_rumdl_config
from kompend.core.models import (
    ChunkTree,
    DocumentElement,
    ParsedDocument,
)
from kompend.core.pipeline.backend_capture import (
    BackendBanners,
    backend_name,
    capture_backend_output,
    log_backend_banners,
)
from kompend.core.pipeline.failure_bundle import write_failure_bundle
from kompend.core.pipeline.ingest import (
    parse_target_path,
    relativize_image_paths,
    restore_source_path,
)
from kompend.core.pipeline.reasons import reason_code_for
from kompend.core.pipeline.runtime import PipelineRuntime
from kompend.core.pipeline.stages.parse_store import (
    _chunk_cache_deps,
    _save_cache,
    _try_load_chunk_cache,
    _try_load_parsed_cache,
)
from kompend.core.pipeline.stall_watch import StallWatchdog, WatchStage
from kompend.core.reporting.ledger import StageLedger
from kompend.core.reporting.parse_quality import measure_parse_quality
from kompend.core.text.rumdl import normalize_element_markdown
from kompend.core.types import ReasonCode, StageOutcome
from kompend.core.verbosity import is_verbose
from kompend.logging import get_logger
from kompend.preprocessing.host_memory import (
    HeavyParseGuard,
    InsufficientHostMemoryError,
)
from kompend.trust import filter_sensitive

#: Shared inert watchdog (zero budget): call sites hand it around without
#: branching on ``watchdog is None`` — every beat against it is a no-op.
_INERT_WATCHDOG = StallWatchdog(0.0)

logger = get_logger(__name__)


def _dispatch_single_source(
    pipeline: PipelineRuntime,
    source_file: Path,
    dedup_similar: dict[str, dict[str, object]],
    heavy_guard: HeavyParseGuard,
    ledger: StageLedger,
    parser_identity: str | None,
    watchdog: StallWatchdog | None = None,
) -> ChunkTree | None:
    """Watch-bound unit the tier dispatches per file (knox-ty8f.8).

    Registers the source with the run's stall watchdog, runs
    :func:`_process_single_source` (which beats at every phase
    transition), and guarantees the item is forgotten even when the
    worker dies mid-file — a lost thread must not leave a ghost that
    warns later.
    """
    watch = watchdog if watchdog is not None else _INERT_WATCHDOG
    watch.watch(pipeline.rel_file(source_file), stage=WatchStage.PARSE)
    try:
        return _process_single_source(pipeline, source_file, dedup_similar, heavy_guard, ledger, parser_identity, watch)
    finally:
        watch.unwatch(pipeline.rel_file(source_file))


def _process_single_source(
    pipeline: PipelineRuntime,
    source_file: Path,
    dedup_similar: dict[str, dict[str, object]],
    heavy_guard: HeavyParseGuard,
    ledger: StageLedger,
    parser_identity: str | None,
    watchdog: StallWatchdog | None = None,
) -> ChunkTree | None:
    """Parse a single source file (cache-aware) and chunk it.

    Records the attempt's outcome per stage (parse, chunk, enrich) in
    *ledger*.

    Returns the :class:`ChunkTree` or *None* if the file was skipped
    (no parser, cache hit with no tree, or parse failure).
    """
    watch = watchdog if watchdog is not None else _INERT_WATCHDOG
    stem = source_file.stem
    use_cache = not (pipeline.config.no_cache or pipeline.config.reparse)
    content_hash = ""

    if use_cache and not pipeline.config.force:
        content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()

    parsed = _try_load_parsed_cache(pipeline, source_file, stem, content_hash, use_cache)

    if parsed is not None:
        ledger.record(
            pipeline.rel_file(source_file),
            "parse",
            outcome=StageOutcome.CACHED,
            parser=parser_identity,
            sha=content_hash or None,
        )
        ledger.record_parse_quality(
            measure_parse_quality(
                parsed,
                source=pipeline.rel_file(source_file),
                sha=content_hash,
                parser=parser_identity or "",
            )
        )
    else:
        parsed = _parse_source(pipeline, source_file, use_cache, content_hash, heavy_guard, ledger, parser_identity, watch)
        if parsed is None:
            return None
        if use_cache and not content_hash:
            content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()

    watch.beat(pipeline.rel_file(source_file), stage=WatchStage.POSTPROCESS)
    # Normalize source_file to relative POSIX on parsed doc.
    parsed.source_file = Path(pipeline.rel_file(parsed.source_file))
    for el in parsed.elements:
        if el.source_file is not None:
            el.source_file = Path(pipeline.rel_file(el.source_file))

    # Portable cache: absolute image paths -> relative to the vault. Must run
    # BEFORE chunking so asset ids (which hash figure image_ref) stay
    # consistent between the parse cache and the asset store.
    relativize_image_paths(parsed, Path(pipeline.base_directory))

    if pipeline.config.privacy_filter:
        apply_privacy_filter(parsed.elements)

    if not pipeline.chunker:
        return None

    # Attach dedup similarity warning if applicable.
    _attach_dedup_warning(parsed, stem, dedup_similar)

    # Chunk cache lookup (Tier 1).
    tree = _try_load_chunk_cache(pipeline, source_file, content_hash, use_cache)
    if tree is not None:
        ledger.record(
            pipeline.rel_file(source_file),
            "chunk",
            outcome=StageOutcome.CACHED,
            sha=content_hash or None,
        )
        return tree

    watch.beat(pipeline.rel_file(source_file), stage=WatchStage.CHUNK)
    # Heartbeat: about to chunk (the parse doc is final by now).
    watch.beat(pipeline.rel_file(source_file), stage=WatchStage.CHUNK)

    # Chunk and cache.
    chunk_start = time.perf_counter()
    tree = pipeline.chunker.chunk(parsed)
    pipeline.apply_tags_to_tree(tree, source_file)
    ledger.record(
        pipeline.rel_file(source_file),
        "chunk",
        outcome=StageOutcome.OK,
        duration_ms=(time.perf_counter() - chunk_start) * 1000.0,
        sha=content_hash or None,
    )

    watch.beat(pipeline.rel_file(source_file), stage=WatchStage.ENRICH)
    # Heartbeat: chunking is done, enrichment may follow.
    watch.beat(pipeline.rel_file(source_file), stage=WatchStage.ENRICH)
    if pipeline.enricher is not None:
        enrich_start = time.perf_counter()
        tree = pipeline.enricher.enrich_tree(tree, parsed)
        ledger.record(
            pipeline.rel_file(source_file),
            "enrich",
            outcome=StageOutcome.OK,
            duration_ms=(time.perf_counter() - enrich_start) * 1000.0,
        )

    if content_hash:
        pipeline.state_store.record(pipeline.state, stem, Path(pipeline.rel_file(source_file)), content_hash, tier=1)
        chunk_deps = _chunk_cache_deps(pipeline, content_hash)
        pipeline.cache_store.set("chunk", chunk_deps, tree.to_dict())
    return tree


def apply_privacy_filter(elements: list[DocumentElement]) -> None:
    """Apply privacy filter in-place to document elements."""
    for el in elements:
        if isinstance(el.content, str):
            el.content, _ = filter_sensitive(el.content)
        elif isinstance(el.content, list):
            el.content = [[filter_sensitive(cell)[0] for cell in row] for row in el.content]


def _parse_source(
    pipeline: PipelineRuntime,
    source_file: Path,
    use_cache: bool,
    content_hash: str,
    heavy_guard: HeavyParseGuard,
    ledger: StageLedger,
    parser_identity: str | None,
    watchdog: StallWatchdog | None = None,
) -> ParsedDocument | None:
    """Parse a source file via the parser and optionally cache the result.

    Records the parse attempt's outcome in *ledger* and beats the stall
    watchdog around the backend call.
    """
    watch = watchdog if watchdog is not None else _INERT_WATCHDOG
    if pipeline.parser is None:
        return None
    attempt_start = time.perf_counter()
    selected: Any = None
    parse_target: Path | None = None
    try:
        selected = pipeline.parser.route(source_file)
        if selected is None:
            logger.warning("No parser backend for %s, skipping", source_file, extra={"reason_code": ReasonCode.NO_PARSER})
            ledger.record(
                pipeline.rel_file(source_file),
                "parse",
                outcome=StageOutcome.SKIPPED,
                duration_ms=(time.perf_counter() - attempt_start) * 1000.0,
                reason_code=ReasonCode.NO_PARSER,
                parser=parser_identity,
            )
            return None
        # The parser has already picked a backend: the format table only
        # decides which bytes that backend receives (UTF-8 for known text
        # formats, the raw file otherwise). It never gates parsing.
        parse_target = parse_target_path(source_file, pipeline.cache_directory())
        watch.beat(pipeline.rel_file(source_file), stage=WatchStage.PARSE)
        banners = BackendBanners(source=pipeline.rel_file(source_file), backend=backend_name(selected))
        try:
            with capture_backend_output(banners), heavy_guard.guard(source_file):
                parsed = selected.parse(parse_target)
        finally:
            log_backend_banners(banners)
        _ = len(parsed.elements)
    except InsufficientHostMemoryError as exc:
        logger.error("Skipped %s: %s", source_file, exc, extra={"reason_code": ReasonCode.BACKEND_UNAVAILABLE})
        write_failure_bundle(
            base_dir=pipeline.base_directory,
            source=pipeline.rel_file(source_file),
            source_file=source_file,
            error=exc,
            reason_code=ReasonCode.BACKEND_UNAVAILABLE,
            stage="parse",
            parse_target=parse_target,
            parser=parser_identity,
            backend=type(selected).__name__ if selected is not None else None,
            sha=content_hash,
        )
        ledger.record(
            pipeline.rel_file(source_file),
            "parse",
            outcome=StageOutcome.SKIPPED,
            duration_ms=(time.perf_counter() - attempt_start) * 1000.0,
            reason_code=ReasonCode.BACKEND_UNAVAILABLE,
            detail=str(exc),
            parser=parser_identity,
            backend=type(selected).__name__ if selected is not None else None,
        )
        return None
    except Exception as exc:
        reason = reason_code_for(exc)
        logger.error(
            "Failed to parse %s: %s",
            source_file,
            exc,
            exc_info=is_verbose(),
            extra={"reason_code": reason},
        )
        write_failure_bundle(
            base_dir=pipeline.base_directory,
            source=pipeline.rel_file(source_file),
            source_file=source_file,
            error=exc,
            reason_code=reason,
            stage="parse",
            parse_target=parse_target,
            parser=parser_identity,
            backend=type(selected).__name__ if selected is not None else None,
            sha=content_hash,
        )
        ledger.record(
            pipeline.rel_file(source_file),
            "parse",
            outcome=StageOutcome.FAILED,
            duration_ms=(time.perf_counter() - attempt_start) * 1000.0,
            reason_code=reason,
            detail=str(exc),
            parser=parser_identity,
            backend=type(selected).__name__ if selected is not None else None,
        )
        return None
    ledger.record(
        pipeline.rel_file(source_file),
        "parse",
        outcome=StageOutcome.OK,
        duration_ms=(time.perf_counter() - attempt_start) * 1000.0,
        parser=parser_identity,
        backend=type(selected).__name__,
        sha=content_hash or None,
    )
    watch.beat(pipeline.rel_file(source_file), stage=WatchStage.POSTPROCESS)
    # The parser saw the encoding sibling (if any); citations and the cache
    # entry must reference the real source, not the cache copy.
    restore_source_path(parsed, parse_target=parse_target, source_file=source_file)
    # Portable cache: absolute image paths -> relative to the vault, BEFORE the
    # cache write so cache and in-memory doc agree (asset ids hash figure
    # image_ref; the same doc is chunked after this function returns).
    relativize_image_paths(parsed, Path(pipeline.base_directory))
    # Canonical markdown at the part level (one rumdl call per document), so
    # backend formatting differences are levelled before anything downstream
    # reads the content. Content-only: page/reading-order/metadata are kept.
    normalize_element_markdown(
        parsed.elements,
        rumdl_config=resolve_rumdl_config(Path(pipeline.base_directory), pipeline.config.output.rumdl_config),
        flavor=pipeline.config.output.flavor,
    )
    ledger.record_parse_quality(
        measure_parse_quality(
            parsed,
            source=pipeline.rel_file(source_file),
            sha=content_hash,
            parser=parser_identity or "",
            backend=type(selected).__name__,
        )
    )
    if use_cache:
        _save_cache(pipeline, source_file, parsed)
    return parsed


def _attach_dedup_warning(parsed: ParsedDocument, stem: str, dedup_similar: dict[str, dict[str, object]]) -> None:
    """Attach a dedup similarity warning to the parsed document's metadata."""
    similar_entry = dedup_similar.get(stem)
    if not similar_entry:
        return
    warning = (
        f"> [!warning] Possible duplicate\n"
        f"> This document is similar to **{similar_entry.get('closest_match', 'unknown')}** "
        f"> (score={similar_entry.get('closest_score', 0):.2f}).\n"
    )
    if not parsed.metadata:
        parsed.metadata = {}
    parsed.metadata["_dedup_warning"] = warning
