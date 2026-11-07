"""One source through Tier 1: parse, normalise, chunk, enrich.

:func:`_process_single_source` is the unit the tier dispatches per file;
the cache halves it consults live in :mod:`kompend.core.pipeline.stages.parse_store`.
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
from kompend.core.reporting.ledger import StageLedger
from kompend.core.text.rumdl import normalize_element_markdown
from kompend.core.types import ReasonCode, StageOutcome
from kompend.core.verbosity import is_verbose
from kompend.logging import get_logger
from kompend.preprocessing.host_memory import (
    HeavyParseGuard,
    InsufficientHostMemoryError,
)
from kompend.trust import filter_sensitive

logger = get_logger(__name__)


def _process_single_source(
    pipeline: PipelineRuntime,
    source_file: Path,
    dedup_similar: dict[str, dict[str, object]],
    heavy_guard: HeavyParseGuard,
    ledger: StageLedger,
    parser_identity: str | None,
) -> ChunkTree | None:
    """Parse a single source file (cache-aware) and chunk it.

    Records the attempt's outcome per stage (parse, chunk, enrich) in
    *ledger*.

    Returns the :class:`ChunkTree` or *None* if the file was skipped
    (no parser, cache hit with no tree, or parse failure).
    """
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
    else:
        parsed = _parse_source(pipeline, source_file, use_cache, content_hash, heavy_guard, ledger, parser_identity)
        if parsed is None:
            return None
        if use_cache and not content_hash:
            content_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()

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
) -> ParsedDocument | None:
    """Parse a source file via the parser and optionally cache the result.

    Records the parse attempt's outcome in *ledger*.
    """
    if pipeline.parser is None:
        return None
    attempt_start = time.perf_counter()
    selected: Any = None
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
        with heavy_guard.guard(source_file):
            parsed = selected.parse(parse_target)
        _ = len(parsed.elements)
    except InsufficientHostMemoryError as exc:
        logger.error("Skipped %s: %s", source_file, exc, extra={"reason_code": ReasonCode.BACKEND_UNAVAILABLE})
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
