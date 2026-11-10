"""One source through Tier 1: parse, normalise, split, chunk, enrich.

:func:`_dispatch_single_source` is the unit the tier dispatches per file —
watchdog registration wrapped around :func:`_process_single_source`, which
beats at every phase transition; the cache halves live in
:mod:`kompend.core.pipeline.stages.parse_store`.

Both return *one tree per document*: a packet that
:class:`~kompend.preprocessing.splitting.SourceSplitter` cuts along page ranges
yields one tree per child source. With no splitter injected — the default —
the list holds exactly the one tree today's code returned, in the same order
and with the same ledger rows, so the off path is unchanged.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from kompend.core.config.rumdl import resolve_rumdl_config
from kompend.core.identity import source_key
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
from kompend.preprocessing.splitting import (
    LEDGER_DETAIL_CHARS,
    SPLIT_FIELDS,
    SourceSplitter,
    SplitDecision,
    SplitReason,
    partition_document,
)
from kompend.preprocessing.splitting._provenance import attachment_owner_stem, parent_source_of
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
    splitting: SourceSplitter | None = None,
) -> list[ChunkTree]:
    """Watch-bound unit the tier dispatches per file (knox-ty8f.8).

    Registers the source with the run's stall watchdog, runs
    :func:`_process_single_source` (which beats at every phase
    transition), and guarantees the item is forgotten even when the
    worker dies mid-file — a lost thread must not leave a ghost that
    warns later.

    Returns one tree per produced document (``[]`` when the source was
    skipped) — see :func:`_process_single_source`.
    """
    watch = watchdog if watchdog is not None else _INERT_WATCHDOG
    watch.watch(pipeline.rel_file(source_file), stage=WatchStage.PARSE)
    try:
        return _process_single_source(pipeline, source_file, dedup_similar, heavy_guard, ledger, parser_identity, watch, splitting)
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
    splitting: SourceSplitter | None = None,
) -> list[ChunkTree]:
    """Parse a single source file (cache-aware), split it, and chunk each document.

    Records the attempt's outcome per stage (parse, split, chunk, enrich) in
    *ledger*.

    When *splitting* is ``None`` (the default) the source is one document and
    the returned list holds the one tree the previous single-tree return gave
    (``[]`` when the file was skipped: no parser, cache hit with no tree, or
    parse failure). When a splitter is injected and its plan applies, the
    parsed packet becomes N virtual child sources and the list holds one tree
    per child — the parent itself is not chunked.
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
            return []
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
        return []

    # Attach dedup similarity warning if applicable.
    _attach_dedup_warning(parsed, stem, dedup_similar)

    watch.beat(pipeline.rel_file(source_file), stage=WatchStage.POSTPROCESS)
    documents = _split_documents(pipeline, parsed, splitting, ledger)

    trees: list[ChunkTree] = []
    for document in documents:
        # A child's path is synthetic (``<parent>#p005-009.<suffix>``) and does
        # not exist on disk, so directory tags come from the file it was cut
        # out of — which is *source_file* for the parent itself as well.
        tree = _chunk_document(
            pipeline,
            document=document,
            tag_source=source_file,
            content_hash=content_hash,
            use_cache=use_cache,
            ledger=ledger,
            watch=watch,
            # The parent keeps the plain dependency set: its cache key stays
            # byte-identical. A child adds a discriminator because siblings
            # share the parent's content hash and would otherwise collide.
            identity="" if document is parsed else _document_identity(document, content_hash),
        )
        if tree is not None:
            trees.append(tree)
    return trees


def _split_documents(
    pipeline: PipelineRuntime,
    parsed: ParsedDocument,
    splitting: SourceSplitter | None,
    ledger: StageLedger,
) -> list[ParsedDocument]:
    """The documents *parsed* becomes: itself, or one child per planned segment.

    With no splitter injected this is exactly ``[parsed]`` and writes no ledger
    row — the off path stays byte-identical. When a splitter is injected every
    attempt is recorded, whether it cut or declined: silent splitting is
    forbidden, and so is a silent decline.
    """
    if splitting is None:
        return [parsed]

    source = pipeline.rel_file(parsed.source_file)
    decision = splitting.decide(parsed)
    if not decision.applies:
        _record_split_decline(ledger, source, decision)
        return [parsed]

    try:
        children = partition_document(
            parsed,
            decision.segments,
            parent_source=parsed.source_file,
            decided_by=decision.decided_by,
            rules_sha256=decision.rules_sha256,
        )
    except ValueError as exc:
        # partition_document refuses to lose an element; that is a decline, not
        # a crash — the packet stays whole and the run log says why.
        _record_split_decline(
            ledger,
            source,
            SplitDecision(reason=SplitReason.PLAN_INVALID, detail=str(exc), decided_by=decision.decided_by),
        )
        return [parsed]

    ledger.record(source, "split", outcome=StageOutcome.OK, detail=decision.ledger_detail())
    logger.info(
        "Split %s — %s (attachments stay with %s)",
        source,
        decision.summary(),
        attachment_owner_stem(parsed.source_file, parsed.metadata) or source,
    )
    return children


def _record_split_decline(ledger: StageLedger, source: str, decision: SplitDecision) -> None:
    """Record a split that did not happen, with the reason it did not.

    A judge failure or an unusable plan marks the attempt ``degraded`` (the
    deterministic path kept the source whole) and carries
    :data:`ReasonCode.JUDGE_DEGRADED`; a normal decline — too few pages, no
    boundary, unpaged, one page too large — is ``skipped`` with the reason in
    the detail. (Splitting has no dedicated ``ReasonCode`` yet; adding one
    belongs in ``core.types``, which this change does not own.)
    """
    detail = f"{decision.reason.value}: {decision.detail}" if decision.detail else decision.reason.value
    if len(detail) > LEDGER_DETAIL_CHARS:
        detail = decision.ledger_detail()
    if decision.reason in {SplitReason.JUDGE_FAILED, SplitReason.PLAN_INVALID}:
        logger.warning("Split declined for %s — %s", source, detail, extra={"reason_code": ReasonCode.JUDGE_DEGRADED})
        ledger.record(source, "split", outcome=StageOutcome.DEGRADED, reason_code=ReasonCode.JUDGE_DEGRADED, detail=detail)
        return
    logger.debug("Split declined for %s — %s", source, detail)
    ledger.record(source, "split", outcome=StageOutcome.SKIPPED, detail=detail)


def _document_identity(document: ParsedDocument, content_hash: str) -> str:
    """Chunk-cache discriminator for a split child: its own source identity.

    ``<stem>-<hash8>`` (``kompend.core.identity.source_key``) is the rule every
    other cache layer already uses; children share the parent's content hash, so
    the stem half is what keeps siblings apart.
    """
    return source_key(document.source_file.stem, content_hash)


def _chunk_document(
    pipeline: PipelineRuntime,
    *,
    document: ParsedDocument,
    tag_source: Path,
    content_hash: str,
    use_cache: bool,
    ledger: StageLedger,
    watch: StallWatchdog,
    identity: str,
) -> ChunkTree | None:
    """Chunk (or load from cache) and enrich one document; ``None`` on a skip.

    For the unsplit source (``identity == ""``) this is the pre-split code path
    unchanged, including :func:`_try_load_chunk_cache` and the plain dependency
    set. A split child takes :func:`_load_child_tree`, which carries the extra
    discriminator in the same dependency set.
    """
    source_label = pipeline.rel_file(document.source_file)
    deps = _chunk_deps(pipeline, content_hash, identity=identity)

    # Chunk cache lookup (Tier 1).
    if identity:
        tree = _load_child_tree(pipeline, document, content_hash, use_cache, deps, tag_source)
    else:
        tree = _try_load_chunk_cache(pipeline, document.source_file, content_hash, use_cache)
    if tree is not None:
        ledger.record(source_label, "chunk", outcome=StageOutcome.CACHED, sha=content_hash or None)
        return tree

    # Heartbeat: about to chunk (the parse doc is final by now). The unsplit
    # source beats twice here, exactly as it did before splitting existed; the
    # second beat is kept so the off path stays byte-identical.
    watch.beat(pipeline.rel_file(tag_source), stage=WatchStage.CHUNK)
    if not identity:
        watch.beat(pipeline.rel_file(tag_source), stage=WatchStage.CHUNK)
    if pipeline.chunker is None:
        return None

    # Chunk and cache.
    chunk_start = time.perf_counter()
    tree = pipeline.chunker.chunk(document)
    _stamp_split_provenance(tree, document)
    pipeline.apply_tags_to_tree(tree, tag_source)
    ledger.record(
        source_label,
        "chunk",
        outcome=StageOutcome.OK,
        duration_ms=(time.perf_counter() - chunk_start) * 1000.0,
        sha=content_hash or None,
    )

    # Heartbeat: chunking is done, enrichment may follow — same double beat on
    # the unsplit path as before (see the chunk heartbeat above).
    watch.beat(pipeline.rel_file(tag_source), stage=WatchStage.ENRICH)
    if not identity:
        watch.beat(pipeline.rel_file(tag_source), stage=WatchStage.ENRICH)
    if pipeline.enricher is not None:
        enrich_start = time.perf_counter()
        tree = pipeline.enricher.enrich_tree(tree, document)
        ledger.record(
            source_label,
            "enrich",
            outcome=StageOutcome.OK,
            duration_ms=(time.perf_counter() - enrich_start) * 1000.0,
        )

    if content_hash:
        pipeline.state_store.record(pipeline.state, document.source_file.stem, Path(source_label), content_hash, tier=1)
        pipeline.cache_store.set("chunk", deps, tree.to_dict())
    return tree


def _stamp_split_provenance(tree: ChunkTree, document: ParsedDocument) -> None:
    """Copy a split child's provenance onto every leaf chunk of *tree*.

    A child document carries the split keys in ``ParsedDocument.metadata``, but
    a chunk only inherits its *element*'s metadata — so a leaf of a child tree
    would reach the compile and export stages knowing nothing about the packet
    it came from. That matters twice: attachment lookup keys off the ingested
    source's stem (the parent, since the child was never ingested on its own),
    and a page's provenance should be traceable back to a file that exists.

    The unsplit source has no such keys, so this is a no-op for it.
    """
    stamped = {str(key.value): value for key in SPLIT_FIELDS if (value := document.metadata.get(key.value))}
    if not stamped:
        return
    parent = parent_source_of(document.metadata)
    if parent is not None:
        logger.debug("Chunked %s — cut out of %s", document.source_file, parent)
    for node in tree.leaves():
        if node.chunk is None:
            continue
        node.chunk.metadata.update(stamped)


def _chunk_deps(pipeline: PipelineRuntime, content_hash: str, *, identity: str) -> dict[str, str]:
    """Chunk-cache dependencies, plus a per-document discriminator when split.

    The base set is :func:`~kompend.core.pipeline.stages.parse_store._chunk_cache_deps`
    and is not re-derived here. The split child's extra ``source_identity`` key
    belongs in that function long-term (it is the chunk-cache identity rule's
    home); it lives here only because this change does not own ``parse_store``.
    """
    deps = _chunk_cache_deps(pipeline, content_hash)
    if identity:
        deps["source_identity"] = identity
    return deps


def _load_child_tree(
    pipeline: PipelineRuntime,
    document: ParsedDocument,
    content_hash: str,
    use_cache: bool,
    deps: dict[str, str],
    tag_source: Path,
) -> ChunkTree | None:
    """Chunk-cache hit for a split child (the discriminating twin of the store helper).

    Same behaviour as :func:`_try_load_chunk_cache` — state row, rebound
    ``source_file``, tags — over the child's own dependency set so two children
    of one packet never share an entry.
    """
    if not (use_cache and content_hash):
        return None
    cached = pipeline.cache_store.get_dict("chunk", deps)
    if cached is None:
        return None
    pipeline.state_store.record(pipeline.state, document.source_file.stem, Path(pipeline.rel_file(document.source_file)), content_hash, tier=1)
    tree = ChunkTree.from_dict(cached)
    tree.source_file = document.source_file
    pipeline.apply_tags_to_tree(tree, tag_source)
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
