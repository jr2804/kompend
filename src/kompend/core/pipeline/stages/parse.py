"""Tier 1 (parse + chunk) orchestration helpers."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextvars import copy_context
from pathlib import Path
from typing import Any, cast

from kompend.agents.failure import UnitFailureLog
from kompend.agents.vlm import VlmAgent
from kompend.core.config.rumdl import resolve_rumdl_config
from kompend.core.models import (
    ChunkTree,
    DocumentElement,
    ElementType,
    MetadataField,
    ParsedDocument,
    WikiField,
)
from kompend.core.models.attachments import AttachmentManifest, adopt_orphan_attachments
from kompend.core.models.elements import element_text
from kompend.core.paths import VaultPaths
from kompend.core.pipeline.attachments import (
    ingest_attachment,
    iter_ingestible_attachments,
)
from kompend.core.pipeline.config import PipelineConfig
from kompend.core.pipeline.ingest import (
    PARSE_CACHE_PARSER_KEY,
    parse_target_path,
    preserve_cached_metadata,
    relativize_image_paths,
    restore_source_path,
)
from kompend.core.pipeline.parallelism import describe_parallelism
from kompend.core.pipeline.parse_cache import (
    cache_key,
    content_hash_prefix,
    parse_cache_is_current,
    read_document,
    source_key,
    write_document,
)
from kompend.core.pipeline.reasons import reason_code_for
from kompend.core.pipeline.runtime import PipelineRuntime
from kompend.core.reporting.ledger import StageLedger, stage_ledger
from kompend.core.store import vault_db_path
from kompend.core.text.rumdl import normalize_element_markdown
from kompend.core.types import ReasonCode, StageOutcome
from kompend.core.verbosity import is_verbose
from kompend.llm.models import resolve_vlm_model
from kompend.llm.user_providers import effective_workers
from kompend.logging import get_logger
from kompend.preprocessing.dedup import DedupStore
from kompend.preprocessing.host_memory import (
    MEGABYTE,
    HeavyParseGuard,
    InsufficientHostMemoryError,
)
from kompend.preprocessing.parsers.base import scrub_document
from kompend.resolution.base import BaseResolver
from kompend.resolution.chain import ResolverChain
from kompend.resolution.vlm_enricher import VLMEnricher
from kompend.trust import filter_sensitive
from kompend.types import JSONDict

logger = get_logger(__name__)


def parse_and_chunk(pipeline: PipelineRuntime) -> list[ChunkTree]:
    """Parse documents and chunk them, with optional caching and parallelism.

    When tier state is available, sources that already match the
    persisted hash and tier are skipped to keep the per-tier
    execution incremental.

    When content-level duplicate detection has run during a prior ingest,
    the DedupStore is checked. Sources flagged as ``identical`` are
    silently skipped. Sources flagged as ``similar`` get a duplication
    warning attached to their metadata.
    """
    source_files = list(pipeline.iter_sources())
    if not source_files:
        return []

    pipeline.emit("parse", 0, len(source_files), "starting")

    # Auto-adopt orphaned attachments (files under attachments/<slug>/ with
    # no manifest entry — recovery after manifest loss, or files dropped in
    # manually). Flags derive from extension classes + global embed config.
    try:
        adopted = adopt_orphan_attachments(
            Path(pipeline.base_directory),
            ingest_adopted=pipeline.config.attachments_ingest_adopted,
        )
        if adopted:
            logger.info("Adopted %d orphaned attachment(s): %s", len(adopted), [a.path for a in adopted])
    except OSError as exc:
        logger.warning("Attachment adoption failed — continuing: %s", exc)

    dedup_skip, dedup_similar = _load_dedup_state(pipeline)
    if dedup_skip:
        logger.info("Dedup: %d source(s) flagged as identical — will skip", len(dedup_skip))
        with stage_ledger(vault_db_path(pipeline.cache_directory())) as ledger:
            for f in source_files:
                if f.stem in dedup_skip:
                    ledger.record(
                        pipeline.rel_file(f),
                        "parse",
                        outcome=StageOutcome.SKIPPED,
                        detail="dedup: identical to another source",
                    )
        source_files = [f for f in source_files if f.stem not in dedup_skip]
        if not source_files:
            pipeline.emit("parse", 0, 0, "all skipped by dedup")
            return []

    workers = max(1, pipeline.config.workers)
    # Clip against the LLM provider's client-side connection budget
    # ([providers.<name>].max_connections — overridable for integrated
    # providers like ollama). Silent throttling is impossible: a clip logs
    # a warning naming both values.
    workers = effective_workers(pipeline.config.llm.default_model or "", workers)

    # One guard per run: serializes heavy parses (a PDF/office conversion
    # costs several GB of commit charge — N workers cost N times that) and
    # refuses one outright when the host cannot back it. A refused source
    # fails loudly per-file; the run continues with the rest.
    heavy_guard = HeavyParseGuard(
        min_free_commit_bytes=pipeline.config.parse_min_free_commit_mb * MEGABYTE,
        max_concurrent=pipeline.config.max_concurrent_heavy_parses,
    )

    trees: list[ChunkTree] = []
    # The ledger records one row per (source, stage, attempt) for this run;
    # rows correlate with the CLI run via the handle context. Opened here,
    # closed when the stage function returns — per-row commits keep rows
    # even if the run dies inside.
    with stage_ledger(vault_db_path(pipeline.cache_directory())) as ledger:
        parser_identity = pipeline.parser.backend_identity() if pipeline.parser is not None else None
        if workers > 1 and len(source_files) > 1:
            # Say which kind of interpreter this is: a thread pool only runs
            # Python code in parallel on a free-threaded build.
            logger.info("%s", describe_parallelism(workers))
            completed = 0
            # Explicit executor (not a ``with`` block): on a graceful cancel the
            # emit below raises KompendCancelled, and shutdown(cancel_futures=True)
            # then drops queued futures instead of parsing the whole corpus.
            # In-flight workers still finish (bounded by ``workers``) so caches
            # stay consistent.
            executor = ThreadPoolExecutor(max_workers=workers)
            # Each submission carries a copy of the caller's context: the verbosity
            # switch lives in a ContextVar, and a fresh thread would read its
            # default (False) and lose ``--verbose`` for every worker-thread
            # traceback. ``copy_context()`` per iteration keeps each Context
            # single-use, which ``Context.run`` requires. The ledger (lock-guarded)
            # and the run-id context travel with it.
            futures: dict[Future[ChunkTree | None], Path] = {
                executor.submit(_context_bound_call(_process_single_source, pipeline, sf, dedup_similar, heavy_guard, ledger, parser_identity)): sf
                for sf in source_files
            }
            try:
                for future in as_completed(futures):
                    sf = futures[future]
                    try:
                        tree = future.result()
                        if tree is not None:
                            trees.append(tree)
                    except Exception as exc:
                        logger.error(
                            "Worker failed for %s: %s",
                            sf,
                            exc,
                            exc_info=is_verbose(),
                            extra={"reason_code": reason_code_for(exc)},
                        )
                    finally:
                        completed += 1
                        pipeline.emit("parse", completed, len(source_files), sf.name)
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        else:
            for idx, sf in enumerate(source_files):
                tree = _process_single_source(pipeline, sf, dedup_similar, heavy_guard, ledger, parser_identity)
                if tree is not None:
                    trees.append(tree)
                pipeline.emit("parse", idx + 1, len(source_files), sf.name)

        pipeline.emit("parse", len(source_files), len(source_files), "done")

        # ── Process ingestible attachments ──────────────────────────────
        att_trees = _process_attachments(pipeline, ledger)
        if att_trees:
            trees.extend(att_trees)

    if pipeline.summarizer is not None:
        # Summarization runs one LLM session per tree and can take hours for
        # large vaults — emit the summary sub-phase so the bar tracks it
        # instead of pinning at "Parsed N file(s) 100%".
        pipeline.emit("summary", 0, len(trees), "starting")
        for idx, tree in enumerate(trees):
            pipeline.emit("summary", idx + 1, len(trees), tree_label(tree))
            pipeline.summarizer.summarize(tree)
        pipeline.emit("summary", len(trees), len(trees), "done")

    return trees


def tree_label(tree: ChunkTree) -> str:
    """Short label identifying a tree in progress output."""
    source = tree.source_file
    return source.name if source is not None else tree.root_id


def _context_bound_call(fn: Callable[..., ChunkTree | None], *args: Any) -> Callable[[], ChunkTree | None]:
    """Bind *fn* to a copy of the caller's context, for pool submission.

    Called in the submitting thread on purpose: ``copy_context()`` inside the
    worker would copy the worker's own (empty) context. ``Context.run`` accepts
    one context object per call, so each submission gets its own copy.
    """
    context = copy_context()
    return lambda: context.run(fn, *args)


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


def _try_load_parsed_cache(pipeline: PipelineRuntime, source_file: Path, stem: str, content_hash: str, use_cache: bool) -> ParsedDocument | None:
    """Try to load a parsed document from cache. Returns *None* on miss."""
    if not use_cache:
        return None
    parsed = _load_cache(pipeline, source_file)
    if parsed is not None:
        return parsed
    if pipeline.config.force:
        return None
    cached_entry = pipeline.state.entries.get(source_key(stem, content_hash))
    if cached_entry and cached_entry.content_hash == content_hash and cached_entry.current_tier >= 1:
        return _load_cache(pipeline, source_file)
    return None


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


def _try_load_chunk_cache(pipeline: PipelineRuntime, source_file: Path, content_hash: str, use_cache: bool) -> ChunkTree | None:
    """Try to load a chunked tree from cache. Returns *None* on miss.

    The tree is rebound to *source_file*: the chunk-cache identity is the
    content hash, so a source may be moved or regrouped under ``sources/``
    and still hit; the path it carries must be the one this run is compiling.
    """
    if not (use_cache and content_hash):
        return None
    chunk_deps = _chunk_cache_deps(pipeline, content_hash)
    cached = pipeline.cache_store.get_dict("chunk", chunk_deps)
    if cached is None:
        return None
    stem = source_file.stem
    pipeline.state_store.record(pipeline.state, stem, Path(pipeline.rel_file(source_file)), content_hash, tier=1)
    tree = ChunkTree.from_dict(cached)
    tree.source_file = source_file
    pipeline.apply_tags_to_tree(tree, source_file)
    return tree


def _chunk_cache_deps(pipeline: PipelineRuntime, content_hash: str) -> dict[str, str]:
    """Cache identity for a chunked tree.

    ``source`` + chunker size identify the chunking.
    """
    return {
        "source": content_hash,
        "chunk_effective_max": str(getattr(pipeline.chunker, "_effective_max", 0)),
        "contextual": "1" if (pipeline.config.search.contextual.enabled and pipeline.enricher is not None) else "0",
    }


def _load_cache(pipeline: PipelineRuntime, source_file: Path) -> ParsedDocument | None:
    """Load the cached ParsedDocument for *source_file*, or *None* on a miss.

    The cache key is ``<stem>-<hash8>``, so the entry is content-addressed and
    a source may be moved or regrouped under ``sources/`` without losing it.
    The stored ``source_file`` is therefore only a historical note: a hit is
    rebound to *source_file* (the path this run looked the entry up for) so
    citations, tags and the state store never resurrect a path that no longer
    exists. See ``tests/unit/test_ingest_reparse.py``.

    Content alone does not make an entry current: the parser identity decides
    whether this run would produce the same document. A changed backend set, or
    a changed routing input (judge, preference, backend options), makes the
    stored entry a miss — otherwise a vault would keep serving parses that the
    configuration no longer asks for.
    """
    db_path = vault_db_path(Path(pipeline.base_directory))
    content_hash = content_hash_prefix(source_file.read_bytes())
    key = cache_key(source_file, content_hash)
    data = read_document(db_path, key)
    if data is None:
        return None
    if not parse_cache_is_current(data, parse_identity(pipeline)):
        logger.info("Re-parsing %s: cached parse is not from the current parser", source_file.name)
        return None
    try:
        elements = [
            DocumentElement(
                type=ElementType(el["type"]),
                content=el["content"],
                page=el.get("page"),
                reading_order=el.get(MetadataField.READING_ORDER),
                heading_level=el.get("heading_level"),
                metadata=el.get("metadata", {}),
            )
            for el in cast("list[dict[str, Any]]", data.get("elements") or [])
        ]
        # scrub_document: caches written before surrogate scrubbing carry
        # lone \\udXXX escapes that json.loads restores as surrogate chars.
        return scrub_document(
            ParsedDocument(
                source_file=source_file,
                elements=elements,
                title=cast("str", data.get("title", "")),
                author=cast("str", data.get("author", "")),
                language=cast("str", data.get("language", "en")),
                page_count=cast("int", data.get("page_count", 0)),
                metadata=cast("dict[str, Any]", data.get(WikiField.METADATA) or {}),
            )
        )
    except (KeyError, ValueError, TypeError) as exc:
        # Corrupt entry = miss: re-parse. Warn so it's visible.
        logger.warning("Corrupt parse cache for %s — reparsing: %s", key, exc, exc_info=is_verbose(), extra={"reason_code": ReasonCode.CORRUPT_CACHE})
        return None


def _save_cache(pipeline: PipelineRuntime, source_file: Path, doc: ParsedDocument) -> None:
    """Write a ParsedDocument to the store, stamped with the parser identity."""
    db_path = vault_db_path(Path(pipeline.base_directory))
    content_hash = content_hash_prefix(source_file.read_bytes())
    key = cache_key(source_file, content_hash)
    try:
        # Preserve user-set tags/attributes across the overwrite.
        preserve_cached_metadata(doc, read_document(db_path, key))
        data = {
            str(MetadataField.SOURCE_FILE): pipeline.rel_file(doc.source_file),
            PARSE_CACHE_PARSER_KEY: parse_identity(pipeline),
            "title": doc.title,
            "author": doc.author,
            "language": doc.language,
            "page_count": doc.page_count,
            "metadata": doc.metadata,
            "elements": [
                {
                    "type": str(el.type),
                    # element_text, not str(): bytes would land in the cache (and
                    # every later read of it) as a repr blob.
                    "content": el.content if isinstance(el.content, list) else element_text(el.content),
                    "page": el.page,
                    MetadataField.READING_ORDER: el.reading_order,
                    "heading_level": el.heading_level,
                    "metadata": el.metadata,
                }
                for el in doc.elements
            ],
        }
        write_document(db_path, key, cast("JSONDict", data))
    except Exception as exc:
        logger.debug("Cache write failed for %s — ignoring: %s", key, exc, exc_info=is_verbose())


def parse_identity(pipeline: PipelineRuntime) -> str:
    """The identity of the parse this run would produce.

    ``DocumentParser.backend_identity()`` composes both halves: the backends that
    can read here, and the routing inputs that change what they emit (judge tier,
    preference, per-backend options, upstream version). The parse cache stores it
    and compares it on read, so a configuration change re-parses instead of being
    silently ignored for every document already in the cache.
    """
    parser = pipeline.parser
    return parser.backend_identity() if parser is not None else ""


def _load_dedup_state(pipeline: PipelineRuntime) -> tuple[set[str], dict[str, dict[str, object]]]:
    """Best-effort load of dedup store.

    Returns ``(dedup_skip, dedup_similar)``.
    """
    dedup_skip: set[str] = set()
    dedup_similar: dict[str, dict[str, object]] = {}
    try:
        ds = DedupStore(pipeline.dedup_store_file())
        ds.load()
        if ds.is_fresh():
            for slug in ds.slugs:
                entry = ds.get(slug, {})
                if entry.get("verdict") == "identical":
                    dedup_skip.add(slug)
                elif entry.get("verdict") == "similar":
                    dedup_similar[slug] = entry
    except Exception as exc:
        logger.debug("Dedup store not available — skipping dedup check: %s", exc, exc_info=is_verbose())
    return dedup_skip, dedup_similar


def _process_attachments(pipeline: PipelineRuntime, ledger: StageLedger) -> list[ChunkTree]:
    """Process ingestible attachments and return their ChunkTrees.

    Attachments are parsed via the same parser as sources.
    Image attachments (PNG/JPG) are captioned via VLM if configured.
    Each resulting tree carries ``source_asset=True`` and ``parent_source``
    metadata so the rest of the pipeline can distinguish them.
    """
    vp = VaultPaths(pipeline.base_directory)
    manifest = AttachmentManifest(vp.attachments_manifest_file)
    if not manifest.exists():
        return []

    # Global gate: ``attachments.ingest_adopted`` ANDs with the per-attachment
    # manifest flag. When the gate is off, nothing is ingested regardless of
    # how many attachments were explicitly marked ingestible.
    if not pipeline.config.attachments_ingest_adopted:
        logger.info("Attachment ingestion disabled (attachments.ingest_adopted=false) — skipping %d manifest entry(s)", len(manifest.read_all()))
        return []

    ingestible = iter_ingestible_attachments(vp)
    if not ingestible:
        return []

    parser = pipeline.parser
    if parser is None:
        logger.warning("No parser available — skipping attachment ingestion", extra={"reason_code": ReasonCode.NO_PARSER})
        return []
    vlm_captioner = None
    vlm_model = resolve_vlm_model(pipeline.config.llm.default_model, pipeline.config.llm.vlm_model)
    if vlm_model:
        try:
            vlm_captioner = VlmAgent(
                vlm_model,
                timeout=pipeline.config.llm.timeout,
                llm=pipeline.config.llm,
            )
        except Exception as exc:
            logger.debug("VLM captioner not available for attachments: %s", exc, exc_info=is_verbose())

    trees: list[ChunkTree] = []
    pipeline.emit("parse", 0, len(ingestible), "starting")
    for idx, att in enumerate(ingestible):
        pipeline.emit("parse", idx + 1, len(ingestible), f"asset: {att.filename}")

        try:
            doc = ingest_attachment(vp, att, parser, vlm_captioner=vlm_captioner, vlm_model=vlm_model, ledger=ledger)
            if doc is None:
                continue

            if not pipeline.chunker:
                continue

            chunk_start = time.perf_counter()
            tree = pipeline.chunker.chunk(doc)
            ledger.record(
                att.path,
                "chunk",
                outcome=StageOutcome.OK,
                duration_ms=(time.perf_counter() - chunk_start) * 1000.0,
            )

            # Stamp source-asset metadata on all leaves so downstream
            # resolve/index stages can distinguish attachment content.
            for node in tree.leaves():
                if node.chunk is None:
                    continue
                node.chunk.metadata["source_asset"] = True
                node.chunk.metadata["parent_source"] = att.source
                node.chunk.metadata["asset_file"] = att.path

            if pipeline.enricher is not None:
                enrich_start = time.perf_counter()
                tree = pipeline.enricher.enrich_tree(tree, doc)
                ledger.record(
                    att.path,
                    "enrich",
                    outcome=StageOutcome.OK,
                    duration_ms=(time.perf_counter() - enrich_start) * 1000.0,
                )

            trees.append(tree)
            logger.info("Processed attachment: %s → %d chunk(s)", att.filename, len(list(tree.leaves())))

        except Exception as exc:
            logger.debug("Failed to process attachment %s: %s", att.filename, exc, exc_info=is_verbose())
            ledger.record(
                att.path,
                "attachment",
                outcome=StageOutcome.FAILED,
                reason_code=reason_code_for(exc),
                detail=str(exc),
            )

    if trees:
        logger.info("Processed %d ingestible attachment(s)", len(trees))

    return trees


def build_resolver_chain(config: PipelineConfig) -> BaseResolver:
    """Construct the resolver chain.

    When a VLM model is configured, a :class:`VLMEnricher` enriches
    image chunks with generated captions.
    """
    resolvers: list[BaseResolver] = []

    vlm_model = resolve_vlm_model(config.llm.default_model, config.llm.vlm_model)
    if vlm_model:
        try:
            vlm_agent = VlmAgent(
                vlm_model,
                timeout=config.llm.timeout,
                llm=config.llm,
            )
            resolvers.append(
                VLMEnricher(
                    agent=vlm_agent,
                    min_image_bytes=config.vlm_min_image_bytes,
                    min_image_dimension=config.vlm_min_image_dimension,
                    failures=UnitFailureLog(
                        stage="vlm",
                        ratio=config.llm.batch_failure_threshold_ratio,
                        min_units=config.llm.batch_failure_threshold_min,
                    ),
                )
            )
            logger.info("VLM image captioning enabled (model=%s)", vlm_model)
        except Exception as exc:
            logger.debug("VLMEnricher not available: %s", exc, exc_info=is_verbose())

    if not resolvers:
        logger.info("No resolvers configured — chunks will pass through without resolution")

    return ResolverChain(resolvers)
