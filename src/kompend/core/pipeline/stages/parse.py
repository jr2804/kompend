"""Tier 1 (parse + chunk) orchestration: dispatch, dedup gating, attachments.

One :class:`HeavyParseGuard` per run, one worker pool, one ledger; the
per-source unit lives in :mod:`kompend.core.pipeline.stages.parse_sources` and
the cache halves in :mod:`kompend.core.pipeline.stages.parse_store`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextvars import copy_context
from pathlib import Path
from typing import Any

from kompend.agents.failure import UnitFailureLog
from kompend.agents.vlm import VlmAgent
from kompend.core.models import (
    ChunkTree,
)
from kompend.core.models.attachments import AttachmentManifest, adopt_orphan_attachments
from kompend.core.paths import VaultPaths
from kompend.core.pipeline.attachments import (
    ingest_attachment,
    iter_ingestible_attachments,
)
from kompend.core.pipeline.config import PipelineConfig
from kompend.core.pipeline.parallelism import describe_parallelism
from kompend.core.pipeline.reasons import reason_code_for
from kompend.core.pipeline.runtime import PipelineRuntime
from kompend.core.pipeline.stages.parse_sources import _process_single_source
from kompend.core.reporting.ledger import StageLedger, stage_ledger
from kompend.core.store import vault_db_path
from kompend.core.types import ReasonCode, StageOutcome
from kompend.core.verbosity import is_verbose
from kompend.llm.models import resolve_vlm_model
from kompend.llm.user_providers import effective_workers
from kompend.logging import get_logger
from kompend.preprocessing.dedup import DedupStore
from kompend.preprocessing.host_memory import (
    MEGABYTE,
    HeavyParseGuard,
)
from kompend.resolution.base import BaseResolver
from kompend.resolution.chain import ResolverChain
from kompend.resolution.vlm_enricher import VLMEnricher

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
