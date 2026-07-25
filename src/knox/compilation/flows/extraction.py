"""Extraction flow — LLM extraction from source documents.

Extracted from :class:`TwoPhaseCompiler` — extraction phase methods
that operate on compiler instance state.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from knox.agents.deps import PipelineAgentDeps
from knox.agents.extraction import ExtractionAgent
from knox.agents.extraction_reduce import ExtractionReduceAgent
from knox.compilation._two_phase_core import (
    Extraction,
    ExtractionSet,
    merge_extractions,
    tokens_to_chars,
    try_pydanticai_extraction,
)
from knox.compilation.flows.assembly import build_tree_texts
from knox.core.config._configs import ExtractionStrategy
from knox.core.models import ChunkTree, ResolvedChunk
from knox.core.models.enums import CacheDeps
from knox.core.templates import get_template_loader
from knox.core.utils.exceptions import LLMOutputError
from knox.logging import get_logger

logger = get_logger(__name__)

_loader = get_template_loader()


def batch_texts(texts: list[str], char_budget: int) -> list[str]:
    """Group texts into batches that fit within *char_budget* (chars)."""
    batches: list[str] = []
    parts: list[str] = []
    length = 0
    for text in texts:
        if length + len(text) > char_budget and parts:
            batches.append("\n\n".join(parts))
            parts = []
            length = 0
        parts.append(text)
        length += len(text)
    if parts:
        batches.append("\n\n".join(parts))
    return batches


def run_extraction(
    compiler: Any,
    chunks: list[ResolvedChunk],
    *,
    trees: list[ChunkTree] | None = None,
) -> ExtractionSet:  # noqa: PLR0915
    """Phase 1: run the LLM in extraction mode for each source."""
    compiler._llm_stats = []
    extractions = ExtractionSet()
    by_source: dict[str, list[ResolvedChunk]] = defaultdict(list)
    for rc in chunks:
        label = str(rc.chunk.source_file or "unknown")
        by_source[label].append(rc)

    tree_by_source: dict[str, ChunkTree] = {}
    if trees:
        for t in trees:
            if t.source_file is not None:
                tree_by_source[str(t.source_file)] = t

    extraction_system = _loader.get_body("compilation/extraction_system.md.j2")
    num_sources = len(by_source)
    logger.info("TwoPhase: extracting %d source(s)", num_sources)

    cw = compiler._context_window
    budget = compiler._budget
    extract_char_budget = tokens_to_chars(
        max(1, cw.max_input - compiler._extract_system_tokens - budget.extract_output_max),
        chars_per_token=compiler._config.llm.default_chars_per_token,
    )

    # Pre-compute total chars for weighted progress bar (larger files = more progress)
    source_char_totals: list[int] = []
    for _source, source_chunks in by_source.items():
        source_char_totals.append(sum(len(rc.chunk.text or "") for rc in source_chunks))
    total_chars_all = sum(source_char_totals) or len(by_source)
    cumulative_chars = 0

    for source_idx, (source, source_chunks) in enumerate(by_source.items(), start=1):
        logger.info("TwoPhase: extract [%d/%d] %s", source_idx, num_sources, source)
        src_start = time.monotonic()

        total_chars = source_char_totals[source_idx - 1]
        cumulative_chars += total_chars
        source_name = Path(source).name
        compiler._report(cumulative_chars, total_chars_all, f"extracting [{source_idx}/{num_sources}] {source_name}")

        tree = tree_by_source.get(source)
        texts = build_tree_texts(tree) if tree is not None else [rc.chunk.text for rc in source_chunks if rc.chunk.text]

        if not texts:
            extractions.add(Extraction(source=source))
            compiler._extraction_stats.append(
                {
                    "source": source,
                    "summary": "",
                    "duration_s": 0.0,
                    "summary_chars": 0,
                    "summary_words": 0,
                    "entity_count": 0,
                    "concept_count": 0,
                    "tag_count": 0,
                    "batches": 0,
                }
            )
            continue

        batches_list = batch_texts(texts, extract_char_budget)
        batch_extractions: list[Extraction | None] = []

        strategy = compiler._config.llm.extraction_strategy
        if strategy == ExtractionStrategy.INCREMENTAL and len(batches_list) > 1:
            batch_extractions, _ = extract_incremental(compiler, source, batches_list, extraction_system)
        else:
            for combined in batches_list:
                batch_extractions.append(extract_one(compiler, source, combined, extraction_system))
                if compiler._batch_tracker.should_stop():
                    break

        valid = [e for e in batch_extractions if e is not None]
        if not valid:
            logger.warning("TwoPhase: all extraction batches failed for '%s' — skipped", source)
        merged = combine_extractions(compiler, source, valid)
        extractions.add(merged)
        compiler._extraction_stats.append(
            {
                "source": source,
                "summary": merged.summary,
                "duration_s": round(time.monotonic() - src_start, 1),
                "summary_chars": len(merged.summary),
                "summary_words": len(merged.summary.split()),
                "entity_count": len(merged.entities),
                "concept_count": len(merged.concepts),
                "tag_count": len(merged.tags),
                "batches": len(batch_extractions),
            }
        )
        if compiler._batch_tracker.should_stop():
            logger.error(
                "TwoPhase: stopping extraction early — %d/%d batches failed (model '%s' is not suitable for structured JSON output)",
                compiler._batch_tracker.failed,
                compiler._batch_tracker.total,
                compiler._model,
            )
            break
    return extractions


def extract_one(compiler: Any, source: str, combined: str, extraction_system: str) -> Extraction | None:
    """Run a single extraction LLM call with cache."""
    budget = compiler._budget
    extract_deps: dict[str, str] = {
        CacheDeps.SOURCE_TEXT: hashlib.sha256(combined.encode()).hexdigest(),
        CacheDeps.CHUNK_SIZE_MIN: str(budget.chunk_size_min),
        CacheDeps.CHUNK_SIZE_MAX: str(budget.chunk_size_max),
    }
    if not compiler._config.no_cache:
        cached = compiler._cache_store.get("extract", extract_deps)
        if cached is not None:
            logger.info("TwoPhase: extraction cache hit for '%s'", source)
            return cached
        logger.info(
            "TwoPhase: extraction cache miss for '%s' (text_hash=%.16s, chunk_min=%s, chunk_max=%s)",
            source,
            extract_deps[CacheDeps.SOURCE_TEXT],
            extract_deps[CacheDeps.CHUNK_SIZE_MIN],
            extract_deps[CacheDeps.CHUNK_SIZE_MAX],
        )

    agent = ExtractionAgent(
        model_string=compiler._model or "",
        timeout=compiler._timeout,
        max_tokens=compiler._budget.extract_output_max,
        skills=compiler._config.skills,
        retries=compiler._config.llm.output_retries if compiler._config.llm.output_retries_enabled else 0,
        delta_retry_enabled=compiler._config.llm.delta_retry_enabled,
        reasoning_effort=compiler._config.llm.reasoning_effort,
        context_manager=compiler._config.llm.context_manager,
        context_window=compiler._context_window.max_input,
    )

    # Per-batch retry: the agent already retries internally (via PydanticAI's
    # ``retries``), but this loop retries at the batch level when the entire
    # agent call returns None (e.g. transient timeout, empty response that
    # exhausts internal retries). ``compile.llm.batch_retry_attempts`` is
    # configurable — set to 0 to disable.
    batch_retries = compiler._config.llm.batch_retry_attempts
    validated: Any | None = None
    for attempt in range(1 + batch_retries):
        validated = try_pydanticai_extraction(
            agent,
            text=combined,
            source_id=source,
            timeout=compiler._timeout,
        )
        if validated is not None:
            break
        if attempt < batch_retries:
            logger.info(
                "TwoPhase: extraction batch retry %d/%d for '%s'",
                attempt + 1,
                batch_retries,
                source,
            )

    compiler._batch_tracker.record(failed=validated is None)
    if validated is None:
        logger.warning(
            "TwoPhase: extraction skipped for source '%s' — agent returned no result after %d batch attempt(s)",
            source,
            1 + batch_retries,
        )
        return None
    result = Extraction(
        source=source,
        summary=validated.summary,
        entities=[e.model_dump() for e in validated.entities],
        concepts=[c.model_dump() for c in validated.concepts],
        tags=validated.tags,
    )
    if not compiler._config.no_cache:
        compiler._cache_store.set("extract", extract_deps, result)
    return result


def extract_incremental(
    compiler: Any,
    source: str,
    batches_list: list[str],
    extraction_system: str,
) -> tuple[list[Extraction | None], bool]:
    """Run extraction with message_history threading between batches.

    Per-batch cache is checked first — cached batches skip the LLM call
    entirely. Only uncached batches are sent to the model, and the
    message history threads between consecutive uncached batches.
    """
    budget = compiler._budget
    results: list[Extraction | None] = []
    history: list[Any] = []
    agent: ExtractionAgent | None = None
    for batch_idx, combined in enumerate(batches_list):
        # Per-batch cache (same key as extract_one)
        extract_deps: dict[str, str] = {
            CacheDeps.SOURCE_TEXT: hashlib.sha256(combined.encode()).hexdigest(),
            CacheDeps.CHUNK_SIZE_MIN: str(budget.chunk_size_min),
            CacheDeps.CHUNK_SIZE_MAX: str(budget.chunk_size_max),
        }
        if not compiler._config.no_cache:
            cached = compiler._cache_store.get("extract", extract_deps)
            if cached is not None:
                logger.info("TwoPhase: extraction cache hit for '%s' batch %d", source, batch_idx)
                results.append(cached)
                continue
            logger.info(
                "TwoPhase: extraction cache miss for '%s' batch %d (text_hash=%.16s)",
                source,
                batch_idx,
                extract_deps[CacheDeps.SOURCE_TEXT],
            )

        if agent is None:
            agent = ExtractionAgent(
                model_string=compiler._model or "",
                timeout=compiler._timeout,
                max_tokens=compiler._budget.extract_output_max,
                skills=compiler._config.skills,
                retries=compiler._config.llm.output_retries if compiler._config.llm.output_retries_enabled else 0,
                delta_retry_enabled=compiler._config.llm.delta_retry_enabled,
                reasoning_effort=compiler._config.llm.reasoning_effort,
                context_manager=compiler._config.llm.context_manager,
                context_window=compiler._context_window.max_input,
            )
        deps = PipelineAgentDeps(
            text=combined,
            source_id=f"{source}:{batch_idx}",
            timeout=compiler._timeout,
        )
        try:
            validated, new_messages = agent.run_with_messages(
                deps,
                message_history=history if history else None,
            )
            history = new_messages
            compiler._batch_tracker.record(failed=False)
            result = Extraction(
                source=source,
                summary=validated.summary,
                entities=[e.model_dump() for e in validated.entities],
                concepts=[c.model_dump() for c in validated.concepts],
                tags=validated.tags,
            )
            if not compiler._config.no_cache:
                compiler._cache_store.set("extract", extract_deps, result)
            results.append(result)
        except LLMOutputError:
            logger.warning(
                "TwoPhase: incremental extraction failed for '%s' batch %d",
                source,
                batch_idx,
            )
            compiler._batch_tracker.record(failed=True)
            results.append(None)
        if compiler._batch_tracker.should_stop():
            return results, True
    return results, False


def combine_extractions(compiler: Any, source: str, parts: list[Extraction]) -> Extraction:
    """Combine per-batch extractions into one document-level Extraction."""
    if not parts:
        return Extraction(source=source)
    if len(parts) == 1:
        return parts[0]
    strategy = compiler._config.llm.extraction_strategy
    if strategy == ExtractionStrategy.REDUCE:
        return compiler._reduce_extractions(source, parts)
    return merge_extractions(source, parts)


def reduce_extractions(compiler: Any, source: str, parts: list[Extraction]) -> Extraction:
    """LLM-powered reduce: merge partial extractions into a coherent whole."""
    partial_json = json.dumps([p.to_dict() for p in parts], ensure_ascii=False)
    reduce_deps_key: dict[str, str] = {
        CacheDeps.SOURCE_TEXT: hashlib.sha256(partial_json.encode()).hexdigest(),
        CacheDeps.MODEL: compiler._model,
        CacheDeps.STAGE: "reduce",
    }
    if not compiler._config.no_cache:
        cached = compiler._cache_store.get("extract", reduce_deps_key)
        if cached is not None:
            return cached

    agent = ExtractionReduceAgent(
        model_string=compiler._model or "",
        timeout=compiler._timeout,
        max_tokens=compiler._budget.extract_output_max,
        skills=compiler._config.skills,
        retries=compiler._config.llm.output_retries,
        reasoning_effort=compiler._config.llm.reasoning_effort,
    )
    try:
        deps = PipelineAgentDeps(text=partial_json, source_id=f"{source}:reduce", timeout=compiler._timeout)
        validated = agent.run(deps)
    except LLMOutputError:
        logger.warning("TwoPhase: reduce failed for '%s' — falling back to naive merge", source)
        return merge_extractions(source, parts)
    result = Extraction(
        source=source,
        summary=validated.summary,
        entities=[e.model_dump() for e in validated.entities],
        concepts=[c.model_dump() for c in validated.concepts],
        tags=validated.tags,
    )
    if not compiler._config.no_cache:
        compiler._cache_store.set("extract", reduce_deps_key, result)
    return result
