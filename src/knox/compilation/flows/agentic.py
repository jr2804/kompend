"""Agentic extraction flow — structured memory with compaction-managed digests.

``extract_agentic()`` implements the ``AGENTIC`` extraction strategy.
Unlike ``INCREMENTAL`` (raw ``message_history`` that grows unbounded),
this flow maintains a structured ``ExtractionMemory`` digest and injects
a compacted summary as a ``SystemPromptPart`` between sections.

Key differences vs other strategies:

- **Memory-not-history**: the digest is structured (concepts/entities/tags)
  and compaction-managed, not raw message history.
- **Sequential per source**: the agent walks sections in order; each
  section adds to memory, but memory never holds more than a configurable
  token budget.
- **Per-source final output**: the final ``Extraction`` is derived from
  memory, not merged from per-batch outputs.
- **Source-level caching**: since memory state depends on prior sections,
  per-batch caching is disabled; the final result is cached keyed on
  ``(source, full_text_hash, strategy)``.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from pydantic_ai.messages import ModelRequest, SystemPromptPart

from knox.agents.deps import PipelineAgentDeps
from knox.agents.extraction import ExtractionAgent
from knox.compilation._two_phase_core import Extraction
from knox.compilation.asset_tools import build_asset_tools
from knox.compilation.extraction_memory import ExtractionMemory
from knox.core.models.enums import CacheDeps
from knox.core.utils.exceptions import LLMOutputError

logger = logging.getLogger(__name__)


def extract_agentic(
    compiler: Any,
    source: str,
    texts: list[str],
    extraction_system: str,
) -> Extraction | None:
    """Run agentic extraction with structured memory across sections.

    Parameters
    ----------
    compiler:
        The TwoPhaseCompiler instance (provides config, cache, budget).
    source:
        Source identifier (file path stem, URL, etc.).
    texts:
        List of section text strings, one per agent call, in document
        order. Each section is submitted independently (not batched).
    extraction_system:
        The system prompt body (for caching / progress logging).

    Returns:
    -------
    Extraction or None if all sections failed.

    Notes:
    -----
    - Caching is at the source level (keyed on text hash), NOT per
      section, because memory depends on prior section results.
    - Memory compaction triggers when estimated digest tokens exceed
      ``extract_output_max`` * 2.
    """
    budget = compiler._budget
    full_text_hash = _full_text_hash(texts)

    # Source-level cache
    cache_deps: dict[str, str] = {
        CacheDeps.SOURCE_TEXT: full_text_hash,
        CacheDeps.STAGE: "agentic",
    }
    if not compiler._config.no_cache:
        cached = compiler._cache_store.get("extract", cache_deps)
        if cached is not None:
            logger.info("TwoPhase: agentic extraction cache hit for '%s'", source)
            return cached
        logger.info(
            "TwoPhase: agentic extraction cache miss for '%s' (text_hash=%.16s)",
            source,
            full_text_hash,
        )

    # Build agent — with asset lookup tools when the store is populated
    tools = (
        build_asset_tools(compiler._asset_store)
        if compiler._asset_store and compiler._asset_store.total_count() > 0
        else None
    )
    agent = ExtractionAgent(
        model_string=compiler._model or "",
        timeout=compiler._timeout,
        max_tokens=budget.extract_output_max,
        skills=compiler._config.skills,
        retries=compiler._config.llm.output_retries if compiler._config.llm.output_retries_enabled else 0,
        delta_retry_enabled=compiler._config.llm.delta_retry_enabled,
        reasoning_effort=compiler._config.llm.reasoning_effort,
        context_manager=compiler._config.llm.context_manager,
        context_window=compiler._context_window.max_input,
        tools=tools,
    )

    memory = ExtractionMemory()
    compaction_threshold = budget.extract_output_max * 2
    batch_retries = compiler._config.llm.batch_retry_attempts
    overall_succeeded = False

    for section_idx, section_text in enumerate(texts):
        logger.debug(
            "TwoPhase: agentic section %d/%d for '%s'",
            section_idx + 1,
            len(texts),
            source,
        )

        # Build message history with memory digest injected as a system message
        memory_digest = memory.summarize(max_tokens=budget.extract_output_max)
        message_history: list[Any] = []
        if tools:
            message_history.append(
                ModelRequest(
                    parts=[
                        SystemPromptPart(
                            content="You have access to on-demand asset lookup tools: "
                            "`get_table(asset_id)` returns pipe-delimited table text; "
                            "`get_figure_description(asset_id)` returns figure caption/alt-text. "
                            "Use them when you need full table data or figure details "
                            "beyond the compact `\u00b6` reference in the section text."
                        )
                    ]
                )
            )
        if memory_digest.strip():
            msg = ModelRequest(parts=[SystemPromptPart(content=f"=== Context from prior sections ===\n{memory_digest}")])
            message_history.append(msg)

        deps = PipelineAgentDeps(
            text=section_text,
            source_id=f"{source}:{section_idx}",
            timeout=compiler._timeout,
        )

        validated: Any = None
        for attempt in range(1 + batch_retries):
            try:
                validated, _new_messages = agent.run_with_messages(
                    deps,
                    message_history=message_history or None,
                )
            except (LLMOutputError, Exception) as exc:
                logger.debug(
                    "TwoPhase: agentic section %d attempt %d failed: %s",
                    section_idx + 1,
                    attempt + 1,
                    exc,
                )
                validated = None

            if validated is not None:
                break
            if attempt < batch_retries:
                logger.info(
                    "TwoPhase: agentic section retry %d/%d for '%s' batch %d",
                    attempt + 1,
                    batch_retries,
                    source,
                    section_idx,
                )

        compiler._batch_tracker.record(failed=validated is None)

        if validated is None:
            logger.warning(
                "TwoPhase: agentic section %d failed for '%s' — skipping",
                section_idx + 1,
                source,
            )
            continue

        # Convert validated output to Extraction and ingest into memory
        section_extraction = Extraction(
            source=source,
            summary=validated.summary,
            entities=[e.model_dump() for e in validated.entities],
            concepts=[c.model_dump() for c in validated.concepts],
            tags=validated.tags,
        )
        memory.ingest(section_extraction)
        overall_succeeded = True

        if memory._estimate_tokens() > compaction_threshold:
            memory.compact()
            logger.info(
                "TwoPhase: compacted memory for '%s' (section %d) — %d concepts, %d entities, ~%d tokens",
                source,
                section_idx + 1,
                len(memory.concepts),
                len(memory.entities),
                memory._estimate_tokens(),
            )

    if not overall_succeeded:
        logger.warning("TwoPhase: agentic extraction failed for all sections of '%s'", source)
        return None

    final = memory.to_extraction(source)

    if not compiler._config.no_cache:
        compiler._cache_store.set("extract", cache_deps, final)
        logger.info("TwoPhase: cached agentic result for '%s'", source)

    return final


def _full_text_hash(texts: list[str]) -> str:
    """SHA-256 of concatenated section texts for cache key."""
    return hashlib.sha256("".join(texts).encode()).hexdigest()
