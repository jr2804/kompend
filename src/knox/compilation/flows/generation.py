"""Generation flow — per-concept page generation and wiki assembly.

Extracted from :class:`TwoPhaseCompiler` — generation phase methods
that operate on compiler instance state.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from knox.agents.generation import GenerationAgent, make_agentic_generation_agent
from knox.agents.types import ToolGenerationDeps
from knox.compilation._checkpoints import (
    delete_gen_checkpoint,
    load_gen_checkpoint,
    save_gen_checkpoint,
)
from knox.compilation._two_phase_core import (
    ExtractionSet,
    TwoPhaseRuntime,
    compute_quality_score,
    extract_images_from_chunks,
    segment_sources,
    slugify,
    tokens_to_chars,
    try_pydanticai_generation,
    upsert_node,
)
from knox.compilation.concept_broker import ConceptBroker
from knox.compilation.concept_keys import normalize_concept_key
from knox.compilation.flows.assembly import (
    attach_attributes_to_nodes,
    attach_tags_to_nodes,
    build_edges,
    merge_similar_concepts,
)
from knox.compilation.schema import GenerationPage
from knox.core.config.keys import MANDATORY_KINDS
from knox.core.models import ChunkTree, CompiledWiki, ImageRef, ResolvedChunk, TocEntry, WikiField, WikiNode, load_cached_wiki
from knox.core.models.enums import CacheDeps, ExtractionField, MetadataField
from knox.core.templates import get_template_loader
from knox.core.utils.exceptions import LLMOutputError
from knox.lint.base import LintRuleOutcome
from knox.lint.structural import BrokenWikilinksRule
from knox.llm.format import encode_for_llm
from knox.logging import get_logger

# Matches wikilink-wrapped source citations like [[sources/foo.pdf]] so they
# can be normalized to the intended [sources/foo.pdf] inline citation format
# regardless of LLM compliance with the system prompt.
_SOURCE_WIKILINK_RE = re.compile(r"\[\[\s*(sources/[^\]\n]+?)\s*\]\]")

logger = get_logger(__name__)

_loader = get_template_loader()


def run_generation(
    compiler: TwoPhaseRuntime,
    extractions: ExtractionSet,
    chunks: list[ResolvedChunk],
    *,
    trees: list[ChunkTree] | None = None,
    pinned_facts: list[str] | None = None,
) -> CompiledWiki:
    """Phase 2: merge extractions and generate per-concept pages."""
    concepts = prepare_concepts(compiler, extractions)
    if not concepts:
        return CompiledWiki(
            nodes=[],
            edges=[],
            toc=[],
            metadata={
                WikiField.CHUNK_COUNT: len(chunks),
                MetadataField.TWO_PHASE: True,
                MetadataField.GENERATION_SKIPPED: True,
                MetadataField.CONCEPTS_AVAILABLE: 0,
                MetadataField.SKIP_REASON: "extraction returned no concepts",
                MetadataField.NODE_COUNT: 0,
            },
        )

    all_images = extract_images_from_chunks(chunks)
    source_segments = prepare_source_segments(compiler, chunks)
    pinned_clean = list(dict.fromkeys(f.strip() for f in (pinned_facts or []) if f and f.strip()))
    _loader.get_body("compilation/generation_system.md.j2")
    known_titles = build_known_titles(compiler, concepts)

    try:
        nodes, slug_to_idx, _caller_progress = run_generation_with_checkpoint(
            compiler,
            concepts,
            source_segments,
            all_images,
            pinned_clean,
            known_titles=known_titles,
        )
    except LLMOutputError:
        # Generation produced no pages for at least one segment. The
        # per-segment checkpoint save inside ``generate_all_pages_agentic``
        # already preserves progress; do not delete it. Re-raise so the
        # caller surfaces a clear "re-run to retry" message.
        logger.warning(
            "TwoPhase: agentic generation raised — checkpoint preserved. Re-run to retry the failed segment(s) only.",
        )
        raise

    # Success: delete the checkpoint so the next run starts fresh.
    if compiler._active_gen_deps:
        delete_gen_checkpoint(compiler._config, compiler._active_gen_deps)

    return assemble_wiki(compiler, nodes, slug_to_idx, chunks, extractions, concepts=concepts, pinned_clean=pinned_clean)


def prepare_concepts(compiler: TwoPhaseRuntime, extractions: ExtractionSet) -> list[dict[str, Any]]:
    """Merge and validate concepts and entities from extractions.

    Entities become mandatory pages (kind ``entity``); entity specs whose
    normalized key already exists as a concept are dropped so no page is
    duplicated. Returns empty list if neither yielded anything.
    """
    concepts = extractions.concept_pages()
    embed_model = compiler._config.embedding.model
    budget = compiler._config.budget
    concepts = merge_similar_concepts(
        concepts,
        embed_model,
        threshold=budget.concept_merge_threshold,
        string_threshold=budget.concept_string_merge_threshold,
        jaccard_min_token_len=budget.jaccard_min_token_len,
    )
    concepts = _with_entity_pages(compiler, extractions, concepts)
    if concepts:
        sample_limit = compiler._config.display.concept_names
        logger.info(
            "TwoPhase: %d concept(s) ready for generation (sample: %s)",
            len(concepts),
            [c[ExtractionField.NAME] for c in concepts[:sample_limit]],
        )
    else:
        sample_keys: set[str] = set()
        for ext in extractions.extractions:
            for c in ext.concepts:
                sample_keys.update(c.model_fields)
        logger.warning(
            "TwoPhase: concept_pages() empty — extraction concept keys: %s",
            sorted(sample_keys) or "(no concepts returned)",
        )
    return concepts


def _with_entity_pages(
    compiler: TwoPhaseRuntime,
    extractions: ExtractionSet,
    concepts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append mandatory entity pages not already covered by a concept."""
    concept_keys = {normalize_concept_key(str(c.get(ExtractionField.NAME, ""))) for c in concepts}
    entities = [e for e in extractions.entity_pages() if normalize_concept_key(str(e.get("name", ""))) not in concept_keys]
    if entities:
        logger.info("TwoPhase: %d entity page(s) queued for generation", len(entities))
        concepts.extend(entities)
    return concepts


def prepare_source_segments(compiler: TwoPhaseRuntime, chunks: list[ResolvedChunk]) -> list[list[tuple[str, str]]]:
    """Build source segments sized to the context window."""
    by_source: dict[str, str] = defaultdict(str)
    for rc in chunks:
        label = Path(str(rc.chunk.source_file or "unknown")).stem
        if rc.chunk.text:
            by_source[label] += rc.chunk.text + "\n\n"
    sources: list[tuple[str, str]] = list(by_source.items())

    cw = compiler._context_window
    budget = compiler._budget
    segment_char_size = tokens_to_chars(
        max(compiler._config.llm.min_segment_tokens, cw.max_input - compiler._gen_system_tokens - budget.gen_output_max),
        chars_per_token=compiler._config.llm.default_chars_per_token,
    )
    logger.info(
        "TwoPhase: segment_char_size=%d chars (cw=%d, sys_tok=%d, gen_max=%d)",
        segment_char_size,
        cw.max_input,
        compiler._gen_system_tokens,
        budget.gen_output_max,
    )
    source_segments = segment_sources(sources, segment_char_size)
    logger.info(
        "TwoPhase: %d source(s) split into %d segment(s) (total content %d chars)",
        len(sources),
        len(source_segments),
        sum(len(c) for _, c in sources),
    )
    return source_segments


def run_generation_with_checkpoint(
    compiler: TwoPhaseRuntime,
    concepts: list[dict[str, Any]],
    source_segments: list[list[tuple[str, str]]],
    all_images: list[ImageRef],
    pinned_clean: list[str],
    *,
    known_titles: list[str] | None = None,
) -> tuple[list[WikiNode], dict[str, int], dict[str, list[str]]]:
    """Run agentic generation with checkpoint resume.

    On entry, loads the per-deps checkpoint (if any) and seeds the
    agentic loop with the already-produced nodes + broker progress so
    finished segments are skipped without further LLM calls. On exit
    (success), the checkpoint is deleted; on ``LLMOutputError`` (a
    segment produced no pages) the per-segment checkpoint save inside
    ``generate_all_pages_agentic`` preserves progress so the next run
    resumes cheaply.
    """
    if not compiler._active_gen_deps:
        # No checkpoint key (e.g. caller forgot to set _active_gen_deps).
        # Skip resume; the in-loop save will still happen if a key is set.
        ck_nodes: list[WikiNode] = []
        ck_slug_to_idx: dict[str, int] = {}
        ck_caller_progress: dict[str, list[str]] = {}
    else:
        checkpoint = load_gen_checkpoint(compiler._config, compiler._active_gen_deps)
        if checkpoint is not None:
            ck_nodes, ck_slug_to_idx, _completed_batches, ck_caller_progress = checkpoint
            resumed_pages = sum(len(v) for v in ck_caller_progress.values())
            if resumed_pages:
                logger.info(
                    "TwoPhase: generation checkpoint loaded — %d segment(s) already done, %d page(s) to resume.",
                    len(ck_caller_progress),
                    resumed_pages,
                )
        else:
            ck_nodes, ck_slug_to_idx, ck_caller_progress = [], {}, {}

    return generate_all_pages_agentic(
        compiler,
        concepts,
        source_segments,
        all_images,
        pinned_facts=pinned_clean,
        known_titles=known_titles,
        initial_nodes=ck_nodes,
        initial_slug_to_idx=ck_slug_to_idx,
        initial_caller_progress=ck_caller_progress,
    )


def generate_all_pages(  # noqa: PLR0915 — legacy template path; long by design
    compiler: TwoPhaseRuntime,
    concepts: list[dict[str, Any]],
    source_segments: list[list[tuple[str, str]]],
    generation_system: str,
    all_images: list[ImageRef],
    *,
    pinned_facts: list[str],
    initial_nodes: list[WikiNode] | None = None,
    initial_slug_to_idx: dict[str, int] | None = None,
    completed_batches: set[int] | None = None,
    known_titles: list[str] | None = None,
) -> tuple[list[WikiNode], dict[str, int]]:
    """Generate (or fallback-generate) one node per concept/segment pair."""
    nodes: list[WikiNode] = list(initial_nodes or [])
    slug_to_idx: dict[str, int] = dict(initial_slug_to_idx or {})
    completed: set[int] = set(completed_batches or [])

    num_concepts = len(concepts)
    num_segments = len(source_segments)
    # The legacy non-agentic loop uses a fixed batch size.
    # Since we've removed concepts_per_batch from BudgetConfig,
    # we use a reasonable fallback default (8).
    batch_size = 8
    logger.info(
        "TwoPhase: generating %d concept(s) across %d source segment(s) (batch_size=%d, pinned=%d)",
        num_concepts,
        num_segments,
        batch_size,
        len(pinned_facts),
    )
    gen_progress_total = ((num_concepts + batch_size - 1) // batch_size) * num_segments
    gen_progress_step = 0

    # ponytail: cap failed segment retries per batch.
    MAX_FAILED = 3

    for batch_start in range(0, num_concepts, batch_size):
        if batch_start in completed:
            gen_progress_step += 1
            continue
        batch = concepts[batch_start : batch_start + batch_size]
        failed_count = 0

        for seg_idx, segment in enumerate(source_segments):
            if failed_count >= MAX_FAILED:
                break
            gen_progress_step += 1
            batch_num = batch_start // batch_size + 1
            total_batches = (num_concepts + batch_size - 1) // batch_size
            seg_label = f"generating [{gen_progress_step}/{gen_progress_total}] batch {batch_num}/{total_batches} seg {seg_idx + 1}"
            compiler._report(gen_progress_step, gen_progress_total, seg_label)
            logger.info("TwoPhase: %s", seg_label)
            content_block = "\n\n".join(f"### {label}\n{content}" for label, content in segment)

            agent = GenerationAgent(
                model_string=compiler._model or "",
                timeout=compiler._timeout,
                max_tokens=compiler._budget.gen_output_max,
                skills=compiler._config.skills,
                retries=compiler._config.llm.output_retries if compiler._config.llm.output_retries_enabled else 0,
                delta_retry_enabled=compiler._config.llm.delta_retry_enabled,
                reasoning_effort=compiler._config.llm.reasoning_effort,
                context_manager=compiler._config.llm.context_manager,
                context_window=compiler._context_window.max_input,
                rlm=compiler._config.llm.rlm,
                rlm_context_field="content",
            )
            gen_output = try_pydanticai_generation(
                agent,
                concepts_json=encode_for_llm(batch, compiler._config.llm.data_format),
                content=content_block,
                source_id=seg_label,
                pinned_facts=pinned_facts,
                categories=_categories_block(compiler._config.extraction_categories),
                known_titles=known_titles,
                timeout=compiler._timeout,
            )
            debug_file = getattr(agent, "_last_debug_file", None)

            compiler._batch_tracker.record(failed=gen_output is None)

            if gen_output is None:
                # This segment failed — but the batch will be retried against the
                # next source segment. Only log at debug level here.
                remaining = num_segments - seg_idx - 1
                if remaining > 0:
                    logger.debug(
                        "TwoPhase: segment %d/%d failed for concepts #%d-%d — will retry with %d remaining segment(s)",
                        seg_idx + 1,
                        num_segments,
                        batch_start,
                        batch_start + len(batch) - 1,
                        remaining,
                    )
                failed_count += 1
                if failed_count >= MAX_FAILED or seg_idx == num_segments - 1:
                    # All segments exhausted — these concepts are truly missing.
                    concept_names = [c.get(ExtractionField.NAME, f"concept-{batch_start + i}") for i, c in enumerate(batch)]
                    skipped_str = ", ".join(concept_names)
                    logger.warning(
                        "TwoPhase: %d/%d concepts will have no wiki page (all %d source segment(s) failed for batch #%d-%d): [%s]",
                        len(batch),
                        num_concepts,
                        num_segments,
                        batch_start,
                        batch_start + len(batch) - 1,
                        skipped_str,
                    )
                if compiler._batch_tracker.should_stop():
                    logger.error(
                        "TwoPhase: stopping generation early — %d/%d batches failed (model '%s' is not suitable for structured JSON output)",
                        compiler._batch_tracker.failed,
                        compiler._batch_tracker.total,
                        compiler._model,
                    )
                    return nodes, slug_to_idx
                continue

            # Filter out pages with empty content — small models frequently
            # produce valid titles/slugs but empty content when overwhelmed.
            # Retrying causes degradation, so accept and filter instead.
            empty_pages = [p for p in gen_output.pages if not (p.content or "").strip()]
            valid_pages = [p for p in gen_output.pages if (p.content or "").strip()]
            if empty_pages:
                names = ", ".join(p.title or "(untitled)" for p in empty_pages)
                debug_hint = f" (debug: {debug_file})" if debug_file else ""
                logger.warning(
                    "TwoPhase: batch %d-%d: %d/%d pages have empty content — filtered out ([%s]).%s",
                    batch_start,
                    batch_start + len(batch) - 1,
                    len(empty_pages),
                    len(gen_output.pages),
                    names,
                    debug_hint,
                )

            for idx, page in enumerate(valid_pages):
                node_id = slugify(page.title or f"concept-{batch_start + idx}", max_length=compiler._config.llm.slug_max_length)
                kind = _page_kind(page, batch, compiler._config.extraction_categories)
                # Strip double brackets around source citations so [[sources/x.pdf]]
                # becomes [sources/x.pdf] — matches the renderer’s _CITATION_RE and
                # keeps provenance out of broken-wikilinks lint findings.
                page_content = _SOURCE_WIKILINK_RE.sub(r"[\1]", page.content)
                upsert_node(
                    WikiNode(
                        id=node_id,
                        title=page.title or f"Concept {batch_start + idx}",
                        content=page_content,
                        node_type=kind,
                        source_chunks=[],
                        metadata={
                            MetadataField.PROVENANCE: page.provenance,
                            MetadataField.RELATED: page.related,
                            MetadataField.TWO_PHASE: True,
                            "images": all_images,
                            "frontmatter": {
                                "title": page.title or f"Concept {batch_start + idx}",
                            },
                        },
                    ),
                    nodes,
                    slug_to_idx,
                )

            # ponytail: break after the first successful segment for this batch.
            break

            # Checkpoint after each completed batch so interrupted runs can resume.
        completed.add(batch_start)
        save_gen_checkpoint(
            compiler._config,
            compiler._active_gen_deps,
            nodes,
            slug_to_idx,
            completed,
        )

    compiler._report(gen_progress_total, gen_progress_total, "generating complete")
    return nodes, slug_to_idx


def generate_all_pages_agentic(  # noqa: PLR0915 — per-segment loop + checkpoint save; long by design
    compiler: TwoPhaseRuntime,
    concepts: list[dict[str, Any]],
    source_segments: list[list[tuple[str, str]]],
    all_images: list[ImageRef],
    *,
    pinned_facts: list[str],
    known_titles: list[str] | None = None,
    initial_nodes: list[WikiNode] | None = None,
    initial_slug_to_idx: dict[str, int] | None = None,
    initial_caller_progress: dict[str, list[str]] | None = None,
) -> tuple[list[WikiNode], dict[str, int], dict[str, list[str]]]:
    """Generate wiki pages using per-segment agentic tool loop.

    Replaces the nested ``for batch × for segment`` loop with per-segment
    agent runs.  Each agent discovers concepts via ``fetch_concepts()`` and
    loads source content via ``read_segment()`` tools, deciding batch size
    and coverage strategy itself.

    Uses :class:`ConceptBroker` for server-side concept assignment and
    ``Memory`` + ``SlidingWindowCompaction`` + ``ClampOversizedMessages``
    as agent capabilities for cross-segment judgment.

    Returns ``(nodes, slug_to_idx, caller_progress)`` — the caller writes
    ``caller_progress`` to the generation checkpoint after every segment
    so an interrupted run resumes cheaply on the next invocation.
    """
    nodes: list[WikiNode] = list(initial_nodes or [])
    slug_to_idx: dict[str, int] = dict(initial_slug_to_idx or {})
    caller_progress: dict[str, list[str]] = {str(k): list(v) for k, v in (initial_caller_progress or {}).items()}
    num_segments = len(source_segments)
    num_concepts = len(concepts)

    broker = ConceptBroker(concepts, num_callers=num_segments)
    # Skip segments whose pages are already in the checkpoint: advance the
    # broker cursor past their consumed concepts so ``fetch`` returns []
    # immediately for those callers.
    if initial_caller_progress:
        for seg_idx in range(num_segments):
            seg_id = ConceptBroker.caller_id(seg_idx)
            names = initial_caller_progress.get(seg_id) or []
            if names:
                broker.mark_caller_consumed(seg_id, names)
                # Mirror the consumed set so resumed segments can be
                # reported as already-done in the progress UI.
                caller_progress.setdefault(seg_id, list(names))
    categories = _categories_block(compiler._config.extraction_categories)
    # Entity sub-type per concept (person/organization/standard/...) — routed
    # into entities/<type>/ subfolders by the vault writer.
    entity_types = {
        normalize_concept_key(str(c.get(ExtractionField.NAME, ""))): _sanitize_entity_type(str(c.get("type", "")))
        for c in concepts
        if c.get("kind") == "entity"
    }

    logger.info(
        "TwoPhase: agentic generation — %d concept(s), %d segment(s)",
        num_concepts,
        num_segments,
    )
    gen_progress_total = num_concepts
    pages_done = sum(len(initial_caller_progress.get(ConceptBroker.caller_id(i), [])) for i in range(num_segments)) if initial_caller_progress else 0
    if num_segments > 0:
        compiler._report(0, gen_progress_total, f"generating wiki ({num_concepts} concepts, {num_segments} segments)")

    for seg_idx, segment in enumerate(source_segments):
        seg_label = f"segment {seg_idx + 1}/{num_segments}"
        seg_id = ConceptBroker.caller_id(seg_idx)
        # Resume support: skip segments whose pages are already produced and
        # persisted in the checkpoint (broker cursor advanced past them,
        # caller_progress records the consumed concept names).
        if broker.remaining(seg_id) == 0 and seg_id in caller_progress:
            logger.info(
                "TwoPhase: agentic %s already complete in checkpoint — skipping (%d page(s)).",
                seg_label,
                len(caller_progress[seg_id]),
            )
            pages_done += len(caller_progress[seg_id])
            compiler._report(
                pages_done,
                gen_progress_total,
                f"generating [{pages_done}/{gen_progress_total}] {seg_label} (resumed)",
            )
            continue
        compiler._report(pages_done, gen_progress_total, f"generating [{pages_done}/{gen_progress_total}] {seg_label}")
        logger.info("TwoPhase: agentic %s", seg_label)

        content_block = "\n\n".join(f"### {label}\n{content}" for label, content in segment)

        agent = make_agentic_generation_agent(
            model_string=compiler._model or "",
            timeout=compiler._timeout,
            max_tokens=compiler._budget.gen_output_max,
            skills=compiler._config.skills,
            retries=compiler._config.llm.output_retries if compiler._config.llm.output_retries_enabled else 0,
            output_retries=compiler._config.llm.agentic_output_retries if compiler._config.llm.output_retries_enabled else 0,
            delta_retry_enabled=compiler._config.llm.delta_retry_enabled,
            reasoning_effort=compiler._config.llm.reasoning_effort,
            context_manager=compiler._config.llm.context_manager,
            context_window=compiler._context_window.max_input,
            rlm=compiler._config.llm.rlm,
        )

        deps = ToolGenerationDeps(
            source_content=content_block,
            source_id=ConceptBroker.caller_id(seg_idx),
            pinned_facts=pinned_facts,
            categories=categories,
            known_titles=known_titles or [],
            concept_broker=broker,
            timeout=compiler._timeout,
        )

        # Force full concept coverage: small models often stop after the first
        # fetch_concepts batch instead of looping until the broker is drained.
        # Re-run the segment agent until its assigned concepts are exhausted,
        # bounded by a hard cap (one pass per concept maximum).
        start_remaining = broker.remaining(seg_id)
        cap = start_remaining + 1
        passes = 0
        gen_output = None
        seg_start_nodes = len(nodes)
        seg_titles: set[str] = set()  # normalized page titles produced for this segment
        seg_consumed_names: list[str] = list(caller_progress.get(seg_id, []))  # accumulate for checkpoint
        while broker.remaining(seg_id) > 0 and passes < cap:
            passes += 1
            try:
                gen_output = agent.run(deps)
            except LLMOutputError as exc:
                logger.warning("Agentic generation failed for '%s': %s", seg_id, exc)
                gen_output = None
            except Exception as exc:  # ponytail: PydanticAI raises many subclassed errors; treat uniformly
                logger.warning("Agentic generation unexpected error for '%s': %s", seg_id, exc)
                gen_output = None
            debug_file = getattr(agent, "_last_debug_file", None)

            compiler._batch_tracker.record(failed=gen_output is None)

            if gen_output is None:
                logger.warning(
                    "TwoPhase: agentic %s failed — skipping (debug: %s)",
                    seg_label,
                    debug_file or "(none)",
                )
                break

            empty_pages = [p for p in gen_output.pages if not (p.content or "").strip()]
            valid_pages = [p for p in gen_output.pages if (p.content or "").strip()]
            if empty_pages:
                names = ", ".join(p.title or "(untitled)" for p in empty_pages)
                debug_hint = f" (debug: {debug_file})" if debug_file else ""
                logger.warning(
                    "TwoPhase: agentic %s: %d/%d pages have empty content — filtered out ([%s]).%s",
                    seg_label,
                    len(empty_pages),
                    len(gen_output.pages),
                    names,
                    debug_hint,
                )

            for idx, page in enumerate(valid_pages):
                seg_titles.add(normalize_concept_key(page.title or ""))
                node_id = slugify(page.title or f"concept-{seg_idx}-{idx}", max_length=compiler._config.llm.slug_max_length)
                kind = _allowed_kind(page.kind, compiler._config.extraction_categories)
                page_content = _SOURCE_WIKILINK_RE.sub(r"[\1]", page.content)
                node_metadata: dict[str, Any] = {
                    MetadataField.PROVENANCE: page.provenance,
                    MetadataField.RELATED: page.related,
                    MetadataField.TWO_PHASE: True,
                    "images": all_images,
                    "frontmatter": {
                        "title": page.title or f"Concept {idx}",
                    },
                }
                if kind == "entity":
                    node_metadata[MetadataField.ENTITY_TYPE] = entity_types.get(normalize_concept_key(page.title or ""), "entity")
                upsert_node(
                    WikiNode(
                        id=node_id,
                        title=page.title or f"Concept {idx}",
                        content=page_content,
                        node_type=kind,
                        metadata=node_metadata,
                    ),
                    nodes,
                    slug_to_idx,
                )

            # Requeue fetched-but-uncovered concepts: the model may fetch a
            # batch and only produce pages for part of it. Requeued concepts
            # are handed out again on the next pass (within the same cap).
            uncovered_fetched = [c for c in broker.consumed(seg_id) if normalize_concept_key(str(c.get(ExtractionField.NAME, ""))) not in seg_titles]
            if uncovered_fetched:
                broker.requeue(seg_id, uncovered_fetched)

            # Report page-based progress: pages produced for this segment,
            # bounded by its broker share (prior segments count fully).
            compiler._report(
                pages_done + min(len(seg_titles), start_remaining),
                gen_progress_total,
                f"generating [{pages_done + min(len(seg_titles), start_remaining)}/{gen_progress_total}] {seg_label}",
            )

        pages_done += start_remaining

        # Record every consumed concept (page produced for it) for the
        # checkpoint so the next run can resume past this segment.
        produced_names = sorted(seg_titles)
        seg_consumed_names.extend(produced_names)
        caller_progress[seg_id] = list(dict.fromkeys(seg_consumed_names))  # dedupe, preserve order

        # Persist checkpoint after every segment so an interrupted or
        # partially-failed run can resume cheaply without redoing finished
        # segments.
        if compiler._active_gen_deps:
            save_gen_checkpoint(
                compiler._config,
                compiler._active_gen_deps,
                nodes,
                slug_to_idx,
                set(),  # legacy template-path field; unused for agentic
                caller_progress=caller_progress,
            )

        uncovered = broker.remaining(seg_id)
        if uncovered > 0:
            len(nodes) - seg_start_nodes
            names = [str(c.get(ExtractionField.NAME, "?")) for c in broker.remaining_concepts(seg_id)]
            sample = ", ".join(names[:10]) + ("…" if len(names) > 10 else "")
            debug_hint = f" (debug: {debug_file})" if debug_file else ""
            if not seg_titles:
                # Hard fail: no pages at all. Preserve checkpoint (already saved
                # above) so the user can re-run to retry the failed segment.
                raise LLMOutputError(
                    f"Agentic generation for {seg_label} produced no page(s) for any of its "
                    f"{start_remaining} assigned concept(s) (passes={passes}, cap={cap}). "
                    f"Uncovered: [{sample}].{debug_hint} "
                    f"The model is not using the fetch_concepts tool; check its tool-calling support."
                )
            else:
                logger.warning(
                    "TwoPhase: agentic %s ended with %d/%d concept(s) still uncovered (passes=%d, cap=%d) — these will have no page. Uncovered: [%s].%s",
                    seg_label,
                    uncovered,
                    start_remaining,
                    passes,
                    cap,
                    sample,
                    debug_hint,
                )

    compiler._report(gen_progress_total, gen_progress_total, "generating complete")
    return nodes, slug_to_idx, caller_progress


def _page_kind(page: GenerationPage, batch: list[dict[str, Any]], extras: dict[str, str]) -> str:
    """Resolve a generated page's kind.

    A pre-assigned kind on the batch entry (entity pages) wins over the
    LLM's label — entity coverage is mandatory, not left to model discretion.
    """
    fixed = {normalize_concept_key(str(b.get("name", ""))): str(b.get("kind")) for b in batch if b.get("kind")}
    expected = fixed.get(normalize_concept_key(page.title or ""))
    return _allowed_kind(expected or page.kind, extras)


def _allowed_kind(kind: str, extras: dict[str, str]) -> str:
    """Validate an LLM-assigned page kind against the taxonomy.

    Mandatory kinds plus configured extras are accepted; anything else falls
    back to ``concept`` (permissive-consumption model, OKF §10).
    """
    if kind in MANDATORY_KINDS or kind in extras:
        return kind
    return "concept"


def _sanitize_entity_type(raw: str) -> str:
    """Normalize an entity type for use as a folder name ("Person" → "person")."""
    sanitized = slugify(raw or "", max_length=40)
    return sanitized or "entity"


def _categories_block(extras: dict[str, str]) -> str:
    """Render configured extra categories as prompt instruction lines."""
    return "\n".join(f'- "{name}" — {desc}'.rstrip(".") for name, desc in extras.items())


def assemble_wiki(
    compiler: TwoPhaseRuntime,
    nodes: list[WikiNode],
    slug_to_idx: dict[str, int],
    chunks: list[ResolvedChunk],
    extractions: ExtractionSet,
    *,
    concepts: list[dict[str, Any]],
    pinned_clean: list[str],
) -> CompiledWiki:
    """Filter, dedup, and assemble the final CompiledWiki."""
    cw = compiler._context_window

    if not nodes:
        raise LLMOutputError(
            f"Generation failed: 0 of {len(concepts)} concept(s) synthesized. "
            f"All LLM calls returned invalid data after retries. "
            f"Check that model '{compiler._model}' can produce structured JSON output."
        )

    compiler.merge_duplicate_nodes(
        nodes,
        compiler._config.embedding.model,
        threshold=compiler._config.budget.concept_merge_threshold,
    )
    compiler.disambiguate_duplicate_titles(nodes, chunks)
    nodes = compiler.filter_empty_nodes(nodes, min_content_length=compiler._config.budget.min_content_length)

    if not nodes:
        logger.warning(
            "TwoPhase: generation produced 0 pages after filtering (min_content_length=%d). Check LLM output quality.",
            compiler._config.budget.min_content_length,
        )
        return CompiledWiki(
            nodes=[],
            edges=[],
            toc=[],
            metadata={
                WikiField.CHUNK_COUNT: len(chunks),
                MetadataField.TWO_PHASE: True,
                MetadataField.GENERATION_SKIPPED: True,
                MetadataField.SKIP_REASON: "all pages filtered as empty",
                MetadataField.NODE_COUNT: 0,
            },
        )

    nodes = compiler.dedup_content_duplicate_nodes(nodes)
    attach_tags_to_nodes(nodes, chunks)
    attach_attributes_to_nodes(nodes, chunks)
    edges = build_edges(nodes, extractions, min_title_len=compiler._config.budget.edge_min_title_len)
    quality = compute_quality_score(nodes, edges)
    toc: list[TocEntry] = [{"title": n.title, "id": n.id, "level": 1} for n in nodes]
    aliases = learn_wikilink_aliases(nodes)
    if aliases:
        logger.info("TwoPhase: learned %d wikilink alias(es), e.g. %s", len(aliases), list(aliases.items())[:5])
    return CompiledWiki(
        nodes=nodes,
        edges=edges,
        toc=toc,
        quality_score=quality,
        metadata={
            WikiField.CHUNK_COUNT: len(chunks),
            MetadataField.NODE_COUNT: len(nodes),
            MetadataField.EDGE_COUNT: len(edges),
            MetadataField.TWO_PHASE: True,
            MetadataField.CONCEPTS_MERGED: len(concepts),
            MetadataField.CONTEXT_WINDOW: cw.max_input,
            MetadataField.PINNED_FACTS_COUNT: len(pinned_clean),
            MetadataField.WIKILINK_ALIASES: aliases,
        },
    )


def build_known_titles(compiler: TwoPhaseRuntime, concepts: list[dict[str, Any]]) -> list[str]:
    """Canonical page titles the generation LLM should link to verbatim.

    Combines the merged concept names from this compile with the canonical
    titles learned by the *previous* compile (its
    ``MetadataField.WIKILINK_ALIASES`` map, read from the on-disk wiki cache
    — still the previous compile's output at generation time, since the
    cache is written after generation). Deduplicated, order-preserving.
    """
    titles: list[str] = []
    seen: set[str] = set()

    def _add(title: str) -> None:
        if title and title not in seen:
            seen.add(title)
            titles.append(title)

    for c in concepts:
        _add(str(c.get(ExtractionField.NAME, "")))

    try:
        previous = load_cached_wiki(compiler._config.paths.base_dir)
    except Exception:  # noqa: BLE001 — cache read is best-effort; never fail compile
        previous = None
    if previous is not None:
        learned = previous.metadata.get(MetadataField.WIKILINK_ALIASES) or {}
        if isinstance(learned, dict):
            for canonical in learned.values():
                _add(str(canonical))

    return titles


def learn_wikilink_aliases(nodes: list[WikiNode]) -> dict[str, str]:
    """Learn an alias → canonical-title map from wikilinks in *nodes*.

    Runs :class:`BrokenWikilinksRule` over the freshly generated pages and
    collects every failure the rule tagged ``alias_of`` (a fuzzy/substring
    match against an existing page). The result persists in
    ``wiki.metadata[MetadataField.WIKILINK_ALIASES]`` and feeds
    :func:`build_known_titles` on the next compile, so the LLM is nudged
    toward canonical titles while the renderer resolves leftovers.
    """
    if not nodes:
        return {}
    wiki = CompiledWiki(nodes=nodes, edges=[], toc=[])
    id_to_title = {n.id: n.title for n in nodes}
    aliases: dict[str, str] = {}
    for result in BrokenWikilinksRule().check(wiki):
        if result.outcome != LintRuleOutcome.FAIL:
            continue
        alias_of = result.details.get("alias_of")
        canonical = id_to_title.get(str(alias_of))
        if canonical:
            aliases.setdefault(result.details.get("target", ""), canonical)
    aliases.pop("", None)
    return aliases


def gen_cache_deps(
    compiler: TwoPhaseRuntime,
    extractions: ExtractionSet,
    *,
    pinned_facts: list[str] | None = None,
) -> dict[str, str]:
    """Compute dependency hashes for generation cache."""
    blob = "|".join(json.dumps(e.to_dict(), sort_keys=True) for e in extractions.extractions)
    ext_hash = hashlib.sha256(blob.encode()).hexdigest()
    pin_blob = "\n".join(sorted(f.strip() for f in (pinned_facts or []) if f and f.strip()))
    pin_hash = hashlib.sha256(pin_blob.encode()).hexdigest()
    budget = compiler._budget
    deps: dict[str, str] = {}
    deps[CacheDeps.EXTRACTIONS] = ext_hash or ""
    deps[CacheDeps.MODEL] = compiler._model or ""
    deps[CacheDeps.CHUNK_SIZE_MIN] = str(budget.chunk_size_min or "")
    deps[CacheDeps.CHUNK_SIZE_MAX] = str(budget.chunk_size_max or "")
    deps[CacheDeps.PINNED_FACTS] = pin_hash or ""
    return deps
