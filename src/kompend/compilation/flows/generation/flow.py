"""Generation flow — per-concept page generation and wiki assembly.

Extracted from :class:`TwoPhaseCompiler` — generation phase methods
that operate on compiler instance state.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import Any

from pydantic_ai_harness.step_persistence import SqliteStepStore, StepStore

from kompend.agents.failure import AgentFailureKind, classify_agent_failure, record_unit_failure
from kompend.agents.generation import make_agentic_generation_agent
from kompend.agents.types import ToolGenerationDeps
from kompend.compilation.adaptation import ShrinkOutcome
from kompend.compilation.concept_broker import ConceptBroker
from kompend.compilation.concept_keys import normalize_concept_key
from kompend.compilation.flows.assembly import (
    merge_similar_concepts,
)
from kompend.compilation.flows.generation.budget import (
    _MAX_CONTEXT_CLAMPS,
    _agentic_budget,
    _known_titles_windows,
    _token_estimate,
)
from kompend.compilation.flows.generation.prompt import _categories_block, build_known_titles
from kompend.compilation.flows.generation.segments import (
    _PASS_CONTINUATION_KINDS,
    _CoverageCredit,
    _failure_action,
    _format_uncovered_concepts,
    _SegmentUnit,
    _shrink_and_requeue,
    prepare_source_segments,
)
from kompend.compilation.flows.generation.writer import (
    _SOURCE_WIKILINK_RE,
    PlaceholderPageError,
    _reject_placeholder_pages,
    _resolve_page_kind,
    _sanitize_entity_type,
    assemble_wiki,
)
from kompend.compilation.schema import GenerationOutput
from kompend.compilation.source_search import SourceIndex
from kompend.compilation.step_recovery import (
    SegmentRecovery,
    extract_final_output,
    generation_conversation,
    pass_run_id,
    scan_prefix,
    segment_run_prefix,
)
from kompend.compilation.two_phase import (
    ExtractionSet,
    TwoPhaseRuntime,
    extract_images_from_chunks,
    rate_suffix,
    slugify,
    upsert_node,
)
from kompend.core.models import ChunkTree, CompiledWiki, ImageRef, ResolvedChunk, WikiField, WikiNode, images_for_provenance
from kompend.core.models.enums import CacheDeps, ExtractionField, MetadataField, NodeType
from kompend.core.pipeline.entity_types import EntityTypeRegistry
from kompend.core.templates import get_template_loader
from kompend.core.utils.exceptions import (
    LLMContextOverflowError,
    LLMOutputError,
    LLMTransientError,
)
from kompend.llm.decision import ErrorAction
from kompend.llm.request_budget import RequestBudget
from kompend.logging import get_logger

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
    pinned_clean = list(dict.fromkeys(f.strip() for f in (pinned_facts or []) if f and f.strip()))
    # Window the titles FIRST so the reservation and the prompt itself are
    # computed from the same head. The full list stays intact in deps and is
    # reachable through list_known_titles() — the head is a window, not a cut.
    known_titles = build_known_titles(compiler, concepts)
    titles_prompt, titles_page_chars = _known_titles_windows(compiler, known_titles)
    # Agentic-conversation fixed cost (everything that lands in the request
    # that is NOT the segment content): system template + rendered user
    # prompt (template + known_titles + pinned_facts) + tool-call framing
    # (read_segment + fetch_concepts round-trips) + the fetch result. The
    # planner adds the answer reserve and the safety margin, so the full
    # request (segment content + output budget) stays inside the context.
    budget = _agentic_budget(compiler, titles_prompt, titles_page_chars, pinned_clean)
    source_segments = prepare_source_segments(compiler, chunks, budget=budget)

    # The provider may enforce a smaller window than kompend planned against,
    # and it may name a still smaller one on a later attempt (a failover model
    # with a narrower context). Every rejection that names a lower limit lowers
    # the window, re-windows the titles, recomputes the segments and runs again -
    # the content is split across more segments, never cropped. The loop ends
    # when the provider repeats a limit kompend already plans against
    # (``_clamp_window`` refuses to move) or the cap is reached.
    clamps = 0
    while True:
        try:
            nodes, slug_to_idx, _caller_progress = run_generation_resumable(
                compiler,
                concepts,
                source_segments,
                all_images,
                pinned_clean,
                known_titles=known_titles,
                known_titles_prompt=titles_prompt,
                known_titles_page_chars=titles_page_chars,
                budget=budget,
            )
            break
        except LLMContextOverflowError as exc:
            if (
                exc.stated_limit is None
                or clamps >= _MAX_CONTEXT_CLAMPS
                or not compiler._clamp_window(exc.stated_limit, reason=f"provider enforces {exc.stated_limit} tokens")
            ):
                raise
            clamps += 1
            # The titles windows are a fraction of the window; after the clamp
            # they must shrink with it, or the prompt keeps paying the old
            # window's share and the re-segmentation gains nothing. The full
            # list is untouched — only the window sizes move.
            titles_prompt, titles_page_chars = _known_titles_windows(compiler, known_titles)
            budget = _agentic_budget(compiler, titles_prompt, titles_page_chars, pinned_clean)
            source_segments = prepare_source_segments(compiler, chunks, budget=budget)
        except LLMOutputError:
            # Generation produced no pages for at least one segment. The step
            # store keeps every segment transcript; a re-run recovers finished
            # segments and resumes the interrupted one mid-conversation.
            logger.warning(
                "TwoPhase: agentic generation raised — transcripts persisted. Re-run to retry the failed segment(s) only.",
            )
            raise

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
        max_reduction=budget.concept_merge_max_reduction,
        max_sources=budget.concept_merge_max_sources,
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
    """Append entity pages that pass the per-type source gate.

    An entity spec reaches generation only when it spans at least
    ``compile.budget.entity_min_sources`` distinct sources, or the override
    for its ``type``. Entities below the gate stay folded into their source
    summary pages — their mentions are still part of the source content.
    """
    concept_keys = {normalize_concept_key(str(c.get(ExtractionField.NAME, ""))) for c in concepts}
    budget = compiler._config.budget
    thresholds = {str(kind).strip().casefold(): minimum for kind, minimum in budget.entity_min_sources_by_type.items()}
    entities: list[dict[str, Any]] = []
    folded = 0
    for entity in extractions.entity_pages():
        if normalize_concept_key(str(entity.get(ExtractionField.NAME, ""))) in concept_keys:
            continue
        entity_type = str(entity.get(ExtractionField.TYPE, "")).strip().casefold()
        required = thresholds.get(entity_type, budget.entity_min_sources)
        if len(entity.get("sources", [])) < required:
            folded += 1
            continue
        entities.append(entity)
    if folded:
        logger.info("TwoPhase: %d entity page(s) below the source gate folded into source summaries", folded)
    if entities:
        logger.info("TwoPhase: %d entity page(s) queued for generation", len(entities))
        concepts.extend(entities)
    return concepts


def run_generation_resumable(
    compiler: TwoPhaseRuntime,
    concepts: list[dict[str, Any]],
    source_segments: list[list[tuple[str, str]]],
    all_images: list[ImageRef],
    pinned_clean: list[str],
    *,
    known_titles: list[str] | None = None,
    known_titles_prompt: list[str] | None = None,
    known_titles_page_chars: int = 0,
    budget: RequestBudget | None = None,
) -> tuple[list[WikiNode], dict[str, int], dict[str, list[str]]]:
    """Run agentic generation with transcript-level resume.

    Every segment agent run is persisted to the step store
    (``.kompend/steps.db``) by the harness ``StepPersistence`` capability.
    On entry, finished segments are recovered from the store without a
    single LLM call, an interrupted segment continues its own conversation
    where the process died, and only segments without any persisted
    conversation start fresh. The store is keyed by the generation deps
    (extraction set, model, sizes, pinned facts): changed inputs open a
    new conversation instead of mixing old results into a new wiki.
    """
    store = SqliteStepStore(database=compiler._config.paths.step_db_file)
    if not compiler._active_gen_deps:
        logger.warning("TwoPhase: no generation deps set — step-store resume disabled for this run.")
    return generate_all_pages_agentic(
        compiler,
        concepts,
        source_segments,
        all_images,
        pinned_facts=pinned_clean,
        known_titles=known_titles,
        known_titles_prompt=known_titles_prompt,
        known_titles_page_chars=known_titles_page_chars,
        budget=budget,
        step_store=store,
    )


def generate_all_pages_agentic(  # noqa: PLR0915 — per-segment loop + recovery; long by design
    compiler: TwoPhaseRuntime,
    concepts: list[dict[str, Any]],
    source_segments: list[list[tuple[str, str]]],
    all_images: list[ImageRef],
    *,
    pinned_facts: list[str],
    known_titles: list[str] | None = None,
    known_titles_prompt: list[str] | None = None,
    known_titles_page_chars: int = 0,
    budget: RequestBudget | None = None,
    step_store: StepStore | None = None,
) -> tuple[list[WikiNode], dict[str, int], dict[str, list[str]]]:
    """Generate wiki pages using per-segment agentic tool loop.

    Replaces the nested ``for batch × for segment`` loop with per-segment
    agent runs.  Each agent discovers concepts via ``fetch_concepts()`` and
    loads source content via ``read_segment()`` tools, deciding batch size
    and coverage strategy itself.

    Uses :class:`ConceptBroker` for server-side concept assignment and
    ``Memory`` + ``SlidingWindowCompaction`` + ``ClampOversizedMessages``
    as agent capabilities for cross-segment judgment.

    When *step_store* is set, every pass is persisted
    (``StepPersistence``) under the generation deps as the conversation id
    and a deterministic per-pass run id: finished segments are recovered
    from the store, an interrupted segment resumes its conversation, and
    each pass gets ``search_conversation_history`` over its own transcript
    (``ConversationSearch``) so compaction cannot erase what the model
    already produced.

    Returns ``(nodes, slug_to_idx, caller_progress)``.
    """
    nodes: list[WikiNode] = []
    slug_to_idx: dict[str, int] = {}
    caller_progress: dict[str, list[str]] = {}
    num_source_segments = len(source_segments)
    num_concepts = len(concepts)
    # One segment owns at most ``budget.concepts_per_segment`` concepts: every
    # pass of the agent loop below re-sends the segment content as context, so
    # a segment that owns a whole document's concept list pays one full prefill
    # per batch of pages. Extra segments reuse the source content round-robin -
    # content is read through the read_segment() tool, never cropped.
    per_segment = max(0, int(compiler._config.budget.concepts_per_segment))
    if per_segment and num_source_segments:
        wanted = -(-num_concepts // per_segment)
        segment_plan = [source_segments[i % num_source_segments] for i in range(max(num_source_segments, wanted))]
    else:
        segment_plan = list(source_segments)
    num_segments = len(segment_plan)

    broker = ConceptBroker(concepts, num_callers=num_segments)
    # Transcript identity: the generation deps key (extraction set, model,
    # sizes, pinned facts). Same inputs → same conversation → finished
    # segments recover from the step store; changed inputs open a fresh
    # conversation instead of mixing old results into the new wiki.
    step_conversation = generation_conversation(compiler._active_gen_deps) if step_store is not None and compiler._active_gen_deps else None
    resume_enabled = step_conversation is not None and not compiler._config.no_cache
    if step_store is not None and step_conversation is None:
        logger.warning("TwoPhase: step store present but no generation deps — transcripts will not be persisted.")
    categories = _categories_block(compiler._config.extraction_categories)

    entity_types_registry = EntityTypeRegistry(compiler._config.paths.entity_types_file)
    entity_types: dict[str, str] = {}
    overrides_applied = 0
    for c in concepts:
        if c.get("kind") != NodeType.ENTITY.value:
            continue
        name = str(c.get(ExtractionField.NAME, ""))
        key = normalize_concept_key(name)
        if not key:
            continue
        sanitized_model_choice = _sanitize_entity_type(str(c.get("type", "")))
        registered = entity_types_registry.get(key)
        if registered is not None:
            # Registry wins — surface the override count for diagnostics.
            entity_types[key] = registered
            if registered != sanitized_model_choice:
                overrides_applied += 1
        else:
            entity_types[key] = sanitized_model_choice
    if overrides_applied:
        logger.info(
            "Entity-type registry overrode model choice for %d concept(s)",
            overrides_applied,
        )

    # Deterministic kind for pre-assigned concept entries (mandatory entity
    # pages). A model that drops or alters the kind must not misfile these;
    # pages the model invents keep their own kind.
    preassigned_kinds: dict[str, str] = {normalize_concept_key(str(c.get(ExtractionField.NAME, ""))): str(c["kind"]) for c in concepts if c.get("kind")}
    # Coverage credit: rapidfuzz T2 + embedding T3 (single instances shared
    # across segments so title/concept embeddings are cached compile-wide).
    coverage = _CoverageCredit(compiler._config.embedding.model)
    # Segment-scoped generation is closed-world by default: the agent sees its
    # own segment and the concepts assigned to it. The index gives it the
    # reach a researcher has - name a concept, find the sources that mention
    # it, then read them in windows. Built once per run from the same
    # segments the plan slices up, so it costs one token scan per source.
    source_index = SourceIndex.from_segments(
        source_segments,
        embedding_model=compiler._config.embedding.model,
        decision=compiler._config.decision,
    )

    def _absorb_pages(gen_output: GenerationOutput, seg_idx: int) -> set[str]:
        """Upsert a segment's produced pages into ``nodes``; return their titles."""
        titles: set[str] = set()
        for idx, page in enumerate(gen_output.pages):
            if not (page.content or "").strip():
                continue
            titles.add(normalize_concept_key(page.title or ""))
            node_id = slugify(page.title or f"concept-{seg_idx}-{idx}", max_length=compiler._config.llm.slug_max_length)
            kind = _resolve_page_kind(page.kind, page.title or "", preassigned_kinds, compiler._config.extraction_categories)
            page_content = _SOURCE_WIKILINK_RE.sub(r"[\1]", page.content)
            node_metadata: dict[str, Any] = {
                MetadataField.PROVENANCE: page.provenance,
                MetadataField.RELATED: page.related,
                MetadataField.TWO_PHASE: True,
                # The model that wrote THIS page - survives cache hits, so a
                # mixed-model wiki stays attributable per page (and a page
                # can be traced back to (and recompiled by) a better model).
                MetadataField.MODEL: compiler._model or "",
                # Only this page's sources' figures - see images_for_provenance.
                MetadataField.IMAGES: images_for_provenance(all_images, page.provenance),
                MetadataField.FRONTMATTER: {
                    "title": page.title or f"Concept {idx}",
                },
            }
            if kind is NodeType.ENTITY:
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
        return titles

    logger.info(
        "TwoPhase: agentic generation — %d concept(s), %d segment(s)",
        num_concepts,
        num_segments,
    )
    gen_progress_total = num_concepts
    pages_done = 0
    if num_segments > 0:
        compiler._report(0, gen_progress_total, f"generating wiki ({num_concepts} concepts, {num_segments} segments)")

    # Loop-1 work queue: a unit that overflows the window is halved and its
    # halves re-queued in its place, so the loop below processes strictly
    # smaller units until every concept is paged or the adaptation budget
    # (``budget.segment_shrink_attempts``) is spent.
    max_splits = max(0, int(compiler._config.budget.segment_shrink_attempts))
    pending: deque[_SegmentUnit] = deque(
        _SegmentUnit(
            label=f"segment {i + 1}/{num_segments}",
            seg_id=ConceptBroker.caller_id(i),
            segment=segment,
            run_prefix=segment_run_prefix(step_conversation, i) if step_conversation else None,
            seg_idx=i,
            depth=0,
        )
        for i, segment in enumerate(segment_plan)
    )
    while pending:
        unit = pending.popleft()
        seg_label, seg_id, segment, seg_prefix, seg_idx, depth = unit
        # Transcript recovery: a finished unit replays its pages without an
        # LLM call; an interrupted one resumes its own conversation.
        # Sub-units carry derived run prefixes and recover like their parent.
        recovery = (
            scan_prefix(step_store, step_conversation, seg_prefix)
            if step_store is not None and resume_enabled and seg_prefix is not None
            else SegmentRecovery(snapshot=None, interrupted=False, attempts=0)
        )
        if recovery.snapshot is not None and not recovery.interrupted:
            recovered = extract_final_output(recovery.snapshot.messages, GenerationOutput)
            # Absorb first: a transcript whose pages all carry empty content
            # recovers nothing — the segment runs fresh instead of being
            # skipped with zero pages. The same ingestion gate as the live
            # path applies: a transcript whose final_result carries
            # placeholder pages must not replay them into the wiki — the
            # segment re-runs fresh and the rejection is recorded loudly.
            seg_titles = set()
            rejected_recovery = False
            if recovered is not None:
                try:
                    _reject_placeholder_pages(recovered, unit=seg_label)
                except PlaceholderPageError as exc:
                    rejected_recovery = True
                    record_unit_failure(
                        stage="agentic-generation-recovery",
                        unit=seg_id,
                        exc=exc,
                        attempts=recovery.attempts,
                    )
                else:
                    seg_titles = _absorb_pages(recovered, seg_idx)
            if seg_titles:
                broker.mark_caller_consumed(seg_id, seg_titles)
                caller_progress[seg_id] = sorted(seg_titles)
                pages_done = min(pages_done + len(seg_titles), gen_progress_total)
                logger.info(
                    "TwoPhase: agentic %s recovered from the step store — skipping (%d page(s), attempt %d).",
                    seg_label,
                    len(seg_titles),
                    recovery.attempts,
                )
                compiler._report(
                    pages_done,
                    gen_progress_total,
                    generation_detail(
                        pages_done,
                        gen_progress_total,
                        note="(recovered)",
                        rate=rate_suffix(compiler._usage_tracker),
                    ),
                )
                continue
            if rejected_recovery:
                logger.info(
                    "TwoPhase: agentic %s replayed a transcript carrying rejected placeholder page(s) — re-running fresh.",
                    seg_label,
                )
            else:
                logger.info(
                    "TwoPhase: agentic %s has a finished transcript without a readable result — re-running fresh.",
                    seg_label,
                )
            recovery = SegmentRecovery(snapshot=None, interrupted=False, attempts=recovery.attempts)
        resume_history = recovery.history if recovery.snapshot is not None else None
        if resume_history is not None:
            logger.info(
                "TwoPhase: agentic %s resuming interrupted conversation (attempt %d).",
                seg_label,
                recovery.attempts,
            )
        compiler._report(pages_done, gen_progress_total, generation_detail(pages_done, gen_progress_total))
        logger.info("TwoPhase: agentic %s", seg_label)

        content_block = "\n\n".join(f"### {label}\n{content}" for label, content in segment)

        agent = make_agentic_generation_agent(
            model_string=compiler._model or "",
            timeout=compiler._timeout,
            skills=compiler._config.skills,
            delta_retry_enabled=compiler._config.llm.delta_retry_enabled,
            reasoning_effort=compiler._config.llm.reasoning_effort,
            context_manager=compiler._config.llm.context_manager,
            context_window=compiler._context_window.max_input,
            rlm=compiler._config.llm.rlm,
            usage_tracker=compiler._usage_tracker,
            llm=compiler._config.llm,
            step_store=step_store,
        )

        deps = ToolGenerationDeps(
            source_content=content_block,
            source_id=seg_id,
            pinned_facts=pinned_facts,
            categories=categories,
            known_titles=known_titles or [],
            known_titles_prompt=known_titles_prompt or [],
            known_titles_page_chars=known_titles_page_chars,
            concept_broker=broker,
            source_index=source_index,
        )

        # Force full concept coverage: small models often stop after the first
        # fetch_concepts batch instead of looping until the broker is drained.
        # Re-run the segment agent until its assigned concepts are exhausted,
        # bounded by a hard cap (one pass per concept maximum).
        start_remaining = broker.remaining(seg_id)
        cap = start_remaining + 1
        passes = 0
        gen_output = None
        last_error: str | None = None
        seg_start_nodes = len(nodes)
        seg_titles: set[str] = set()  # normalized page titles produced for this segment
        seg_consumed_names: list[str] = list(caller_progress.get(seg_id, []))  # recovered from the transcript store
        # Request-fit guard: the segment content is placed in the conversation
        # by read_segment (a tool result), and the model still needs room for
        # its answer (context_window.answer_reserve) on top. If even ONE call
        # would exceed the window,
        # fail loudly now instead of letting Ollama stall/time out mid-stream.
        # The no-crop rule is honored: we never silently clamp max_tokens;
        # content is preserved by splitting into smaller segments at the
        # sizing stage.
        _segment_content_tokens = _token_estimate(content_block, compiler._config.llm.default_chars_per_token)
        segment_budget = budget or RequestBudget(
            window=compiler._context_window,
            reserved={"system": compiler._gen_system_tokens},
        )
        segment_budget.require(
            _segment_content_tokens,
            stage=f"Agentic generation for {seg_label}",
        )
        while broker.remaining(seg_id) > 0 and passes < cap:
            # One pass = one agent run. The agent base owns the whole-request
            # retry policy (``compile.llm.transient_retries``), so a transient
            # failure only reaches this loop after those retries are spent.
            passes += 1
            last_error = None
            action = ErrorAction.SKIP
            retry_pass = False
            try:
                gen_output = agent.run(
                    deps,
                    message_history=resume_history,
                    user_prompt=None if resume_history is None else _loader.render("compilation/generation_resume.md.j2"),
                    conversation_id=step_conversation,
                    run_id=pass_run_id(seg_prefix, recovery.attempts + 1, passes) if seg_prefix else None,
                )
                resume_history = None  # only the first pass continues the old conversation
                # Ingestion gate: a placeholder page must fail the unit on
                # the existing failure path, never reach the wiki.
                _reject_placeholder_pages(gen_output, unit=seg_label)
            except LLMContextOverflowError as exc:
                # The unit did not fit the provider's window. Loop 1 earns
                # it a smaller re-plan; only a unit that cannot shrink keeps
                # the loud failure the error carries.
                outcome = _shrink_and_requeue(
                    broker,
                    coverage,
                    compiler,
                    pending,
                    unit,
                    start_remaining=start_remaining,
                    seg_titles=seg_titles,
                    max_splits=max_splits,
                    kind=AgentFailureKind.CONTEXT_OVERFLOW,
                    err=str(exc),
                )
                if outcome is ShrinkOutcome.SPLIT:
                    gen_output = None
                    break
                if outcome is ShrinkOutcome.ABORT:
                    raise LLMOutputError(f"Agentic generation aborted for '{seg_id}': {exc}") from exc
                if outcome is ShrinkOutcome.SKIP:
                    gen_output = None
                    last_error = str(exc)
                    break
                raise
            except (LLMTransientError, LLMOutputError) as exc:
                kind = classify_agent_failure(exc)
                action = _failure_action(
                    compiler,
                    exc,
                    stage=f"agentic-generation:{seg_id}",
                    attempt=passes,
                    max_attempts=cap,
                    transient=kind is AgentFailureKind.TRANSIENT,
                )
                record_unit_failure(stage="agentic-generation", unit=seg_id, exc=exc, attempts=passes)
                if action is ErrorAction.ABORT:
                    raise LLMOutputError(f"Agentic generation aborted for '{seg_id}': {exc}") from exc
                gen_output = None
                last_error = str(exc)
                outcome = _shrink_and_requeue(
                    broker,
                    coverage,
                    compiler,
                    pending,
                    unit,
                    start_remaining=start_remaining,
                    seg_titles=seg_titles,
                    max_splits=max_splits,
                    kind=kind,
                    err=str(exc),
                )
                if outcome is ShrinkOutcome.SPLIT:
                    break
                if outcome is ShrinkOutcome.ABORT:
                    raise LLMOutputError(f"Agentic generation aborted for '{seg_id}': {exc}") from exc
                retry_pass = kind in _PASS_CONTINUATION_KINDS or (kind is AgentFailureKind.TRANSIENT and action is ErrorAction.RETRY)
                if outcome is ShrinkOutcome.SKIP:
                    retry_pass = False
            except Exception as exc:  # PydanticAI raises many subclassed errors; treat uniformly
                kind = classify_agent_failure(exc)
                record_unit_failure(stage="agentic-generation", unit=seg_id, exc=exc, attempts=passes)
                gen_output = None
                last_error = str(exc)
                if kind is AgentFailureKind.TRANSIENT:
                    action = _failure_action(
                        compiler,
                        exc,
                        stage=f"agentic-generation:{seg_id}",
                        attempt=passes,
                        max_attempts=cap,
                        transient=True,
                    )
                    if action is ErrorAction.ABORT:
                        raise LLMOutputError(f"Agentic generation aborted for '{seg_id}': {exc}") from exc
                    retry_pass = action is ErrorAction.RETRY
                else:
                    retry_pass = kind in _PASS_CONTINUATION_KINDS
                outcome = _shrink_and_requeue(
                    broker,
                    coverage,
                    compiler,
                    pending,
                    unit,
                    start_remaining=start_remaining,
                    seg_titles=seg_titles,
                    max_splits=max_splits,
                    kind=kind,
                    err=str(exc),
                )
                if outcome is ShrinkOutcome.SPLIT:
                    break
                if outcome is ShrinkOutcome.ABORT:
                    raise LLMOutputError(f"Agentic generation aborted for '{seg_id}': {exc}") from exc
                if outcome is ShrinkOutcome.SKIP:
                    retry_pass = False
            debug_file = getattr(agent, "_last_debug_file", None)

            compiler._batch_tracker.record(failed=gen_output is None)

            if gen_output is None:
                logger.warning(
                    "TwoPhase: agentic %s (caller_id=%s, pass=%d/%d, remaining=%d) failed — %s (debug: %s)",
                    seg_label,
                    seg_id,
                    passes,
                    cap,
                    broker.remaining(seg_id),
                    "retrying" if retry_pass else "skipping",
                    debug_file or "(none)",
                )
                # Stranded work check: fetched-but-unpaged concepts sit in
                # the consumed region past the cursor — put them back at the
                # front so the next pass (or the tail below) accounts for
                # every assigned concept instead of reporting an empty pool
                # for a unit that did work.
                stranded = [c for c in broker.consumed(seg_id) if not coverage.covered(str(c.get(ExtractionField.NAME, "")), seg_titles)]
                if stranded:
                    broker.requeue(seg_id, stranded)
                if retry_pass:
                    continue
                break

            empty_pages = [p for p in gen_output.pages if not (p.content or "").strip()]
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

            for title_key in _absorb_pages(gen_output, seg_idx):
                seg_titles.add(title_key)

            # Requeue fetched-but-uncovered concepts: the model may fetch a
            # batch and only produce pages for part of it. Coverage credit
            # routes through _CoverageCredit (rapidfuzz T2 + embedding T3)
            # so titles like "P.10 G.100 Recommendation" credit concept
            # "ITU-T P.10/G.100" instead of burning requeue passes.
            uncovered_fetched = [c for c in broker.consumed(seg_id) if not coverage.covered(str(c.get(ExtractionField.NAME, "")), seg_titles)]
            if uncovered_fetched:
                broker.requeue(seg_id, uncovered_fetched)

            # Report page-based progress: pages produced for this segment,
            # bounded by its broker share (prior segments count fully).
            # pages_done itself must NEVER exceed gen_progress_total, so
            # we cap the running value to avoid the bar overflowing
            # ``[N/338]`` when the model under-produces for a segment.
            # The bar carries two numbers and no labels: the pages done
            # overall, and the pages done inside the current segment.
            seg_progress = min(len(seg_titles), start_remaining)
            running = min(pages_done + seg_progress, gen_progress_total)
            rate_s = rate_suffix(compiler._usage_tracker)
            compiler._report(
                running,
                gen_progress_total,
                generation_detail(running, gen_progress_total, seg_progress, start_remaining, rate=rate_s),
            )

        # Credit the segment with the pages actually produced, not the
        # broker share (which assumes full coverage and would over-credit
        # when the model fails to page every fetched concept).
        pages_done = min(pages_done + len(seg_titles), gen_progress_total)

        # caller_progress records the consumed concept names for this run's
        # return value (the transcripts in the step store are the durable
        # resume record).
        produced_names = sorted(seg_titles)
        seg_consumed_names.extend(produced_names)
        caller_progress[seg_id] = list(dict.fromkeys(seg_consumed_names))  # dedupe, preserve order

        uncovered = broker.remaining(seg_id)
        if uncovered > 0:
            len(nodes) - seg_start_nodes
            sample = _format_uncovered_concepts(broker, seg_id, limit=10)
            debug_hint = f" (debug: {debug_file})" if debug_file else ""
            if not seg_titles:
                # Hard fail: no pages at all. The transcript is already in
                # the step store, so re-running retries only this segment.
                # Distinguish a model that never called the tool (tool-call
                # support problem) from one that fetched and then the request
                # died (timeout/overflow — a sizing or provider problem).
                fetched = broker.consumed(seg_id)
                if fetched:
                    cause = f"it called fetch_concepts {len(fetched)}× but the request failed: {last_error}"
                    hint = "This is not a tool-calling problem — check the request size vs the model's context window."
                elif last_error:
                    cause = f"the request failed before it called fetch_concepts: {last_error}"
                    hint = (
                        "The tool was never reached — check the request size vs the model's context window "
                        "(a failed context compaction surfaces here) before suspecting tool-calling support."
                    )
                else:
                    cause = "it never called the fetch_concepts tool"
                    hint = "The model is not using the fetch_concepts tool; check its tool-calling support."
                raise LLMOutputError(
                    f"Agentic generation for {seg_label} (caller_id={seg_id}) produced no page(s) for any of its "
                    f"{start_remaining} assigned concept(s) (passes={passes}, cap={cap}): {cause}. "
                    f"caller_progress={caller_progress.get(seg_id, [])}. "
                    f"Uncovered: [{sample}].{debug_hint} {hint}"
                )
            else:
                logger.warning(
                    "TwoPhase: agentic %s (caller_id=%s) ended with %d/%d concept(s) still uncovered "
                    "(passes=%d, cap=%d) — these will have no page. "
                    "produced_for_this_seg=%d. Uncovered: [%s].%s",
                    seg_label,
                    seg_id,
                    uncovered,
                    start_remaining,
                    passes,
                    cap,
                    len(seg_titles),
                    sample,
                    debug_hint,
                )

    # Merge Loop-1 sub-unit progress back into the parent segment keys, so
    # the run's return value keeps one entry per segment. The transcripts
    # stay per-unit in the step store — that is the durable resume record.
    for i in range(num_segments):
        parent = ConceptBroker.caller_id(i)
        subs = sorted(k for k in caller_progress if k.startswith(parent + "/"))
        if subs:
            merged = list(dict.fromkeys(caller_progress.get(parent, []) + [n for k in subs for n in caller_progress[k]]))
            caller_progress[parent] = merged
            for k in subs:
                del caller_progress[k]

    compiler._report(gen_progress_total, gen_progress_total, "generating complete")
    return nodes, slug_to_idx, caller_progress


def generation_detail(
    pages_done: int,
    pages_total: int,
    segment_done: int | None = None,
    segment_total: int | None = None,
    note: str = "",
    rate: str = "",
) -> str:
    """The generation status line for the progress bar.

    Two counters, no labels: the pages done over all, and — while a
    segment runs — the pages of that segment::

        generating [57/338] (4/12) · 12.3 tok/s

    A counter without its denominator ("segment 2") and a second unit
    ("4/12 concepts covered") next to a page counter are noise; the
    numbers speak for themselves. A *note* names a state, not a count
    ("recovered"), and stands on its own.
    """
    detail = f"generating [{pages_done}/{pages_total}]"
    if segment_done is not None and segment_total is not None:
        detail += f" ({segment_done}/{segment_total})"
    if note:
        detail += f" {note}"
    return f"{detail}{rate}"


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
    if not compiler._config.cache_ignore_model:
        deps[CacheDeps.MODEL] = compiler._model or ""
    deps[CacheDeps.CHUNK_SIZE_MIN] = str(budget.chunk_size_min or "")
    deps[CacheDeps.CHUNK_SIZE_MAX] = str(budget.chunk_size_max or "")
    deps[CacheDeps.PINNED_FACTS] = pin_hash or ""
    return deps
