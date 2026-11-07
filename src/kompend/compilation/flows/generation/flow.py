"""Generation flow — per-concept page generation and wiki assembly.

Extracted from :class:`TwoPhaseCompiler` — generation phase methods
that operate on compiler instance state.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from enum import StrEnum
from typing import Any, Final, NamedTuple

from pydantic_ai_harness.step_persistence import SqliteStepStore, StepStore

from kompend.agents.decision import decide_error_action, decide_unit_adaptation
from kompend.agents.failure import AgentFailureKind, EmptyAnswerError, classify_agent_failure, record_unit_failure
from kompend.agents.generation import make_agentic_generation_agent
from kompend.agents.types import ToolGenerationDeps
from kompend.compilation.adaptation import AdaptationSignal, ShrinkOutcome, plan_adaptation, split_unit
from kompend.compilation.concept_broker import ConceptBroker
from kompend.compilation.concept_keys import normalize_concept_key
from kompend.compilation.flows.assembly import (
    attach_attributes_to_nodes,
    attach_tags_to_nodes,
    build_edges,
    merge_similar_concepts,
)
from kompend.compilation.flows.generation.budget import (
    _MAX_CONTEXT_CLAMPS,
    _agentic_budget,
    _known_titles_windows,
    _token_estimate,
)
from kompend.compilation.flows.generation.prompt import _categories_block, build_known_titles
from kompend.compilation.schema import GenerationOutput, GenerationPage
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
    compute_quality_score,
    extract_images_from_chunks,
    rate_suffix,
    segment_sources,
    slugify,
    tokens_to_chars,
    upsert_node,
)
from kompend.core.keys import MANDATORY_KINDS
from kompend.core.models import ChunkTree, CompiledWiki, ImageRef, ResolvedChunk, TocEntry, WikiField, WikiNode, images_for_provenance
from kompend.core.models.enums import CacheDeps, ExtractionField, MetadataField, NodeType
from kompend.core.paths import safe_stem
from kompend.core.pipeline.entity_types import EntityTypeRegistry
from kompend.core.templates import get_template_loader
from kompend.core.text.matching import best_match, cosine
from kompend.core.utils.exceptions import (
    LLMContextOverflowError,
    LLMOutputError,
    LLMTransientError,
)
from kompend.lint.base import LintRuleOutcome
from kompend.lint.structural import BrokenWikilinksRule
from kompend.llm.decision import ErrorAction, ErrorDecision
from kompend.llm.request_budget import RequestBudget
from kompend.logging import LOG_SAMPLE_LIMIT, get_logger
from kompend.rag.embedding_pydantic import make_provider

# Matches wikilink-wrapped source citations like [[sources/foo.pdf]] so they
# can be normalized to the intended [sources/foo.pdf] inline citation format
# regardless of LLM compliance with the system prompt.
_SOURCE_WIKILINK_RE = re.compile(r"\[\[\s*(sources/[^\]\n]+?)\s*\]\]")


#: rapidfuzz-scale (0–100) threshold for "does a produced page title cover
#: this concept" (T2 string tier).
_COVER_SIMILARITY_THRESHOLD = 85.0


#: A page title shorter than this many characters (a bare "x") is a
#: placeholder identity, never a real page title. Two characters ("3D") pass.
_MIN_PAGE_TITLE_CHARS: Final[int] = 2


#: Failure kinds that end one pass but not the segment. An empty answer (or
#: a request the provider refused for size) says nothing about the next
#: pass: that pass carries different material — the broker's remaining
#: concepts and a fresh prompt — so the segment keeps asking instead of
#: reporting one empty pass as a failed segment.
_PASS_CONTINUATION_KINDS: Final[frozenset[AgentFailureKind]] = frozenset({AgentFailureKind.EMPTY_ANSWER, AgentFailureKind.CONTEXT_OVERFLOW})


#: How many retries to attempt when the upstream returns a transient 4xx
#: with a "timed out" body (typically the model gateway routing hit a
#: non-tool-capable or temporarily slow upstream, e.g. omniroute ``auto``
#: roaming onto ``jr/agent``). Each retry is exponential backoff from
#: :data:`_UPSTREAM_TRANSIENT_BACKOFF_BASE`. Retries do NOT consume a pass
#: slot of the segment agent's coverage loop.
_UPSTREAM_TRANSIENT_MAX_RETRIES: int = 3

#: Initial backoff in seconds for transient upstream timeouts. Doubled per
#: attempt: 1s, 2s, 4s. Total worst-case wait: 7s before surfacing the
#: error and moving to the next capability in the fallback chain.
_UPSTREAM_TRANSIENT_BACKOFF_BASE: float = 1.0


#: Cosine threshold for the T3 semantic tier — catches paraphrases and
#: punctuation variants the string tier misses (e.g. "ITU-T P.10/G.100" vs
#: "P.10 G.100 Recommendation"). Conservative: a false *credit* silently
#: drops a concept's page, so the bar sits above lint's topic-vs-chunk 0.55.
_COVER_SEMANTIC_THRESHOLD = 0.65


#: Failure kinds whose remedy is a judgment call: Loop 2 may be consulted
#: when the deterministic policy declines them. Transport failures and
#: provider overflows are arithmetic — the model is never asked about those.
_AMBIGUOUS_KINDS: Final[frozenset[AgentFailureKind]] = frozenset({AgentFailureKind.SCHEMA_MISMATCH, AgentFailureKind.OTHER, AgentFailureKind.EMPTY_ANSWER})

logger = get_logger(__name__)

_loader = get_template_loader()


class PlaceholderMarker(StrEnum):
    """Filler phrases that mark a page body as a placeholder.

    The two-phase writer can self-test its ``final_result`` tool and submit
    filler instead of a summary: a run recorded in ``steps.db`` accepted a
    page whose whole body was "Placeholder removed", and a reviewed wiki
    carried a page body announcing itself as prose written to test
    submission. None of these phrases occur in real source material, so a
    body containing one is a placeholder, never a summary.
    """

    #: The observed artifact body, verbatim.
    PLACEHOLDER_REMOVED = "placeholder removed"
    #: The classic typesetter filler.
    LOREM_IPSUM = "lorem ipsum"
    #: Announces the body as standing text rather than written prose.
    PLACEHOLDER_TEXT = "placeholder text"
    #: Announces the body as generic filler ("Body text here with …").
    BODY_TEXT_HERE = "body text here"
    #: Self-describing test language ("… prose to test submission").
    PROSE_TO_TEST = "prose to test"


class PlaceholderPageError(EmptyAnswerError):
    """A ``final_result`` submission carried a placeholder page.

    Subclasses :class:`~kompend.agents.failure.EmptyAnswerError` so the
    agentic pass loop applies its existing, proven remedy for "the model
    returned nothing usable": the unit is recorded, the pass is retried or
    the segment stops loudly — and the placeholder page is never silently
    dropped and never written to the wiki.
    """


class _CoverageCredit:
    """Decides whether produced page titles cover fetched concepts.

    T2: rapidfuzz similarity (:func:`kompend.core.text.matching.best_match`).
    T3: embedding cosine via the configured embedding model, embedded
    incrementally and cached per string so repeated passes stay cheap.
    When no embedding provider is available, T2 alone bounds the work —
    callers degrade gracefully.
    """

    def __init__(self, embedding_model: str) -> None:
        self._provider = None
        if embedding_model and embedding_model != "hash":
            try:
                self._provider = make_provider(embedding_model)
            except Exception as exc:
                logger.debug("Coverage semantic tier disabled (no provider): %s", exc)
        self._vectors: dict[str, list[float]] = {}

    def covered(self, concept_name: str, titles: set[str]) -> bool:
        """True when some title in *titles* covers *concept_name*."""
        if not titles:
            return False
        if best_match(concept_name, titles, threshold=_COVER_SIMILARITY_THRESHOLD) is not None:
            return True
        if self._provider is None:
            return False
        cvec = self._vec(concept_name)
        if cvec is None:
            return False
        for title in titles:
            tvec = self._vec(title)
            if tvec is not None and cosine(cvec, tvec) >= _COVER_SEMANTIC_THRESHOLD:
                return True
        return False

    def _vec(self, text: str) -> list[float] | None:
        if self._provider is None:
            return None
        cached = self._vectors.get(text)
        if cached is not None:
            return cached
        try:
            vec = self._provider.embed(text)
        except Exception as exc:
            logger.debug("Coverage embed failed for %r: %s", text, exc)
            self._provider = None  # one failure disables the tier
            return None
        self._vectors[text] = vec
        return vec


class _SegmentUnit(NamedTuple):
    """One generation work unit: Loop 1 splits a segment into smaller ones."""

    label: str
    seg_id: str
    segment: list[tuple[str, str]]
    run_prefix: str | None
    seg_idx: int
    depth: int


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


def prepare_source_segments(
    compiler: TwoPhaseRuntime,
    chunks: list[ResolvedChunk],
    *,
    budget: RequestBudget | None = None,
) -> list[list[tuple[str, str]]]:
    """Build source segments sized to the context window.

    *budget* is the fixed per-request conversation cost that is NOT the
    segment content: system prompt + rendered user prompt (template + the
    known_titles head + pinned_facts) + tool-call framing + the
    fetch_concepts result + the safety margin. The agentic flow passes the measured budget
    so the segment budget = window - reserved - answer reserve, guaranteeing
    the full request (segment + output) fits.
    Content is preserved: oversized sources are split across more
    segments by :func:`segment_sources` — nothing is cropped.
    """
    by_source: dict[str, str] = defaultdict(str)
    for rc in chunks:
        label = safe_stem(rc.chunk.source_file or "unknown")
        if rc.chunk.text:
            by_source[label] += rc.chunk.text + "\n\n"
    sources: list[tuple[str, str]] = list(by_source.items())

    cw = compiler._context_window
    # The rendered content block is ``"\n\n".join(f"### {label}\n{content}")``
    # — each source adds ``### {label}\n`` plus the ``\n\n`` separator. The
    # request-fit guard measures the *rendered* block, so the sizing must
    # reserve this label/separator overhead or the guard fires on segments
    # that are exactly at the raw-content budget (observed: 27-token
    # overshoot on a 39K-token segment). Conservative: reserve the total
    # across all sources (segments share it, so this over-reserves slightly).
    cpt = compiler._config.llm.default_chars_per_token
    label_overhead_tokens = int(sum(len(label) + 7 for label, _ in sources) / cpt)
    # Sizing-feasibility invariant, checked at planning time (before any
    # LLM call): the smallest segment the sizer can emit is
    # min_segment_tokens (the floor in tokens_to_chars below), and its
    # rendered block adds label_overhead on top. If even that cannot fit
    # beside the reservation + answer reserve, every segment would trip
    # the per-segment request-fit guard and the run would abort after all
    # upstream work — fail here instead, naming the components.
    min_floor_tokens = compiler._config.llm.min_segment_tokens
    if budget is None:
        # Legacy template path: only the system template is pre-known; the
        # content block IS the prompt, so no other reservation applies.
        budget = RequestBudget(
            window=cw,
            reserved={"system": compiler._gen_system_tokens},
        )
    if budget.available_tokens - label_overhead_tokens < min_floor_tokens:
        raise LLMOutputError(
            f"Segment sizing impossible: {budget.report()} + source labels ({label_overhead_tokens})"
            + f" leaves {budget.available_tokens - label_overhead_tokens} tokens, below the minimum segment "
            + f"({min_floor_tokens}). Reduce the reserved payloads (known_titles is served in windows; check "
            + "pinned_facts/categories) or raise compile.llm.context_window."
        )
    # The rendered block carries the label overhead on top of the segment
    # content, so it is reserved before the content budget is derived.
    content_budget = max(min_floor_tokens, budget.content_tokens - label_overhead_tokens)
    segment_char_size = tokens_to_chars(
        content_budget,
        chars_per_token=cpt,
    )
    logger.info(
        "TwoPhase: segment_char_size=%d chars (%s, labels=%d, answer_reserve=%d)",
        segment_char_size,
        budget.report(),
        label_overhead_tokens,
        cw.answer_reserve,
    )
    source_segments = segment_sources(sources, segment_char_size)
    logger.info(
        "TwoPhase: %d source(s) split into %d segment(s) (total content %d chars)",
        len(sources),
        len(source_segments),
        sum(len(c) for _, c in sources),
    )
    return source_segments


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


def _reject_placeholder_pages(gen_output: GenerationOutput, *, unit: str) -> None:
    """Refuse the whole submission when any submitted page is a placeholder.

    The gate sits where a model submission enters the pipeline: the live
    pass result, and a transcript replayed from the step store. The
    offender is never silently dropped — the whole
    :class:`~kompend.compilation.schema.GenerationOutput` is refused with
    :class:`PlaceholderPageError`, which routes through the pass loop's
    existing failure path: the unit is recorded, and the segment either
    re-runs or fails loudly. The placeholder page never reaches the wiki.
    """
    offenders: list[str] = []
    for page in gen_output.pages:
        if reasons := _placeholder_reasons(page):
            offenders.append(f"'{page.title or '(untitled)'}' ({'; '.join(reasons)})")
    if offenders:
        raise PlaceholderPageError(f"{unit}: rejected {len(offenders)} placeholder page(s) — {'; '.join(offenders)}")


def _placeholder_reasons(page: GenerationPage) -> list[str]:
    """Reasons *page* is a placeholder, empty when the page is legitimate.

    A page with no provenance had no source to summarize (the generation
    prompt requires ``provenance`` — a list of cited source filenames — on
    every page), a filler body is the writer testing its own tool, and a
    single-character title is a test identity, not a page name.
    """
    reasons: list[str] = []
    if not [entry for entry in page.provenance if str(entry).strip()]:
        reasons.append("empty provenance (no source chunks cited)")
    body = (page.content or "").casefold()
    markers = [marker.value for marker in PlaceholderMarker if marker.value in body]
    if markers:
        reasons.append(f"filler body text ({', '.join(markers)})")
    title = (page.title or "").strip()
    if title and len(title) < _MIN_PAGE_TITLE_CHARS:
        reasons.append(f"title '{title}' is shorter than {_MIN_PAGE_TITLE_CHARS} characters")
    return reasons


def _shrink_and_requeue(
    broker: ConceptBroker,
    coverage: _CoverageCredit,
    compiler: TwoPhaseRuntime,
    pending: deque[_SegmentUnit],
    unit: _SegmentUnit,
    *,
    start_remaining: int,
    seg_titles: set[str],
    max_splits: int,
    kind: AgentFailureKind,
    err: str,
) -> ShrinkOutcome:
    """Shrink a failed unit: Loop 1 decides, Loop 2 gets the ambiguous rest.

    SPLIT means the halves were enqueued — the caller leaves its pass
    loop. DECLINED keeps today's path. SKIP/ABORT carry the decision
    model's verdict for a failure the deterministic policy declined.
    The parent pool is drained on a split, so the tail sees no uncovered
    remainder; the halves carry their own broker pools and transcript
    prefixes.
    """
    # Fetched-but-unpaged concepts are stranded in the parent pool once it
    # drains — they move to the halves with the unfetched remainder, or the
    # split silently drops work.
    unfinished = [c for c in broker.consumed(unit.seg_id) if not coverage.covered(str(c.get(ExtractionField.NAME, "")), seg_titles)]
    unfinished.extend(broker.remaining_concepts(unit.seg_id))
    tracker = compiler._usage_tracker
    signal = AdaptationSignal(
        seg_id=unit.seg_id,
        kind=kind,
        adaptations_used=unit.depth,
        concepts_assigned=start_remaining,
        concepts_remaining=broker.remaining(unit.seg_id),
        fetches=len(broker.consumed(unit.seg_id)),
        context_tokens=tracker.context_tokens or None,
        context_budget=tracker.context_budget or None,
        last_error=err,
    )
    reason = ""
    plan = plan_adaptation(signal, max_adaptations=max_splits)
    if plan.split:
        reason = plan.reason
    else:
        # Loop 2: the deterministic policy declined an ambiguous failure.
        # The decision model may still earn it one more halving — or stop
        # the retries, or fail loudly. Anything else keeps today's path.
        decision = _consult_loop2(compiler, unit, signal, max_splits)
        if decision is None:
            return ShrinkOutcome.DECLINED
        if decision.action is ErrorAction.ABORT:
            return ShrinkOutcome.ABORT
        if decision.action is ErrorAction.SKIP:
            return ShrinkOutcome.SKIP
        if not decision.shrink:
            return ShrinkOutcome.DECLINED
        reason = f"{unit.seg_id}: decision model overrode the decline — halving"
    halves = split_unit(unit.segment, unfinished)
    if halves is None:
        return ShrinkOutcome.DECLINED
    (sources_a, concepts_a), (sources_b, concepts_b) = halves
    broker.drain_caller(unit.seg_id)
    for sub, sub_sources, sub_concepts in (("a", sources_a, concepts_a), ("b", sources_b, concepts_b)):
        sub_id = f"{unit.seg_id}/s{sub}"
        broker.adopt_caller(sub_id, sub_concepts)
        pending.append(
            _SegmentUnit(
                label=f"{unit.label} (split {unit.depth + 1}{sub})",
                seg_id=sub_id,
                segment=sub_sources,
                run_prefix=f"{unit.run_prefix}/sh{unit.depth + 1}-{sub}" if unit.run_prefix else None,
                seg_idx=unit.seg_idx,
                depth=unit.depth + 1,
            )
        )
    logger.warning(
        "TwoPhase: agentic %s exceeds the window (%s) — %s; re-queued as 2 sub-units (%d/%d concepts).",
        unit.label,
        err,
        reason,
        len(concepts_a),
        len(concepts_b),
    )
    return ShrinkOutcome.SPLIT


def _consult_loop2(
    compiler: TwoPhaseRuntime,
    unit: _SegmentUnit,
    signal: AdaptationSignal,
    max_splits: int,
) -> ErrorDecision | None:
    """Ask the decision model about an ambiguous failure Loop 1 declined.

    ``None`` keeps today's path: the kind is not ambiguous, the budget is
    spent, no decision model is configured, or the sidecar call failed.
    """
    if signal.kind not in _AMBIGUOUS_KINDS or signal.adaptations_used >= max_splits:
        return None
    return decide_unit_adaptation(
        compiler._config.decision,
        signal=signal,
        stage=f"agentic-generation:{unit.seg_id}",
        max_adaptations=max_splits,
    )


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


def _format_uncovered_concepts(
    broker: ConceptBroker,  # noqa: F821 — forward ref
    seg_id: str,
    *,
    limit: int = 10,
) -> str:
    """Build a traceable uncovered-concepts summary for warnings/exceptions.

    Each entry is rendered as ``"<name> [sources: <src1>, <src2>...]"`` so the
    user can grep for the source PDF/section that produced the concept and
    cross-check against the wiki's source tree and the step-store transcripts.

    A trailing ellipsis marks truncation when more than *limit* concepts
    remain. Empty ``sources`` lists fall back to ``"(no-source)"`` so the
    field never silently disappears (memory #149).
    """
    items = broker.remaining_concepts(seg_id)
    if not items:
        return "(none)"
    rendered: list[str] = []
    for concept in items[:limit]:
        name = str(concept.get(ExtractionField.NAME, "?")).strip() or "?"
        sources_raw = concept.get("sources") or []
        if isinstance(sources_raw, str):
            sources: list[str] = [sources_raw]
        else:
            sources = [str(s) for s in sources_raw if s]
        sources_str = ", ".join(sources) if sources else "(no-source)"
        rendered.append(f"{name} [sources: {sources_str}]")
    suffix = "…" if len(items) > limit else ""
    return ", ".join(rendered) + suffix


def _failure_action(
    compiler: TwoPhaseRuntime,
    exc: BaseException,
    *,
    stage: str,
    attempt: int,
    max_attempts: int,
    transient: bool,
) -> ErrorAction:
    """Decide what to do with a failed generation call.

    The optional decision model answers this when it is configured. Without it
    the fallback is deterministic: a transport failure is worth repeating, a
    semantic failure is not - repeating it would produce the same output.
    """
    decided = decide_error_action(
        compiler._config.decision,
        error=str(exc),
        stage=stage,
        attempt=attempt,
        max_attempts=max_attempts,
        context=f"model={compiler._model or '?'}",
    )
    if decided is not None:
        return decided
    return ErrorAction.RETRY if transient else ErrorAction.SKIP


def _resolve_page_kind(
    model_kind: str,
    title: str,
    preassigned: dict[str, str],
    extras: dict[str, str],
) -> NodeType:
    """Resolve a page kind: a pre-assigned kind wins, else the model's label.

    ``_with_entity_pages`` pre-assigns ``kind: entity`` to mandatory pages.
    Those are enforced deterministically so a model that drops or alters the
    kind cannot misfile the page (entity subdirectory routing depends on it).
    Pages the model invents keep their own kind, validated by ``_allowed_kind``.
    """
    expected = preassigned.get(normalize_concept_key(title))
    return _allowed_kind(expected or model_kind, extras)


def _allowed_kind(kind: str, extras: dict[str, str]) -> NodeType:
    """Validate an LLM-assigned page kind against the taxonomy.

    Mandatory kinds plus configured extras are accepted; anything else falls
    back to ``concept`` (permissive-consumption model, OKF §10).
    """
    if kind in MANDATORY_KINDS:
        return NodeType(kind)
    if kind in extras:
        return NodeType.of(kind)
    return NodeType.CONCEPT


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
    # Persist any new entity-type decisions to the on-disk registry before
    # assembly runs. Assembly reads from this registry (so previously-known
    # kinds win), but new concepts from THIS compile still need to land here
    # so the NEXT compile sees them.
    try:
        entity_types_registry = EntityTypeRegistry(compiler._config.paths.entity_types_file)
        new_entries = 0
        for c in concepts:
            if c.get("kind") != NodeType.ENTITY.value:
                continue
            key = normalize_concept_key(str(c.get(ExtractionField.NAME, "")))
            if not key:
                continue
            if entity_types_registry.get(key) is not None:
                continue
            sanitized = _sanitize_entity_type(str(c.get("type", "")))
            if sanitized and entity_types_registry.record(key, sanitized):
                new_entries += 1
        if new_entries:
            entity_types_registry.save()
            logger.info(
                "Entity-type registry: persisted %d new entry/entries (total=%d)",
                new_entries,
                len(entity_types_registry),
            )
    except OSError as exc:
        logger.warning("Could not persist entity-type registry: %s", exc, exc_info=True)
    cw = compiler._context_window

    if not nodes:
        raise LLMOutputError(
            f"Generation failed: 0 of {len(concepts)} concept(s) synthesized. "
            f"Every LLM call failed after retries (see the transient/validation warnings above - "
            f"an upstream long-request timeout is not a structured-output problem). "
            f"Check that model '{compiler._model}' can produce structured JSON output."
        )

    # Force-merge by concept key (normalize_concept_key) FIRST. This catches
    # near-duplicates that share a key but different slugs (e.g. "Absolute
    # Category Rating" and "Absolute Category Rating (ACR)") which the
    # embedding cosine pass below would also catch, but doing it
    # deterministically first avoids unnecessary embedding work and avoids
    # silent fallthrough when the embedding model is unavailable.
    nodes = compiler.merge_by_concept_key(nodes)

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
                MetadataField.MODEL: compiler._model or "",
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
        logger.info("TwoPhase: learned %d wikilink alias(es), e.g. %s", len(aliases), list(aliases.items())[:LOG_SAMPLE_LIMIT])
    return CompiledWiki(
        nodes=nodes,
        edges=edges,
        toc=toc,
        quality_score=quality,
        metadata={
            WikiField.CHUNK_COUNT: len(chunks),
            MetadataField.NODE_COUNT: len(nodes),
            MetadataField.EDGE_COUNT: len(edges),
            MetadataField.MODEL: compiler._model or "",
            MetadataField.TWO_PHASE: True,
            MetadataField.CONCEPTS_MERGED: len(concepts),
            MetadataField.CONTEXT_WINDOW: cw.max_input,
            MetadataField.PINNED_FACTS_COUNT: len(pinned_clean),
            MetadataField.WIKILINK_ALIASES: aliases,
        },
    )


def _sanitize_entity_type(raw: str) -> str:
    """Normalize an entity type for use as a folder name ("Person" → "person")."""
    sanitized = slugify(raw or "", max_length=40)
    return sanitized or "entity"


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
    if not compiler._config.cache_ignore_model:
        deps[CacheDeps.MODEL] = compiler._model or ""
    deps[CacheDeps.CHUNK_SIZE_MIN] = str(budget.chunk_size_min or "")
    deps[CacheDeps.CHUNK_SIZE_MAX] = str(budget.chunk_size_max or "")
    deps[CacheDeps.PINNED_FACTS] = pin_hash or ""
    return deps
