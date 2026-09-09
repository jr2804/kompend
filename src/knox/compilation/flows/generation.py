"""Generation flow — per-concept page generation and wiki assembly.

Extracted from :class:`TwoPhaseCompiler` — generation phase methods
that operate on compiler instance state.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
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
from knox.core.text.matching import best_match, cosine
from knox.core.utils.exceptions import LLMOutputError
from knox.lint.base import LintRuleOutcome
from knox.lint.structural import BrokenWikilinksRule
from knox.llm.format import encode_for_llm
from knox.llm.usage import UsageTracker
from knox.logging import get_logger
from knox.rag.zvec_store import make_provider

# Matches wikilink-wrapped source citations like [[sources/foo.pdf]] so they
# can be normalized to the intended [sources/foo.pdf] inline citation format
# regardless of LLM compliance with the system prompt.
_SOURCE_WIKILINK_RE = re.compile(r"\[\[\s*(sources/[^\]\n]+?)\s*\]\]")


#: rapidfuzz-scale (0–100) threshold for "does a produced page title cover
#: this concept" (T2 string tier).
_COVER_SIMILARITY_THRESHOLD = 85.0

#: Fixed per-turn agentic conversation framing (tool call + result parts)
#: beyond the content itself. Sized for a full read_segment + fetch_concepts
#: round and a follow-up fetch round (models often loop several rounds
#: before answering).
_AGENTIC_TOOL_FRAMING_TOKENS = 900
#: Per fetch_concepts round: assistant tool call + result framing.
_AGENTIC_FETCH_ROUND_TOKENS = 600


def _format_uncovered_concepts(
    broker: "ConceptBroker",  # noqa: F821 — forward ref
    seg_id: str,
    *,
    limit: int = 10,
) -> str:
    """Build a traceable uncovered-concepts summary for warnings/exceptions.

    Each entry is rendered as ``"<name> [sources: <src1>, <src2>...]"`` so the
    user can grep for the source PDF/section that produced the concept and
    cross-check against the wiki's source tree and the gen_checkpoint file.

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


def _is_upstream_transient_timeout(exc: BaseException) -> bool:
    """Return True when *exc* looks like an upstream "Request timed out" 4xx.

    Matches ``pydantic_ai.exceptions.ModelHTTPError`` whose ``status_code``
    is below 500 and whose ``body['message']`` (or string body) contains
    ``"timed out"`` case-insensitively. Other exception types are not
    matched — keep this heuristic narrow.
    """
    try:
        from pydantic_ai.exceptions import ModelHTTPError
    except ImportError:  # pragma: no cover — pydantic-ai is a hard dep
        return False

    if not isinstance(exc, ModelHTTPError):
        return False
    if not isinstance(exc.status_code, int) or exc.status_code >= 500:
        return False
    body = exc.body
    if isinstance(body, dict):
        msg = body.get("message") or body.get("error") or ""
    elif isinstance(body, str):
        msg = body
    else:
        msg = str(exc)
    return "timed out" in str(msg).lower()
#: Safety floor so a pathological known_titles/pinned list cannot zero the
#: segment budget (content is preserved via more segments instead).
_AGENTIC_SAFETY_TOKENS = 512
#: Fraction of the context window reserved as extra headroom beyond the
#: measured overhead. Ollama's effective usable context is smaller than the
#: nominal num_ctx (KV cache + prompt-cache rollover), so exact-fit requests
#: still stall; a fraction-based margin keeps the full request inside the
#: *effective* window. Content is never cropped — the segment budget shrinks
#: and sources split across more segments.
_AGENTIC_SAFETY_FRACTION = 0.08
#: Cosine threshold for the T3 semantic tier — catches paraphrases and
#: punctuation variants the string tier misses (e.g. "ITU-T P.10/G.100" vs
#: "P.10 G.100 Recommendation"). Conservative: a false *credit* silently
#: drops a concept's page, so the bar sits above lint's topic-vs-chunk 0.55.
_COVER_SEMANTIC_THRESHOLD = 0.65

logger = get_logger(__name__)

_loader = get_template_loader()


class _CoverageCredit:
    """Decides whether produced page titles cover fetched concepts.

    T2: rapidfuzz similarity (:func:`knox.core.text.matching.best_match`).
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
    known_titles = build_known_titles(compiler, concepts)
    # Agentic-conversation fixed cost (everything that lands in the request
    # that is NOT the segment content): system template + rendered user
    # prompt (template + known_titles + pinned_facts) + tool-call framing
    # (read_segment + fetch_concepts round-trips) + the fetch result + a
    # safety margin. Subtracting this from the window keeps the full
    # request (segment content + output budget) inside the context.
    reserved_tokens = _agentic_reserved_tokens(compiler, known_titles, pinned_clean)
    source_segments = prepare_source_segments(compiler, chunks, reserved_tokens=reserved_tokens)

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


def prepare_source_segments(
    compiler: TwoPhaseRuntime,
    chunks: list[ResolvedChunk],
    *,
    reserved_tokens: int = 0,
) -> list[list[tuple[str, str]]]:
    """Build source segments sized to the context window.

    *reserved_tokens* is the fixed per-request conversation cost that is
    NOT the segment content: system prompt + rendered user prompt
    (template + known_titles + pinned_facts) + tool-call framing + the
    fetch_concepts result + a safety margin. The agentic flow passes the
    measured value so the segment budget = window - reserved - output
    budget, guaranteeing the full request (segment + output) fits.
    Content is preserved: oversized sources are split across more
    segments by :func:`segment_sources` — nothing is cropped.
    """
    by_source: dict[str, str] = defaultdict(str)
    for rc in chunks:
        label = Path(str(rc.chunk.source_file or "unknown")).stem
        if rc.chunk.text:
            by_source[label] += rc.chunk.text + "\n\n"
    sources: list[tuple[str, str]] = list(by_source.items())

    cw = compiler._context_window
    budget = compiler._budget
    # The rendered content block is ``"\n\n".join(f"### {label}\n{content}")``
    # — each source adds ``### {label}\n`` plus the ``\n\n`` separator. The
    # request-fit guard measures the *rendered* block, so the sizing must
    # reserve this label/separator overhead or the guard fires on segments
    # that are exactly at the raw-content budget (observed: 27-token
    # overshoot on a 39K-token segment). Conservative: reserve the total
    # across all sources (segments share it, so this over-reserves slightly).
    cpt = compiler._config.llm.default_chars_per_token
    label_overhead_tokens = int(sum(len(label) + 7 for label, _ in sources) / cpt)
    if reserved_tokens:
        # Agentic path: reserved is the complete fixed overhead (system +
        # prompt + variables + tool framing), so the content budget is the
        # window minus the output budget minus everything reserved.
        content_budget = cw.max_input - budget.gen_output_max - reserved_tokens - label_overhead_tokens
    else:
        # Legacy template path: only the system template is pre-known; the
        # content block IS the prompt, so no other reservation applies.
        content_budget = cw.max_input - compiler._gen_system_tokens - budget.gen_output_max - label_overhead_tokens
    segment_char_size = tokens_to_chars(
        max(compiler._config.llm.min_segment_tokens, content_budget),
        chars_per_token=compiler._config.llm.default_chars_per_token,
    )
    logger.info(
        "TwoPhase: segment_char_size=%d chars (cw=%d, sys_tok=%d, gen_max=%d, reserved=%d)",
        segment_char_size,
        cw.max_input,
        compiler._gen_system_tokens,
        budget.gen_output_max,
        reserved_tokens,
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
                usage_tracker=compiler._usage_tracker,
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
    # Coverage credit: rapidfuzz T2 + embedding T3 (single instances shared
    # across segments so title/concept embeddings are cached compile-wide).
    coverage = _CoverageCredit(compiler._config.embedding.model)

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
            # Cap to total — checkpoint names can exceed the broker share if
            # the model produced duplicate titles across passes.
            pages_done = min(pages_done + len(caller_progress[seg_id]), gen_progress_total)
            compiler._report(
                pages_done,
                gen_progress_total,
                f"generating [{pages_done}/{gen_progress_total}] {seg_label} (resumed){_rate_suffix(compiler._usage_tracker)}",
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
            usage_tracker=compiler._usage_tracker,
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
        last_error: str | None = None
        seg_start_nodes = len(nodes)
        seg_titles: set[str] = set()  # normalized page titles produced for this segment
        seg_consumed_names: list[str] = list(caller_progress.get(seg_id, []))  # accumulate for checkpoint
        # Request-fit guard: the segment content is placed in the conversation
        # by read_segment (a tool result), and the model still emits up to
        # gen_output_max on top. If even ONE call would exceed the window,
        # fail loudly now instead of letting Ollama stall/time out mid-stream.
        # The no-crop rule is honored: we never silently clamp max_tokens;
        # content is preserved by splitting into smaller segments at the
        # sizing stage.
        _segment_content_tokens = _token_estimate(content_block, compiler._config.llm.default_chars_per_token)
        _estimated_request = _agentic_reserved_tokens(compiler, known_titles or [], pinned_facts or []) + _segment_content_tokens
        if _estimated_request + compiler._budget.gen_output_max > compiler._context_window.max_input:
            raise LLMOutputError(
                f"Agentic generation for {seg_label}: segment content ({_segment_content_tokens} tokens) + fixed "
                f"overhead ({_estimated_request - _segment_content_tokens} tokens) + output budget "
                f"({compiler._budget.gen_output_max} tokens) exceeds the {compiler._context_window.max_input}-token "
                f"context window. Segments were sized with reserved_tokens; this indicates the conversation "
                f"grew beyond the reservation. Re-run with a smaller context budget or check config."
            )
        while broker.remaining(seg_id) > 0 and passes < cap:
            # Inner retry: upstream transient errors (4xx with
            # 'timed out' body) get a brief backoff and a re-attempt of the
            # SAME pass (does not consume a pass slot). Hard errors (5xx,
            # validation, schema) skip the backoff and consume a pass.
            retry_attempts = 0
            for retry in range(_UPSTREAM_TRANSIENT_MAX_RETRIES + 1):
                passes += 1 if retry == 0 else 0  # only the first try counts as a pass
                try:
                    gen_output = agent.run(deps)
                    if retry_attempts:
                        logger.info(
                            "Agentic generation for '%s' recovered after %d transient retry attempt(s)",
                            seg_id,
                            retry_attempts,
                        )
                    break
                except LLMOutputError as exc:
                    logger.warning("Agentic generation failed for '%s': %s", seg_id, exc)
                    gen_output = None
                    last_error = str(exc)
                    break
                except Exception as exc:  # ponytail: PydanticAI raises many subclassed errors; treat uniformly
                    if _is_upstream_transient_timeout(exc) and retry < _UPSTREAM_TRANSIENT_MAX_RETRIES:
                        retry_attempts += 1
                        backoff = _UPSTREAM_TRANSIENT_BACKOFF_BASE * (2 ** (retry_attempts - 1))
                        logger.warning(
                            "Agentic generation transient upstream timeout for '%s' "
                            "(attempt %d/%d); retrying in %.1fs. error=%s",
                            seg_id,
                            retry_attempts,
                            _UPSTREAM_TRANSIENT_MAX_RETRIES,
                            backoff,
                            exc,
                        )
                        time.sleep(backoff)
                        continue
                    logger.warning("Agentic generation unexpected error for '%s': %s", seg_id, exc)
                    gen_output = None
                    last_error = str(exc)
                    break
            debug_file = getattr(agent, "_last_debug_file", None)

            compiler._batch_tracker.record(failed=gen_output is None)

            if gen_output is None:
                logger.warning(
                    "TwoPhase: agentic %s (caller_id=%s, pass=%d/%d, remaining=%d) failed — skipping (debug: %s)",
                    seg_label,
                    seg_id,
                    passes,
                    cap,
                    broker.remaining(seg_id),
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
            seg_progress = min(len(seg_titles), start_remaining)
            running = min(pages_done + seg_progress, gen_progress_total)
            rate_s = _rate_suffix(compiler._usage_tracker)
            compiler._report(
                running,
                gen_progress_total,
                f"generating [{running}/{gen_progress_total}] {seg_label}{rate_s}",
            )

        # Credit the segment with the pages actually produced, not the
        # broker share (which assumes full coverage and would over-credit
        # when the model fails to page every fetched concept).
        pages_done = min(pages_done + len(seg_titles), gen_progress_total)

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
            sample = _format_uncovered_concepts(broker, seg_id, limit=10)
            debug_hint = f" (debug: {debug_file})" if debug_file else ""
            if not seg_titles:
                # Hard fail: no pages at all. Preserve checkpoint (already saved
                # above) so the user can re-run to retry the failed segment.
                # Distinguish a model that never called the tool (tool-call
                # support problem) from one that fetched and then the request
                # died (timeout/overflow — a sizing or provider problem).
                fetched = broker.consumed(seg_id)
                if fetched and last_error:
                    cause = f"it called fetch_concepts {len(fetched)}× but the request failed: {last_error}"
                    hint = "This is not a tool-calling problem — check the request size vs the model's context window."
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

    compiler._report(gen_progress_total, gen_progress_total, "generating complete")
    return nodes, slug_to_idx, caller_progress


def _agentic_reserved_tokens(
    compiler: TwoPhaseRuntime,
    known_titles: list[str],
    pinned_clean: list[str],
) -> int:
    """Fixed per-request cost of an agentic generation turn.

    Everything that lands in the request that is NOT the segment content:
    system template + rendered user prompt (template body + known_titles +
    pinned_facts blocks + categories) + the read_segment/fetch_concepts
    tool-call framing + one fetch result + a safety margin.

    Measured from the actual templates and variable payloads (same
    chars-per-token heuristic used for segment sizing), never hard-coded
    estimates.
    """
    cpt = compiler._config.llm.default_chars_per_token
    system_tokens = _token_estimate(_loader.get_body("compilation/generation_agentic_system.md.j2"), cpt)
    prompt_tokens = _token_estimate(_loader.get_body("compilation/generation_agentic_prompt.md.j2"), cpt)
    known_tokens = _token_estimate("\n".join(known_titles), cpt)
    pinned_tokens = _token_estimate("\n".join(pinned_clean), cpt)
    categories = compiler._config.extraction_categories
    categories_tokens = _token_estimate("\n".join(f"{k}: {v}" for k, v in categories.items()), cpt)
    fraction_safety = max(_AGENTIC_SAFETY_TOKENS, int(round(_AGENTIC_SAFETY_FRACTION * compiler._context_window.max_input)))
    total = (
        system_tokens
        + prompt_tokens
        + known_tokens
        + pinned_tokens
        + categories_tokens
        + _AGENTIC_TOOL_FRAMING_TOKENS
        + _AGENTIC_FETCH_ROUND_TOKENS
        + fraction_safety
    )
    logger.info(
        "TwoPhase: agentic reserved_tokens=%d (sys=%d prompt=%d known=%d pinned=%d cats=%d framing=%d fetch=%d safety=%d [%.0f%% of cw])",
        total,
        system_tokens,
        prompt_tokens,
        known_tokens,
        pinned_tokens,
        categories_tokens,
        _AGENTIC_TOOL_FRAMING_TOKENS,
        _AGENTIC_FETCH_ROUND_TOKENS,
        fraction_safety,
        100 * _AGENTIC_SAFETY_FRACTION,
    )
    return total


def _token_estimate(text: str, chars_per_token: float) -> int:
    return max(1, int(round(len(text) / chars_per_token)))


def _rate_suffix(tracker: UsageTracker | None) -> str:
    """`` · 12.3 tok/s (avg 9.8)`` from the shared tracker, or ``""``.

    Uses the tracker's real completion-token event log (short-term window
    + overall average) — the single source of truth for throughput.
    """
    if tracker is None:
        return ""
    overall = tracker.overall_rate()
    recent = tracker.recent_rate()
    if not overall and not recent:
        return ""
    return f" · {recent:.1f} tok/s" if recent else f" · avg {overall:.1f} tok/s"


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
