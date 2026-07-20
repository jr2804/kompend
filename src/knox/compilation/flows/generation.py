"""Generation flow — per-concept page generation and wiki assembly.

Extracted from :class:`TwoPhaseCompiler` — generation phase methods
that operate on compiler instance state.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from knox.agents.generation import GenerationAgent
from knox.compilation._checkpoints import (
    _delete_gen_checkpoint,
    _load_gen_checkpoint,
    _save_gen_checkpoint,
)
from knox.compilation._two_phase_core import (
    _MIN_SEGMENT_TOKENS,
    ExtractionSet,
    _compute_quality_score,
    _extract_images_from_chunks,
    _segment_sources,
    _slugify,
    _tokens_to_chars,
    _try_pydanticai_generation,
    _upsert_node,
)
from knox.compilation.flows.assembly import attach_tags_to_nodes, build_edges, merge_similar_concepts
from knox.core.models import CompiledWiki, ResolvedChunk, WikiField, WikiNode
from knox.core.models.enums import MetadataField
from knox.core.templates import get_template_loader
from knox.core.utils.exceptions import LLMOutputError
from knox.llm.format import encode_for_llm
from knox.logging import get_logger

logger = get_logger(__name__)

_loader = get_template_loader()


def run_generation(
    compiler: Any,
    extractions: ExtractionSet,
    chunks: list[ResolvedChunk],
    *,
    trees: list[Any] | None = None,
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
                "two_phase": True,
                "generation_skipped": True,
                "concepts_available": 0,
                "skip_reason": "extraction returned no concepts",
                "node_count": 0,
            },
        )

    all_images = _extract_images_from_chunks(chunks)
    source_segments = prepare_source_segments(compiler, chunks)
    pinned_clean = list(dict.fromkeys(f.strip() for f in (pinned_facts or []) if f and f.strip()))
    generation_system = _loader.get_body("compilation/generation_system.md.j2")

    num_concepts = len(concepts)
    gen_total = ((num_concepts + compiler._budget.concepts_per_batch - 1) // compiler._budget.concepts_per_batch) * len(source_segments)
    if gen_total > 0:
        compiler._report(0, gen_total, f"generating wiki ({num_concepts} concepts, {len(source_segments)} segments)")

    nodes, slug_to_idx = run_generation_with_checkpoint(
        compiler,
        concepts,
        source_segments,
        generation_system,
        all_images,
        pinned_clean,
    )

    return assemble_wiki(compiler, nodes, slug_to_idx, chunks, extractions, concepts, pinned_clean)


def prepare_concepts(compiler: Any, extractions: ExtractionSet) -> list[dict[str, Any]]:
    """Merge and validate concepts from extractions. Returns empty list if none."""
    concepts = extractions.concept_pages()
    embed_model = compiler._config.embedding.model
    if embed_model and embed_model != "hash":
        concepts = merge_similar_concepts(concepts, embed_model)
    if concepts:
        sample_limit = compiler._config.display.concept_names
        logger.info(
            "TwoPhase: %d concept(s) ready for generation (sample: %s)",
            len(concepts),
            [c["name"] for c in concepts[:sample_limit]],
        )
    else:
        sample_keys: set[str] = set()
        for ext in extractions.extractions:
            for c in ext.concepts:
                sample_keys.update(c.keys())
        logger.warning(
            "TwoPhase: concept_pages() empty — extraction concept keys: %s",
            sorted(sample_keys) or "(no concepts returned)",
        )
    return concepts


def prepare_source_segments(compiler: Any, chunks: list[ResolvedChunk]) -> list[list[tuple[str, str]]]:
    """Build source segments sized to the context window."""
    by_source: dict[str, str] = defaultdict(str)
    for rc in chunks:
        label = Path(str(rc.chunk.source_file or "unknown")).stem
        if rc.chunk.text:
            by_source[label] += rc.chunk.text + "\n\n"
    sources: list[tuple[str, str]] = list(by_source.items())

    cw = compiler._context_window
    budget = compiler._budget
    segment_char_size = _tokens_to_chars(
        max(_MIN_SEGMENT_TOKENS, cw.max_input - compiler._gen_system_tokens - budget.gen_output_max),
    )
    logger.info(
        "TwoPhase: segment_char_size=%d chars (cw=%d, sys_tok=%d, gen_max=%d)",
        segment_char_size,
        cw.max_input,
        compiler._gen_system_tokens,
        budget.gen_output_max,
    )
    source_segments = _segment_sources(sources, segment_char_size)
    logger.info(
        "TwoPhase: %d source(s) split into %d segment(s) (total content %d chars)",
        len(sources),
        len(source_segments),
        sum(len(c) for _, c in sources),
    )
    return source_segments


def run_generation_with_checkpoint(
    compiler: Any,
    concepts: list[dict[str, Any]],
    source_segments: list[list[tuple[str, str]]],
    generation_system: str,
    all_images: list[dict[str, Any]],
    pinned_clean: list[str],
) -> tuple[list[WikiNode], dict[str, int]]:
    """Run generation with checkpoint support, returning nodes and slug map."""
    checkpoint = _load_gen_checkpoint(compiler._config, compiler._active_gen_deps)
    if checkpoint is not None:
        ck_nodes, ck_slug_to_idx, ck_completed = checkpoint
        logger.info(
            "TwoPhase: generation checkpoint loaded — %d batch(es) already done, %d node(s)",
            len(ck_completed),
            len(ck_nodes),
        )
    else:
        ck_nodes, ck_slug_to_idx, ck_completed = [], {}, set()

    nodes, slug_to_idx = generate_all_pages(
        compiler,
        concepts,
        source_segments,
        generation_system,
        all_images,
        pinned_facts=pinned_clean,
        initial_nodes=ck_nodes,
        initial_slug_to_idx=ck_slug_to_idx,
        completed_batches=ck_completed,
    )
    _delete_gen_checkpoint(compiler._config, compiler._active_gen_deps)
    return nodes, slug_to_idx


def generate_all_pages(
    compiler: Any,
    concepts: list[dict[str, Any]],
    source_segments: list[list[tuple[str, str]]],
    generation_system: str,
    all_images: list[dict[str, Any]],
    *,
    pinned_facts: list[str],
    initial_nodes: list[WikiNode] | None = None,
    initial_slug_to_idx: dict[str, int] | None = None,
    completed_batches: set[int] | None = None,
) -> tuple[list[WikiNode], dict[str, int]]:
    """Generate (or fallback-generate) one node per concept/segment pair."""
    nodes: list[WikiNode] = list(initial_nodes or [])
    slug_to_idx: dict[str, int] = dict(initial_slug_to_idx or {})
    completed: set[int] = set(completed_batches or [])

    num_concepts = len(concepts)
    num_segments = len(source_segments)
    batch_size = compiler._budget.concepts_per_batch
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
                retries=compiler._config.llm.output_retries,
                reasoning_effort=compiler._config.llm.reasoning_effort,
                context_manager=compiler._config.llm.context_manager,
                context_window=compiler._context_window.max_input,
            )
            gen_output = _try_pydanticai_generation(
                agent,
                concepts_json=encode_for_llm(batch, compiler._config.llm.data_format),
                content=content_block,
                source_id=seg_label,
                pinned_facts=pinned_facts,
                timeout=compiler._timeout,
            )

            compiler._batch_tracker.record(failed=gen_output is None)

            if gen_output is None:
                # Quantify the impact: which concepts were skipped, how many total.
                concept_names = [c.get("name", f"concept-{batch_start + i}") for i, c in enumerate(batch)]
                total_concepts = len(concepts)
                skipped_str = ", ".join(concept_names[:3])
                if len(concept_names) > 3:
                    skipped_str += f", +{len(concept_names) - 3} more"
                logger.warning(
                    "TwoPhase: generation failed for batch %d-%d/%d (segment %d) — "
                    "%d/%d concepts will have no wiki page: [%s]",
                    batch_start,
                    batch_start + len(batch),
                    total_concepts,
                    seg_idx,
                    len(batch),
                    total_concepts,
                    skipped_str,
                )
                failed_count += 1
                if compiler._batch_tracker.should_stop():
                    logger.error(
                        "TwoPhase: stopping generation early — %d/%d batches failed (model '%s' is not suitable for structured JSON output)",
                        compiler._batch_tracker.failed,
                        compiler._batch_tracker.total,
                        compiler._model,
                    )
                    return nodes, slug_to_idx
                continue

            for idx, page in enumerate(gen_output.pages):
                node_id = _slugify(page.title or f"concept-{batch_start + idx}")
                _upsert_node(
                    WikiNode(
                        id=node_id,
                        title=page.title or f"Concept {batch_start + idx}",
                        content=page.content,
                        node_type="concept",
                        source_chunks=[],
                        metadata={
                            MetadataField.PROVENANCE: page.provenance,
                            "related": page.related,
                            "two_phase": True,
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
        _save_gen_checkpoint(
            compiler._config,
            compiler._active_gen_deps,
            nodes,
            slug_to_idx,
            completed,
        )

    compiler._report(gen_progress_total, gen_progress_total, "generating complete")
    return nodes, slug_to_idx


def assemble_wiki(
    compiler: Any,
    nodes: list[WikiNode],
    slug_to_idx: dict[str, int],
    chunks: list[ResolvedChunk],
    extractions: ExtractionSet,
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
                "two_phase": True,
                "generation_skipped": True,
                "skip_reason": "all pages filtered as empty",
                "node_count": 0,
            },
        )

    nodes = compiler.dedup_content_duplicate_nodes(nodes)
    attach_tags_to_nodes(nodes, chunks)
    edges = build_edges(nodes, extractions)
    quality = _compute_quality_score(nodes, edges)
    toc = [{"title": n.title, "id": n.id, "level": 1} for n in nodes]
    return CompiledWiki(
        nodes=nodes,
        edges=edges,
        toc=toc,
        quality_score=quality,
        metadata={
            WikiField.CHUNK_COUNT: len(chunks),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "two_phase": True,
            "concepts_merged": len(concepts),
            "context_window": cw.max_input,
            "pinned_facts_count": len(pinned_clean),
        },
    )


def gen_cache_deps(
    compiler: Any,
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
    deps["extractions"] = ext_hash or ""
    deps["model"] = compiler._model or ""
    deps["chunk_size_min"] = str(budget.chunk_size_min or "")
    deps["chunk_size_max"] = str(budget.chunk_size_max or "")
    deps["concepts_per_batch"] = str(budget.concepts_per_batch or "")
    deps["pinned_facts"] = pin_hash or ""
    return deps
