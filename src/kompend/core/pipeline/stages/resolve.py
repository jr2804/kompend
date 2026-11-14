"""Tier 2 (resolve) orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from kompend.core.models import Chunk, ChunkTree, ElementType, MetadataField, ResolvedChunk
from kompend.core.pipeline.events import ProgressDetail
from kompend.core.pipeline.parse_cache import content_hash_prefix
from kompend.core.pipeline.runtime import PipelineRuntime
from kompend.core.resolution.entity_registry import EntityRegistry
from kompend.core.resolution.xref_checker import CrossReferenceChecker
from kompend.core.templates import get_template_loader
from kompend.llm.models import resolve_vlm_model
from kompend.logging import get_logger
from kompend.preprocessing.images import is_raster_image
from kompend.resolution import (
    HeuristicConfig,
    ImageRelevance,
    observe_image,
    render_visual_hints,
    verdict_from_observations,
)
from kompend.types import CacheValue

#: The templates whose text decides what the caption prompt says. Their
#: identity is part of the resolve cache key: a changed prompt must not
#: replay a caption the old prompt produced.
_VLM_PROMPT_TEMPLATES = ("vlm/caption.md.j2", "vlm/context.md.j2")

logger = get_logger(__name__)


def resolve_chunks(pipeline: PipelineRuntime, trees: list[ChunkTree]) -> list[ResolvedChunk]:
    """Resolve leaf chunks from trees and build entity registry + xref.

    VLM enrichment (image captions) runs per-leaf when configured.
    Entity registry and cross-reference checks run programmatically
    (no LLM) across the full tree set.

    Resolved chunks are cached under the ``"resolve"`` layer so VLM
    captioning survives a re-run after a mid-pipeline failure. The cache
    key carries the chunk's content hash: ids are positional
    (``<stem>_tN``), so a chunk-size change would otherwise replay stale
    resolutions under the same id.
    """
    registry = EntityRegistry()
    for tree in trees:
        registry.scan_tree(tree)
    checker = CrossReferenceChecker(registry)
    pipeline.xref_result = checker.check_all(trees)
    logger.info(
        "Cross-reference check: %d entities, %d issues",
        pipeline.xref_result.entity_count,
        len(pipeline.xref_result.issues),
    )

    if not pipeline.resolver:
        return [ResolvedChunk(chunk=c) for t in trees for node in t.leaves() if (c := node.chunk)]

    leaf_chunks = [node.chunk for t in trees for node in t.leaves() if node.chunk]
    resolved: list[ResolvedChunk] = []
    total = len(leaf_chunks)

    # Two-pass resolution: non-image chunks first (no VLM), then all
    # image chunks batched together. This keeps the VLM loaded for one
    # contiguous burst instead of reloading it for scattered image
    # chunks across the 1000+ chunk stream (Ollama keep-alive churn).
    non_image = [c for c in leaf_chunks if ElementType.IMAGE not in c.element_types]
    image_chunks = [c for c in leaf_chunks if ElementType.IMAGE in c.element_types]
    if image_chunks:
        logger.info(
            "Resolving %d image chunks in batch (VLM), %d non-image first",
            len(image_chunks),
            len(non_image),
        )

    vlm_model = resolve_vlm_model(pipeline.config.llm.default_model, pipeline.config.llm.vlm_model) or ""
    heuristics = HeuristicConfig.from_pipeline(pipeline.config) if pipeline.config.vlm_heuristic_filter_enabled else None
    image_contexts = _collect_image_contexts(
        trees,
        max_chars=pipeline.config.vlm_context_chars,
        heuristics=heuristics,
    )
    prompt_identity = _vlm_prompt_identity()

    def _resolve_one(chunk: Chunk) -> ResolvedChunk:
        # ``content`` couples the row to the chunk's text. Ids are positional
        # (``<stem>_tN``): a chunk-size change keeps the same ids while the
        # text moves, and an id-only key would replay stale resolutions.
        deps = {
            "chunk_id": chunk.id,
            "content": content_hash_prefix(chunk.text.encode("utf-8")),
            "vlm_model": vlm_model,
        }
        context = image_contexts.get(chunk.id)
        if context is not None:
            # The caption is written against the surroundings, so a changed
            # surrounding (or a changed prompt) must not replay the old
            # caption.
            chunk.metadata[MetadataField.VLM_CONTEXT] = context
            deps["context"] = content_hash_prefix(json.dumps(context, sort_keys=True).encode("utf-8"))
            deps["prompt"] = prompt_identity
        cached = pipeline.cache_store.get_dict("resolve", deps)
        if cached is not None:
            # Validate every element, not just the container: a cache entry is
            # external data, and an entity that is not a mapping would reach
            # the resolver as a broken record.
            entities_raw = cached.get("entities", [])
            entities = [e for e in entities_raw if isinstance(e, dict)] if isinstance(entities_raw, list) else []
            relations_raw = cached.get("relations", [])
            relations = [r for r in relations_raw if isinstance(r, dict)] if isinstance(relations_raw, list) else []
            refs_raw = cached.get("resolved_references", [])
            resolved_references = [r for r in refs_raw if isinstance(r, str)] if isinstance(refs_raw, list) else []
            meta_raw = cached.get("metadata", {})
            metadata = dict(meta_raw) if isinstance(meta_raw, dict) else {}
            chunk.metadata.pop(MetadataField.VLM_CONTEXT, None)
            return ResolvedChunk(
                chunk=chunk,
                entities=entities,
                relations=relations,
                resolved_references=resolved_references,
                metadata=metadata,
            )
        if pipeline.resolver is None:
            raise RuntimeError("Resolver not initialized")
        result = pipeline.resolver.resolve(chunk)
        chunk.metadata.pop(MetadataField.VLM_CONTEXT, None)
        pipeline.cache_store.set(
            "resolve",
            deps,
            # The resolver's entities/relations are JSON-shaped but declared
            # as dict[str, Any], which the cache value union cannot accept
            # positionally (invariant dict values) — erase at this boundary.
            cast(
                "CacheValue",
                {
                    "entities": result.entities,
                    "relations": result.relations,
                    "resolved_references": result.resolved_references,
                    "metadata": result.metadata,
                },
            ),
        )
        return result

    done = 0
    for i, chunk in enumerate(non_image):
        resolved.append(_resolve_one(chunk))
        done = i + 1
        pipeline.emit("resolve", done, total, "")
    if image_chunks:
        # Signal a distinct VLM sub-phase so the CLI bar switches to
        # "Captioning images" — makes the (slow) VLM run visible instead
        # of looking like a stalled resolve progress bar.
        pipeline.emit("resolve", 0, len(image_chunks), ProgressDetail.VLM)
    for i, chunk in enumerate(image_chunks):
        resolved.append(_resolve_one(chunk))
        pipeline.emit("resolve", i + 1, len(image_chunks), ProgressDetail.CAPTIONING)
    _check_resolver_failures(pipeline.resolver)
    return resolved


def _check_resolver_failures(resolver: object) -> None:
    """Raise when a resolver's batch was mostly failures.

    The resolvers that call a model keep a failure log (the shared
    vocabulary in :mod:`kompend.agents.failure`). The log knows the
    thresholds, so this only has to ask.
    """
    for member in getattr(resolver, "resolvers", []):
        log = getattr(member, "failures", None)
        if log is not None:
            log.check()


def _vlm_prompt_identity() -> str:
    """A hash of the templates that decide what the caption prompt says."""
    loader = get_template_loader()
    parts: list[str] = []
    for template in _VLM_PROMPT_TEMPLATES:
        try:
            parts.append(loader.get_text(template))
        except FileNotFoundError:
            parts.append(f"missing:{template}")
    return content_hash_prefix("\n".join(parts).encode("utf-8"))


def _collect_image_contexts(
    trees: list[ChunkTree],
    *,
    max_chars: int,
    heuristics: HeuristicConfig | None = None,
) -> dict[str, dict[str, str]]:
    """What surrounds every image chunk, keyed by chunk id.

    Three pieces, in the order the caption prompt presents them: the
    section the figure sits in, the neighbouring text of the same
    document, and the LLM-written summary of the neighbourhood (present
    only when the contextual enricher ran).

    The neighbouring text is what turns "a diagram" into "a diagram of
    the two-stage pipeline" — a caption without it is a guess.

    When ``heuristics`` is enabled, one bounded observation per raster
    figure adds a factual ``visual_hints`` clause. A confident
    structural/pixel drop also carries its verdict, so the enricher reuses
    this measurement instead of decoding the image a second time.
    """
    contexts: dict[str, dict[str, str]] = {}
    for tree in trees:
        leaves = tree.leaves()
        for index, node in enumerate(leaves):
            chunk = node.chunk
            if chunk is None or ElementType.IMAGE not in chunk.element_types:
                continue
            hints, precomputed = _visual_facts(chunk, heuristics)
            if len(hints) > max_chars:
                hints = hints[:max_chars].rstrip()
            nearby_parts: list[str] = []
            budget = max(0, max_chars - len(hints))
            neighbours = []
            if index > 0:
                neighbours.append(leaves[index - 1])
            if index + 1 < len(leaves):
                neighbours.append(leaves[index + 1])
            for neighbour in neighbours:
                if neighbour.chunk is None or ElementType.IMAGE in neighbour.chunk.element_types:
                    continue
                text = neighbour.chunk.text.strip()
                if not text:
                    continue
                excerpt = text[:budget]
                nearby_parts.append(excerpt)
                budget -= len(excerpt)
                if budget <= 0:
                    break
            context = {
                "section": chunk.heading_context,
                "nearby": "\n".join(nearby_parts),
                "document_context": node.contextual_context,
            }
            if hints:
                context["visual_hints"] = hints
            context.update(precomputed)
            contexts[chunk.id] = context
    return contexts


def _visual_facts(chunk: Chunk, heuristics: HeuristicConfig | None) -> tuple[str, dict[str, str]]:
    """Measure a raster figure once for the prompt hints and the pre-filter.

    Returns the rendered facts and the verdict fields the enricher may
    reuse. Vectors and non-raster paths return empty: their pixels are not
    on disk, and rasterising here would spend the caption budget twice.
    """
    if heuristics is None or not heuristics.enabled:
        return "", {}
    image_file = chunk.metadata.get(MetadataField.IMAGE_FILE)
    if not image_file:
        return "", {}
    path = Path(str(image_file))
    if not is_raster_image(path):
        return "", {}
    observations = observe_image(path, heuristics)
    hints = render_visual_hints(observations, heuristics)
    verdict = verdict_from_observations(observations, heuristics)
    if verdict.decision is not ImageRelevance.DROP or verdict.confidence < heuristics.min_confidence:
        return hints, {}
    return hints, {
        "heuristic_decision": verdict.decision.value,
        "heuristic_reason": verdict.reason,
        "heuristic_confidence": f"{verdict.confidence:.2f}",
        "heuristic_stage": verdict.stage,
    }


def apply_tags_to_tree(tree: ChunkTree, source_file: Path, source_dir: Path, source_tags: list[str]) -> None:
    """Attach directory-based tags and CLI-provided tags to every leaf chunk in *tree*."""
    dir_tags = extract_source_tags(source_file, source_dir)
    cli_tags = list(source_tags)  # copy
    all_tags = dir_tags + cli_tags
    if all_tags:
        for node in tree.leaves():
            if node.chunk:
                node.chunk.metadata["tags"] = all_tags  # nocheck: chunk metadata, not wiki metadata


def extract_source_tags(source_file: Path, source_dir: Path) -> list[str]:
    """Extract directory names as tags from a source file's path within *source_dir*.

    ``sources/2024/conferences/iclr/paper.pdf`` → ``['2024', 'conferences', 'iclr']``

    Returns an empty list when the file is not inside *source_dir* or sits
    at the top level (no parent directories).
    """
    try:
        relative = source_file.relative_to(source_dir)
    except ValueError:
        return []
    parts = list(relative.parts)
    if len(parts) <= 1:
        return []
    return parts[:-1]
