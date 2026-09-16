"""Tier 2 (resolve) orchestration helpers."""

from __future__ import annotations

from pathlib import Path

from knox.core.models import Chunk, ChunkTree, ElementType, ResolvedChunk
from knox.core.pipeline.runtime import PipelineRuntime
from knox.core.resolution.entity_registry import EntityRegistry
from knox.core.resolution.xref_checker import CrossReferenceChecker
from knox.llm.models import resolve_vlm_model
from knox.logging import get_logger

logger = get_logger(__name__)


def resolve_chunks(pipeline: PipelineRuntime, trees: list[ChunkTree]) -> list[ResolvedChunk]:
    """Resolve leaf chunks from trees and build entity registry + xref.

    VLM enrichment (image captions) runs per-leaf when configured.
    Entity registry and cross-reference checks run programmatically
    (no LLM) across the full tree set.

    Resolved chunks are cached under the ``"resolve"`` layer so VLM
    captioning survives a re-run after a mid-pipeline failure.
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

    def _resolve_one(chunk: Chunk) -> ResolvedChunk:
        deps = {"chunk_id": chunk.id, "vlm_model": vlm_model}
        cached = pipeline.cache_store.get_dict("resolve", deps)
        if cached is not None:
            entities_raw = cached.get("entities", [])
            entities = entities_raw if isinstance(entities_raw, list) else []
            relations_raw = cached.get("relations", [])
            relations = relations_raw if isinstance(relations_raw, list) else []
            refs_raw = cached.get("resolved_references", [])
            resolved_references = refs_raw if isinstance(refs_raw, list) else []
            meta_raw = cached.get("metadata", {})
            metadata = meta_raw if isinstance(meta_raw, dict) else {}
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
        pipeline.cache_store.set(
            "resolve",
            deps,
            {
                "entities": result.entities,
                "relations": result.relations,
                "resolved_references": result.resolved_references,
                "metadata": result.metadata,
            },
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
        pipeline.emit("resolve", 0, len(image_chunks), "vlm")
    for i, chunk in enumerate(image_chunks):
        resolved.append(_resolve_one(chunk))
        pipeline.emit("resolve", i + 1, len(image_chunks), "captioning")
    return resolved


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
