"""Tier 2 (resolve) orchestration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knox.core.models import Chunk, ElementType, ResolvedChunk
from knox.core.resolution.entity_registry import EntityRegistry
from knox.core.resolution.xref_checker import CrossReferenceChecker
from knox.logging import get_logger

logger = get_logger(__name__)


def resolve_chunks(pipeline: Any, trees: list[Any]) -> list[Any]:
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
        return [node.chunk for t in trees for node in t.leaves() if node.chunk]

    leaf_chunks = [node.chunk for t in trees for node in t.leaves() if node.chunk]
    resolved: list[Any] = []
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

    vlm_model = pipeline.config.llm.vlm_model if pipeline.config.llm.vlm_model else ""

    def _resolve_one(chunk: Chunk) -> ResolvedChunk:
        deps = {"chunk_id": chunk.id, "vlm_model": vlm_model}
        cached = pipeline._cache_store.get("resolve", deps)
        if cached is not None:
            return ResolvedChunk(
                chunk=chunk,
                entities=cached.get("entities", []),
                relations=cached.get("relations", []),
                resolved_references=cached.get("resolved_references", []),
                metadata=cached.get("metadata", {}),
            )
        if pipeline.resolver is None:
            raise RuntimeError("Resolver not initialized")
        result = pipeline.resolver.resolve(chunk)
        pipeline._cache_store.set(
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
        pipeline._emit("resolve", done, total, "")
    if image_chunks:
        # Signal a distinct VLM sub-phase so the CLI bar switches to
        # "Captioning images" — makes the (slow) VLM run visible instead
        # of looking like a stalled resolve progress bar.
        pipeline._emit("resolve", 0, len(image_chunks), "vlm")
    for i, chunk in enumerate(image_chunks):
        resolved.append(_resolve_one(chunk))
        pipeline._emit("resolve", i + 1, len(image_chunks), "captioning")
    return resolved


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


def apply_tags_to_tree(tree: Any, source_file: Path, source_dir: Path, source_tags: list[str]) -> None:
    """Attach directory-based tags and CLI-provided tags to every leaf chunk in *tree*."""
    dir_tags = extract_source_tags(source_file, source_dir)
    cli_tags = list(source_tags)  # copy
    all_tags = dir_tags + cli_tags
    if all_tags:
        for node in tree.leaves():
            if node.chunk:
                node.chunk.metadata["tags"] = all_tags
