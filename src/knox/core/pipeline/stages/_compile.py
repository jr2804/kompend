"""Tier 3 (compile) and Tier 4 (output) orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knox.core.models import (
    CompiledWiki,
    MetadataField,
    WikiField,
    WikiNode,
)
from knox.core.pipeline.stages._pipeline import WikiPipeline
from knox.core.pipeline.stages._parse import parse_and_chunk
from knox.core.pipeline.stages._resolve import resolve_chunks
from knox.logging import get_logger
from knox.trust import filter_wiki

logger = get_logger(__name__)


def run_compile_tier(pipeline: WikiPipeline) -> CompiledWiki | None:
    """Tier 2+3: resolve leaf chunks from ChunkTree and compile into CompiledWiki."""
    trees = parse_and_chunk(pipeline)
    resolved = resolve_chunks(pipeline, trees)
    if not pipeline.compiler:
        return None
    wiki = pipeline.compiler.compile(resolved, trees=trees)
    build_wiki_hierarchy(wiki, trees)
    if pipeline.config.privacy_filter:
        wiki, _ = filter_wiki(wiki)
    save_compiled_wiki(pipeline, wiki)
    return wiki


def run_output_tier(pipeline: WikiPipeline) -> None:
    """Tier 4: write the compiled wiki to the output directory."""
    wiki = load_compiled_wiki(pipeline)
    if wiki is None or not pipeline.output_writer:
        return
    if pipeline.config.privacy_filter:
        wiki, _ = filter_wiki(wiki)
    pipeline.output_writer.write(wiki, pipeline.config.output_dir)


def compiled_wiki_path(pipeline: WikiPipeline) -> Path:
    """Resolve the compiled wiki JSON path."""
    return pipeline.config.paths.compiled_wiki_path


def save_compiled_wiki(pipeline: WikiPipeline, wiki: CompiledWiki) -> None:
    """Serialize CompiledWiki to JSON on disk."""
    path = compiled_wiki_path(pipeline)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        WikiField.NODES: [
            {
                "id": n.id,
                "title": n.title,
                "content": n.content,
                "node_type": n.node_type,
                "source_chunks": n.source_chunks or [],
                "metadata": pipeline._normalize_metadata_paths(n.metadata),
            }
            for n in wiki.nodes
        ],
        WikiField.EDGES: wiki.edges,
        WikiField.TOC: wiki.toc,
        WikiField.QUALITY_SCORE: wiki.quality_score,
        WikiField.ITERATION: wiki.iteration,
        "metadata": wiki.metadata,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_compiled_wiki(pipeline: WikiPipeline) -> CompiledWiki | None:
    """Load CompiledWiki from JSON on disk. Returns *None* on cache miss."""
    path = compiled_wiki_path(pipeline)
    if not path.exists():
        return None
    # Corrupt snapshot is a bug — let it raise so the real cause surfaces
    # instead of silently recompiling from scratch.
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = [WikiNode(**n) for n in data.get(WikiField.NODES, [])]
    return CompiledWiki(
        nodes=nodes,
        edges=data.get(WikiField.EDGES, []),
        toc=data.get(WikiField.TOC, []),
        quality_score=data.get(WikiField.QUALITY_SCORE, 0.0),
        iteration=data.get(WikiField.ITERATION, 0),
        metadata=data.get(WikiField.METADATA, {}),
    )


def build_wiki_hierarchy(wiki: CompiledWiki, trees: list[Any]) -> None:
    """Populate WikiNode hierarchy fields from ChunkTree data.

    Maps ChunkTree internal node structure onto WikiNodes by matching
    heading paths. Sets parent_id, children_ids, depth, and heading_path
    on each WikiNode.

    # ponytail: matching by (heading_context, source_file) from metadata
    # is fragile — resolver/VLM may use different keys. Silent skip on
    # mismatch, no warning. Fix by having ChunkTree expose a stable
    # source_key that all tiers agree on.
    """
    if not wiki.nodes or not trees:
        return

    tree_map: dict[str, Any] = {}
    for t in trees:
        if t.source_file is not None:
            tree_map[str(t.source_file)] = t

    node_by_heading: dict[tuple[str, str | None], WikiNode] = {}
    for node in wiki.nodes:
        heading = node.metadata.get(MetadataField.HEADING_CONTEXT, "_root")
        source = node.metadata.get(MetadataField.SOURCE_FILE)
        node_by_heading[(heading, source)] = node

    def _link_parent(node: Any, chunk_node: Any, tree: Any, source: str | None, node_map: dict) -> None:
        if not chunk_node.parent_id:
            return
        parent_chunk = tree.nodes.get(chunk_node.parent_id)
        if not parent_chunk or not parent_chunk.chunk:
            return
        parent_heading = parent_chunk.chunk.heading_context or "_root"
        parent_node = node_map.get((parent_heading, source))
        if parent_node is None:
            return
        node.parent_id = parent_node.id
        if node.id not in parent_node.children_ids:
            parent_node.children_ids.append(node.id)

    for node in wiki.nodes:
        heading = node.metadata.get(MetadataField.HEADING_CONTEXT, "_root")
        source = node.metadata.get(MetadataField.SOURCE_FILE)
        if source is None:
            continue
        tree = tree_map.get(source)
        if tree is None:
            continue

        for chunk_node in tree.walk():
            if chunk_node.chunk is None:
                continue
            if (chunk_node.chunk.heading_context or "_root") != heading:
                continue
            node.depth = chunk_node.depth
            node.heading_path = chunk_node.heading_path
            _link_parent(node, chunk_node, tree, source, node_by_heading)
            break
