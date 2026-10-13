"""Tier 3 (compile) and Tier 4 (output) orchestration helpers."""

from __future__ import annotations

from kompend.core.models import (
    ChunkNode,
    ChunkTree,
    CompiledWiki,
    MetadataField,
    WikiNode,
)
from kompend.core.models.wiki import (
    WIKI_SNAPSHOT_COMPILED,
    load_wiki_snapshot,
    write_wiki_snapshot_payload,
)
from kompend.core.pipeline.runtime import PipelineRuntime
from kompend.core.pipeline.stages._parse import parse_and_chunk
from kompend.core.pipeline.stages._resolve import resolve_chunks
from kompend.logging import get_logger
from kompend.trust import filter_wiki

logger = get_logger(__name__)


def run_compile_tier(pipeline: PipelineRuntime) -> CompiledWiki | None:
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


def run_output_tier(pipeline: PipelineRuntime) -> None:
    """Tier 4: write the compiled wiki to the output directory."""
    wiki = load_compiled_wiki(pipeline)
    if wiki is None or not pipeline.output_writer:
        return
    if pipeline.config.privacy_filter:
        wiki, _ = filter_wiki(wiki)
    pipeline.output_writer.write(wiki, pipeline.config.output_dir)


def save_compiled_wiki(pipeline: PipelineRuntime, wiki: CompiledWiki) -> None:
    """Serialize CompiledWiki to the ``compiled`` snapshot in ``kompend.db``."""
    nodes = [
        {
            "id": n.id,
            "title": n.title,
            "content": n.content,
            "node_type": n.node_type,
            "source_chunks": n.source_chunks or [],
            "metadata": pipeline.normalize_metadata_paths(n.metadata),
        }
        for n in wiki.nodes
    ]
    write_wiki_snapshot_payload(
        pipeline.base_directory,
        WIKI_SNAPSHOT_COMPILED,
        nodes=nodes,
        edges=list(wiki.edges),
        toc=list(wiki.toc),
        quality_score=wiki.quality_score,
        iteration=wiki.iteration,
        metadata=dict(wiki.metadata),
    )


def load_compiled_wiki(pipeline: PipelineRuntime) -> CompiledWiki | None:
    """Load the ``compiled`` snapshot from ``kompend.db``. Returns *None* on miss."""
    return load_wiki_snapshot(pipeline.base_directory, WIKI_SNAPSHOT_COMPILED)


def build_wiki_hierarchy(wiki: CompiledWiki, trees: list[ChunkTree]) -> None:
    """Populate WikiNode hierarchy fields from ChunkTree data.

    Maps ChunkTree internal node structure onto WikiNodes by matching
    heading paths. Sets parent_id, children_ids, depth, and heading_chain
    on each WikiNode.

    # ponytail: matching by (heading_context, source_file) from metadata
    # is fragile — resolver/VLM may use different keys. Silent skip on
    # mismatch, no warning. Fix by having ChunkTree expose a stable
    # source_key that all tiers agree on.
    """
    if not wiki.nodes or not trees:
        return

    tree_map: dict[str, ChunkTree] = {}
    for t in trees:
        if t.source_file is not None:
            tree_map[str(t.source_file)] = t

    node_by_heading: dict[tuple[str, str | None], WikiNode] = {}
    for node in wiki.nodes:
        heading = node.metadata.get(MetadataField.HEADING_CONTEXT, "_root")
        source = node.metadata.get(MetadataField.SOURCE_FILE)
        node_by_heading[(heading, source)] = node

    def _link_parent(node: WikiNode, chunk_node: ChunkNode, tree: ChunkTree, source: str | None, node_map: dict[tuple[str, str | None], WikiNode]) -> None:
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
            node.heading_chain = chunk_node.heading_chain
            _link_parent(node, chunk_node, tree, source, node_by_heading)
            break
