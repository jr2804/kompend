"""Contextual Retrieval enrichment for chunk embeddings.

During T2 chunking, this module prepends a brief situating context paragraph
to each chunk before embedding. Each vector then carries the full document
narrative, not just a fragment — improving retrieval quality for both
compilation and query.

Based on the Contextual Retrieval technique from Anthropic's research:
https://www.anthropic.com/news/contextual-retrieval

Config keys (under ``[search.contextual]``):

- ``enabled`` — enable/disable (default: ``false``)
- ``model`` — LLM model override for context generation (default: uses compile.llm.default_model)

An enrichment failure raises: the context an embedding carries is LLM
output, never a hand-crafted stand-in, and the chunk text reaches the model
in full (`.agents/CODE.md`, "LLM calls").
"""

from __future__ import annotations

from kompend.agents.context_chunk import ContextChunkAgent
from kompend.agents.types import ContextChunk, ContextChunkDeps
from kompend.core.models import ChunkNode, ChunkTree, ElementType, ParsedDocument
from kompend.logging import get_logger

logger = get_logger(__name__)


class ContextualEnricher:
    """Enrich chunk nodes with situating context for better embeddings.

    For each chunk in a ChunkTree, generates a brief paragraph that
    situates the chunk within the full document context. This context
    is prepended to the chunk content before embedding, so the vector
    carries the document narrative rather than just a fragment.
    """

    def __init__(
        self,
        agent: ContextChunkAgent | None = None,
    ) -> None:
        self._agent = agent

    def enrich_tree(
        self,
        tree: ChunkTree,
        doc: ParsedDocument,
    ) -> ChunkTree:
        """Enrich all leaf nodes in a ChunkTree with situating context.

        For each leaf node, generates a context paragraph that describes
        where this chunk sits in the document (surrounding headings,
        document title, position in reading order).

        Returns the same tree with ``.contextual_context`` set on each node.
        """
        if self._agent is None:
            logger.debug("Contextual enricher: no agent, skipping")
            return tree
        agent: ContextChunkAgent = self._agent

        leaves = self._collect_leaves(tree)
        if not leaves:
            return tree

        doc_title = doc.title or doc.source_file.stem
        doc_summary = self._summarize_document(doc)

        return ContextualEnricher._enrich_tree(tree, leaves, doc_title, doc_summary, agent)

    @staticmethod
    def _enrich_tree(
        tree: ChunkTree,
        leaves: list[ChunkNode],
        doc_title: str,
        doc_summary: str,
        agent: ContextChunkAgent,
    ) -> ChunkTree:
        """Enrich all leaves in a single agentic session."""
        chunks: list[ContextChunk] = []
        for leaf in leaves:
            headings = [h for h in (leaf.heading_chain or []) if h]
            if leaf.chunk:
                text = leaf.chunk.text
            elif leaf.structured:
                text = str(leaf.structured)
            else:
                text = ""
            chunks.append(ContextChunk(id=id(leaf), heading_chain=headings, text=text))

        deps = ContextChunkDeps(doc_title=doc_title, doc_summary=doc_summary, chunks=chunks, source_id=doc_title)
        agent.run(deps)

        for leaf in leaves:
            context = deps.results.get(id(leaf))
            if context:
                leaf.contextual_context = context

        logger.debug(
            "Contextual enricher (agentic): enriched %d leaves for %s",
            len(leaves),
            doc_title,
        )
        return tree

    @staticmethod
    def _summarize_document(doc: ParsedDocument) -> str:
        """Build the document's heading outline for the enrichment prompt."""
        elements = doc.elements
        headings = [e.content for e in elements if e.type == ElementType.HEADING and isinstance(e.content, str)]
        if headings:
            return "; ".join(headings)
        return doc.title or doc.source_file.stem

    @staticmethod
    def _collect_leaves(tree: ChunkTree) -> list[ChunkNode]:
        """Collect all leaf nodes from a ChunkTree."""
        leaves: list[ChunkNode] = []

        def _walk(node: ChunkNode) -> None:
            if not node.children_ids:
                leaves.append(node)
            for child_id in node.children_ids:
                child = tree.nodes.get(child_id)
                if child is not None:
                    _walk(child)

        root = tree.nodes.get(tree.root_id)
        if root is not None:
            _walk(root)
        return leaves
