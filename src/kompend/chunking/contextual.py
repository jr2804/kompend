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

from typing import cast

from kompend.agents.context_chunk import ContextChunkAgent
from kompend.agents.context_chunk_agentic import AgenticContextChunkAgent
from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.types import ContextChunk, ContextChunkDeps
from kompend.core.models import ChunkNode, ChunkTree, ElementType, ParsedDocument
from kompend.core.templates import TemplateLoader, get_template_loader
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
        agent: ContextChunkAgent | AgenticContextChunkAgent | None = None,
    ) -> None:
        self._agent = agent
        self._template_loader: TemplateLoader = get_template_loader()

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

        leaves = self._collect_leaves(tree)
        if not leaves:
            return tree

        doc_title = doc.title or doc.source_file.stem
        doc_summary = self._summarize_document(doc)

        if isinstance(self._agent, AgenticContextChunkAgent):
            return self._enrich_tree_agentic(tree, leaves, doc_title, doc_summary)

        for leaf in leaves:
            context = self._build_context(leaf, doc_title, doc_summary)
            leaf.contextual_context = context

        logger.debug(
            "Contextual enricher: enriched %d leaves for %s",
            len(leaves),
            doc_title,
        )
        return tree

    def _enrich_tree_agentic(
        self,
        tree: ChunkTree,
        leaves: list[ChunkNode],
        doc_title: str,
        doc_summary: str,
    ) -> ChunkTree:
        """Enrich all leaves in a single agentic session."""
        if not isinstance(self._agent, AgenticContextChunkAgent):
            raise TypeError("AgenticContextChunkAgent expected")
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
        self._agent.run(deps)

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

    def _build_context(
        self,
        node: ChunkNode,
        doc_title: str,
        doc_summary: str,
    ) -> str:
        """Build situating context for a single chunk node.

        Uses the heading path and document metadata to describe where
        this chunk fits in the document. A model failure raises — there
        is no structural stand-in (`.agents/CODE.md`, "LLM calls").
        """
        heading_chain = " > ".join(h for h in (node.heading_chain or []) if h) or "document root"

        if node.chunk:
            chunk_text = node.chunk.text
        elif node.structured:
            chunk_text = str(node.structured)
        else:
            chunk_text = ""

        rendered = self._template_loader.render(
            "chunking/contextual_context.md.j2",
            doc_title=doc_title,
            doc_summary=doc_summary,
            heading_chain=heading_chain,
            content_preview=chunk_text,
        )
        deps = PipelineAgentDeps(text=rendered, source_id=heading_chain)
        if self._agent is None:
            raise RuntimeError("ContextChunkAgent not initialized")
        agent = cast("ContextChunkAgent", self._agent)
        context = agent.run(deps).strip()
        if not context:
            raise ValueError(f"Contextual enrichment for '{heading_chain}' returned an empty response")
        return context

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
