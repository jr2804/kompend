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
"""

from __future__ import annotations

from knox.core.models import ChunkNode, ChunkTree, ElementType, ParsedDocument
from knox.core.template_loader import TemplateLoader, get_template_loader
from knox.llm.client import LLMClient
from knox.logging import get_logger

logger = get_logger(__name__)


class ContextualEnricher:
    """Enrich chunk nodes with situating context for better embeddings.

    For each chunk in a ChunkTree, generates a brief paragraph that
    situates the chunk within the full document context. This context
    is prepended to the chunk content before embedding, so the vector
    carries the document narrative rather than just a fragment.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client
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
        if self._llm_client is None:
            logger.debug("Contextual enricher: no LLM client, skipping")
            return tree

        leaves = self._collect_leaves(tree)
        if not leaves:
            return tree

        doc_title = doc.title or doc.source_file.stem
        doc_summary = self._summarize_document(doc)

        for leaf in leaves:
            context = self._build_context(leaf, doc_title, doc_summary)
            leaf.contextual_context = context

        logger.debug(
            "Contextual enricher: enriched %d leaves for %s",
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
        this chunk fits in the document.
        """
        heading_path = " > ".join(h for h in (node.heading_path or []) if h) or "document root"

        # Estimate chunk content length
        content_preview = ""
        if node.chunk:
            content_preview = node.chunk.text[:200]
        elif node.structured:
            content_preview = str(node.structured)[:200]

        try:
            rendered = self._template_loader.render(
                "chunking/contextual_context.md.j2",
                doc_title=doc_title,
                doc_summary=doc_summary,
                heading_path=heading_path,
                content_preview=content_preview,
            )
            result = self._llm_client.complete(
                messages=[
                    {"role": "system", "content": rendered},
                    {"role": "user", "content": f"Generate context for the chunk at: {heading_path}"},
                ],
                role="chunking.contextual",
                temperature=0.0,
                max_tokens=150,
            )
            context = result.content.strip()
            if context:
                return context
        except Exception as exc:
            logger.debug("Contextual enrichment failed for %s: %s", heading_path, exc)

        # Fallback: structural context without LLM
        return f"This chunk is from '{doc_title}', under section: {heading_path}."

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

    @staticmethod
    def _summarize_document(doc: ParsedDocument) -> str:
        """Generate a one-line summary of the document."""
        elements = doc.elements
        headings = [e.content for e in elements if e.type == ElementType.HEADING and isinstance(e.content, str)]
        if headings:
            return "; ".join(headings[:5])
        return doc.title or doc.source_file.stem
