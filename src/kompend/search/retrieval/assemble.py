"""Turn vector-store hits and wiki nodes into search results.

Everything here reads from a hit row or a compiled node — no store, no
retriever state. The snippet is display-only; ``content`` is the full page
body the synthesiser reads, and ``figures`` is what the page shows, so a
stage that rebuilds a result must forward both (``src/kompend/search/AGENTS.md``).
"""

from __future__ import annotations

from kompend.core.config._configs import DisplayConfig
from kompend.core.models import ElementType, ImageRef, MetadataField, WikiNode
from kompend.core.models.metadata import meta_str_list_dict
from kompend.core.text.page_media import find_page_figures
from kompend.rag.vector_store import VectorField
from kompend.search.retrieval.results import SearchResult
from kompend.search.retrieval.vector import hit_float, hit_opt_str, hit_str


def result_from_hit(
    node: WikiNode | None,
    hit: dict[str, object],
    query_tokens: set[str],
    *,
    snippet_chars: int,
) -> SearchResult:
    """Convert one vector-store hit into a SearchResult, enriching from
    the compiled node when the id resolves.
    """
    node_id = str(hit.get("id", ""))
    title = hit_str(hit, VectorField.TITLE) or (node.title if node else node_id)
    snippet = extract_snippet(node.content, query_tokens, snippet_chars) if node and node.content else hit_str(hit, VectorField.CONTENT_PREVIEW) or title
    source_file = hit_opt_str(hit, VectorField.SOURCE_FILE)
    if source_file is None and node and node.metadata and node.metadata.get(MetadataField.SOURCE):
        source_file = str(node.metadata[MetadataField.SOURCE])
    return SearchResult(
        article_id=node_id,
        title=title,
        snippet=snippet,
        score=hit_float(hit, "score"),
        source_file=source_file,
        tags=node_tags(node),
        attributes=node_attributes(node),
        content=node.content if node else "",
        figures=node_figures(node),
    )


def extract_snippet(content: str, query_tokens: set[str], max_chars: int | None = None) -> str:
    """Extract a snippet around the first query token match."""
    if max_chars is None:
        max_chars = DisplayConfig().search_snippet_chars
    if not content or not query_tokens:
        if content and len(content) > max_chars:
            return content[:max_chars] + "..."
        return content if content else ""

    lower = content.lower()
    best_pos = len(content)
    for token in query_tokens:
        pos = lower.find(token)
        if 0 <= pos < best_pos:
            best_pos = pos

    if best_pos >= len(content):
        if len(content) > max_chars:
            return content[:max_chars] + "..."
        return content

    start = max(0, best_pos - max_chars // 3)
    end = min(len(content), start + max_chars)
    snippet = content[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    return snippet


def node_figures(node: WikiNode | None) -> list[ImageRef]:
    """The figures the page shows, read from the page body.

    The body carries the published link (relative to the page) and the
    caption as the link's alt text — the node metadata still holds the
    parse-cache path the figure was extracted to, which the reader cannot
    open. A page without figures yields an empty list.
    """
    if node is None or not node.content:
        return []
    source_file = ""
    if node.metadata:
        raw = node.metadata.get(MetadataField.SOURCE)
        if raw:
            source_file = str(raw)
    return [
        ImageRef(
            path=path,
            caption=caption,
            element_type=ElementType.IMAGE.value,
            source_file=source_file,
        )
        for caption, path in find_page_figures(node.content)
    ]


def node_tags(node: WikiNode | None) -> set[str]:
    """Tags from node metadata, guarded against missing node/non-list shapes.

    ``node`` is ``None`` when a vector-store hit references an id absent
    from the loaded wiki (stale index after recompile/rename).
    """
    if node is None or not node.metadata:
        return set()
    raw = node.metadata.get(MetadataField.TAGS)
    if isinstance(raw, list):
        return {str(t) for t in raw}
    return set()


def node_attributes(node: WikiNode | None) -> dict[str, list[str]]:
    """Attributes from node metadata, guarded against missing node."""
    if node is None or not node.metadata:
        return {}
    return meta_str_list_dict(node.metadata.get(MetadataField.ATTRIBUTES))
