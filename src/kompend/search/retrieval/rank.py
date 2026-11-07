"""Ranking: the store-less keyword scorer and the graph-expansion pass.

Two stages, both pure functions over wiki nodes. Keyword scoring is what a
wiki without a vector store gets; graph expansion widens a result set by one
hop along typed relations, at half score, so a page the query did not name
but the answer needs is still shown.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from kompend.core.config._configs import DisplayConfig
from kompend.core.models import GraphEdge, MetadataField, WikiNode
from kompend.search.filters import matches_attr_filter, matches_tag_filter
from kompend.search.retrieval.assemble import extract_snippet, node_attributes, node_figures, node_tags
from kompend.search.retrieval.results import SearchResult

TOKEN_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"[^a-zA-Z0-9_]+")


def keyword_search(
    nodes: Iterable[WikiNode],
    query: str,
    top_k: int = 5,
    *,
    tags_all: Sequence[str] | None = None,
    tags_dnf: Sequence[Sequence[str]] | None = None,
    attributes_all: Sequence[str] | None = None,
    attribute_keys: Sequence[str] | None = None,
    display: DisplayConfig | None = None,
) -> list[SearchResult]:
    """Rank wiki pages by token overlap with *query*, honouring filters.

    Title tokens score 3x, description tokens 2x, body tokens 1x. Ties sort
    by title so the same query always yields the same order.
    """
    show = display if display is not None else DisplayConfig()
    query_tokens = set(tokenize(query))
    scored: list[tuple[float, WikiNode]] = []
    for node in nodes:
        if not matches_filters(
            node,
            tags_all=tags_all,
            tags_dnf=tags_dnf,
            attributes_all=attributes_all,
            attribute_keys=attribute_keys,
        ):
            continue
        title_tokens = set(tokenize(node.title))
        content_tokens = set(tokenize(node.content or ""))
        desc = node.description
        desc_tokens = set(tokenize(desc))
        score = len(query_tokens & title_tokens) * 3 + len(query_tokens & desc_tokens) * 2 + len(query_tokens & content_tokens)
        if score > 0:
            scored.append((float(score), node))
    scored.sort(key=lambda t: (-t[0], t[1].title.lower()))

    results: list[SearchResult] = []
    for score, node in scored[:top_k]:
        snippet = extract_snippet(node.content, query_tokens, show.search_snippet_chars) if node.content else node.title
        source_file = None
        if node.metadata and node.metadata.get(MetadataField.SOURCE):
            source_file = str(node.metadata[MetadataField.SOURCE])
        results.append(
            SearchResult(
                article_id=node.id,
                title=node.title,
                snippet=snippet,
                score=score,
                source_file=source_file,
                tags=node_tags(node),
                attributes=node_attributes(node),
                content=node.content or "",
                figures=node_figures(node),
            )
        )
    return results


def graph_expansion(
    initial_results: list[SearchResult],
    *,
    edges: Sequence[GraphEdge],
    node_by_id: Mapping[str, WikiNode],
    top_k: int,
    max_chars: int,
) -> list[SearchResult]:
    """Expand search results by following typed relations in the wiki graph."""
    if not initial_results or not edges:
        return initial_results

    expanded: dict[str, SearchResult] = {r.article_id: r for r in initial_results}
    decay_factor = 0.5

    def add_neighbor(neighbor_id: str, seed_score: float) -> None:
        if neighbor_id in expanded:
            return
        node = node_by_id.get(neighbor_id)
        if node is None:
            return
        expanded[neighbor_id] = SearchResult(
            article_id=neighbor_id,
            title=node.title,
            snippet=node.content[:max_chars] + "..." if node.content else "",
            score=seed_score * decay_factor,
            source_file=str(node.metadata.get(MetadataField.SOURCE)) if node.metadata and node.metadata.get(MetadataField.SOURCE) else None,
            tags=node_tags(node),
            attributes=node_attributes(node),
            content=node.content or "",
            figures=node_figures(node),
        )

    for result in initial_results:
        for edge in edges:
            source_id = edge.get("source")
            target_id = edge.get("target")
            if not (isinstance(source_id, str) and isinstance(target_id, str)):
                continue
            if source_id == result.article_id:
                add_neighbor(target_id, result.score)
            elif target_id == result.article_id:
                add_neighbor(source_id, result.score)

    sorted_results = sorted(expanded.values(), key=lambda x: x.score, reverse=True)
    return sorted_results[:top_k]


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens."""
    return [t for t in TOKEN_SPLIT_RE.split(text.lower()) if t]


def matches_filters(
    node: WikiNode,
    *,
    tags_all: Sequence[str] | None = None,
    tags_dnf: Sequence[Sequence[str]] | None = None,
    attributes_all: Sequence[str] | None = None,
    attribute_keys: Sequence[str] | None = None,
) -> bool:
    """Portable (store-less) equivalent of the store-side filter.

    Used by the keyword scorer; the store-backed retriever pushes the same
    filters into the store instead, and the two must agree.
    """
    tags = node_tags(node)
    if tags_all and not all(tag in tags for tag in tags_all):
        return False
    if tags_dnf and not matches_tag_filter(tags, [list(g) for g in tags_dnf]):
        return False
    attr_filters: list[tuple[str, str | None]] = []
    for pair in attributes_all or ():
        key, _, value = pair.partition("=")
        attr_filters.append((key, value))
    for key in attribute_keys or ():
        attr_filters.append((key, None))
    return matches_attr_filter(node_attributes(node), attr_filters)
