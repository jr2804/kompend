"""Vector-only retriever for semantic search over compiled wikis.

All retrieval goes through the zvec vector store. FTS5, token overlap,
and BM25 raw-source fallbacks have been removed — if no vector store is
available, search returns empty results (the markdown wiki is still
browsable directly).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, TypedDict

from knox.agents.deps import PipelineAgentDeps, QueryDeps
from knox.agents.query import QueryAgent
from knox.agents.search import QueryExpansionAgent
from knox.core.config import DEFAULTS, RlmConfig
from knox.core.config.keys import DISPLAY_QUERY_EXPANSIONS, DISPLAY_SEARCH_SNIPPET_CHARS
from knox.core.models import CompiledWiki, MetadataField, WikiNode
from knox.core.templates import TemplateLoader
from knox.logging import get_logger
from knox.rag.zvec_store import VectorStoreLike, ZvecField

TOKEN_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"[^a-zA-Z0-9_]+")
_TOKEN_SPLIT_RE = TOKEN_SPLIT_RE  # backward-compat alias

logger = get_logger(__name__)


@dataclass
class SearchResult:
    article_id: str
    title: str
    snippet: str
    score: float
    source_file: str | None = None
    tags: set[str] = field(default_factory=set)


@dataclass
class SynthesisResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)


class Citation(TypedDict):
    """Schema for one citation in a synthesis result."""

    article: str
    article_id: str | None
    source: str | None
    snippet: str
    score: float


# ── HybridRetriever (vector-only) ──────────────────────────────────────


class HybridRetriever:
    """Semantic search over compiled wikis via zvec vector store.

    Retrieval is purely vector-based. If no vector store is available,
    all search methods return empty lists.
    """

    _RRF_K: int = 60

    def __init__(
        self,
        wiki: CompiledWiki,
        vector_store: VectorStoreLike | None = None,
    ) -> None:
        self._wiki = wiki
        self._vector_store = vector_store
        self._node_by_id: dict[str, WikiNode] = {n.id: n for n in wiki.nodes}

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return self.search_wiki(query, top_k=top_k)

    def search_wiki(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return self._wiki_vector_results(query, top_k)

    def search_hybrid(
        self,
        query: str,
        sources_dir: Path | None = None,  # unused, kept for API compat
        top_k: int = 5,
        expand_graph: bool = False,
    ) -> list[SearchResult]:
        results = self.search_wiki(query, top_k=top_k)
        if expand_graph and results:
            results = self._graph_expansion_results(results, top_k=top_k)
        return results

    @staticmethod
    def synthesize(
        query: str,
        results: list[SearchResult],
        *,
        verbosity: str = "standard",
        max_tokens: int | None = None,
        skills: dict[str, str] | None = None,
        model: str = "",
        rlm: RlmConfig | None = None,
    ) -> SynthesisResult:
        """Synthesize an answer from search results."""
        return synthesize_results(
            query,
            results,
            model=model,
            verbosity=verbosity,
            max_tokens=max_tokens,
            skills=skills,
            rlm=rlm,
        )

    def _hit_to_result(self, hit: dict[str, object], query_tokens: set[str]) -> SearchResult:
        """Convert one vector-store hit into a SearchResult, enriching from
        the compiled node when the id resolves.
        """
        node_id = str(hit.get("id", ""))
        node = self._node_by_id.get(node_id)
        title = hit.get(ZvecField.TITLE) or (node.title if node else node_id)
        snippet = extract_snippet(node.content, query_tokens) if node and node.content else hit.get(ZvecField.CONTENT_PREVIEW, "") or title
        source_file: str | None = hit.get(ZvecField.SOURCE_FILE)
        if source_file is None and node and node.metadata and node.metadata.get(MetadataField.SOURCE):
            source_file = str(node.metadata[MetadataField.SOURCE])
        return SearchResult(
            article_id=node_id,
            title=title,
            snippet=snippet,
            score=float(hit.get("score", 0.0)),
            source_file=source_file,
            tags=_node_tags(node),
        )

    def _wiki_vector_results(self, query: str, top_k: int) -> list[SearchResult]:
        if self._vector_store is None or not self._vector_store.available:
            return []
        try:
            hits = self._vector_store.search(query, top_k=top_k)
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc, exc_info=True)
            return []
        query_tokens = set(tokenize(query))
        return [self._hit_to_result(hit, query_tokens) for hit in hits]

    def _graph_expansion_results(self, initial_results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Expand search results by following typed relations in the wiki graph."""
        if not initial_results or not self._wiki.edges:
            return initial_results

        expanded: dict[str, SearchResult] = {r.article_id: r for r in initial_results}
        decay_factor = 0.5
        max_chars = DEFAULTS.get(DISPLAY_SEARCH_SNIPPET_CHARS, 300)

        def add_neighbor(neighbor_id: str, seed_score: float) -> None:
            if neighbor_id in expanded:
                return
            node = self._node_by_id.get(neighbor_id)
            if node is None:
                return
            expanded[neighbor_id] = SearchResult(
                article_id=neighbor_id,
                title=node.title,
                snippet=node.content[:max_chars] + "..." if node.content else "",
                score=seed_score * decay_factor,
                source_file=str(node.metadata.get(MetadataField.SOURCE)) if node.metadata and node.metadata.get(MetadataField.SOURCE) else None,
                tags=_node_tags(node),
            )

        for result in initial_results:
            for edge in self._wiki.edges:
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


class KeywordRetriever:
    """Portable keyword search over a CompiledWiki — no vector store needed.

    Used when the ``.knox/`` vector store is absent (e.g. a wiki handed over
    without internal state): title tokens score 3x, content tokens 1x.
    Implements the same surface as :class:`HybridRetriever` so the query
    dispatch is unchanged.
    """

    def __init__(self, wiki: CompiledWiki) -> None:
        self._nodes = {n.id: n for n in wiki.nodes}

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Rank wiki pages by token overlap with *query*."""
        query_tokens = set(tokenize(query))
        scored: list[tuple[float, WikiNode]] = []
        for node in self._nodes.values():
            title_tokens = set(tokenize(node.title))
            content_tokens = set(tokenize(node.content or ""))
            desc = (node.metadata or {}).get("description", "") if node.metadata else ""
            desc_tokens = set(tokenize(str(desc)))
            score = len(query_tokens & title_tokens) * 3 + len(query_tokens & desc_tokens) * 2 + len(query_tokens & content_tokens)
            if score > 0:
                scored.append((float(score), node))
        scored.sort(key=lambda t: (-t[0], t[1].title.lower()))

        results: list[SearchResult] = []
        for score, node in scored[:top_k]:
            snippet = extract_snippet(node.content, query_tokens) if node.content else node.title
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
                    tags=_node_tags(node),
                )
            )
        return results

    def search_wiki(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Alias of :meth:`search`."""
        return self.search(query, top_k=top_k)

    def search_hybrid(
        self,
        query: str,
        sources_dir: Path | None = None,
        top_k: int = 5,
        expand_graph: bool = False,
    ) -> list[SearchResult]:
        """Alias of :meth:`search` — no raw-source fallback in portable mode."""
        return self.search(query, top_k=top_k)


# ── Synthesis (context formatting + LLM call) ──────────────────────────


def synthesize_results(
    query: str,
    results: list[SearchResult],
    model: str = "",
    *,
    verbosity: str = "standard",
    max_tokens: int | None = None,
    skills: dict[str, str] | None = None,
    reasoning_effort: str | None = None,
    rlm: RlmConfig | None = None,
) -> SynthesisResult:
    """Format search results into context and call the LLM for an answer.

    This is a standalone function (no retriever needed) — it takes already-
    retrieved ``SearchResult`` objects, formats them as numbered context
    lines, and passes them to :class:`QueryAgent`. The ``rlm`` kwarg enables
    RLM code-exploration over the synthesized context
    (``compile.llm.rlm`` config).
    """
    if not model:
        citations: list[Citation] = []
        for r in results:
            citations.append(
                {
                    "article": r.title,
                    "article_id": r.article_id,
                    "source": r.source_file,
                    "snippet": r.snippet,
                    "score": round(r.score, 3),
                }
            )
        return SynthesisResult(answer="", citations=citations)

    context_lines: list[str] = []
    for i, r in enumerate(results, 1):
        context_lines.append(f"[{i}] {r.title}")
        if r.source_file:
            context_lines.append(f"    Source: {r.source_file}")
        context_lines.append(f"    {r.snippet}")
        context_lines.append("")

    context_text: str = "\n".join(context_lines)

    try:
        agent = QueryAgent(
            model_string=model,
            timeout=120.0,
            skills=skills or {},
            reasoning_effort=reasoning_effort,
            rlm=rlm,
            rlm_context_field="context",
        )
        deps = QueryDeps(
            question=query,
            context=context_text,
            source_id="query",
            verbosity=verbosity,
        )
        answer = agent.run(deps)
    except Exception as exc:
        logger.warning("Query synthesis failed: %s", exc, exc_info=True)
        answer = ""

    citations_out: list[Citation] = []
    for r in results:
        citations_out.append(
            {
                "article": r.title,
                "article_id": r.article_id,
                "source": r.source_file,
                "snippet": r.snippet,
                "score": round(r.score, 3),
            }
        )

    return SynthesisResult(answer=answer, citations=citations_out)


# ── Query expansion ────────────────────────────────────────────────────


def expand_query(
    query: str,
    template_loader: TemplateLoader,
    expansion_agent: QueryExpansionAgent | None = None,
) -> list[str]:
    """Expand a query into variants using the expansion agent."""
    if expansion_agent is None:
        return [query]

    try:
        system_prompt = template_loader.render("search/expand_query.md.j2", query=query)
        deps = PipelineAgentDeps(text=system_prompt, source_id="expand", timeout=120)
        result = expansion_agent.run(deps)

        expansions: list[str] = [query]
        expansions.extend(result.keyword_rewrites)
        expansions.extend(result.semantic_rewrites)
        expansions_limit = DEFAULTS.get(DISPLAY_QUERY_EXPANSIONS, 7)
        return expansions[:expansions_limit]
    except Exception as exc:
        logger.warning("Query expansion failed: %s", exc, exc_info=True)
        return [query]


# ── Utilities ──────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens."""
    return [t for t in TOKEN_SPLIT_RE.split(text.lower()) if t]


def _node_tags(node: WikiNode) -> set[str]:
    """Tags from node metadata, guarded against missing/non-list shapes."""
    if not node.metadata:
        return set()
    raw = node.metadata.get(MetadataField.TAGS)
    if isinstance(raw, list):
        return {str(t) for t in raw}
    return set()


def extract_snippet(content: str, query_tokens: set[str], max_chars: int | None = None) -> str:
    """Extract a snippet around the first query token match."""
    if max_chars is None:
        max_chars = DEFAULTS.get(DISPLAY_SEARCH_SNIPPET_CHARS, 300)
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
