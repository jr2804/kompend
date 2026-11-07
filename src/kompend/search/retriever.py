"""Vector-only retriever for semantic search over compiled wikis.

All retrieval goes through the vector store. FTS5, token overlap,
and BM25 raw-source fallbacks have been removed — if no vector store is
available, search returns empty results (the markdown wiki is still
browsable directly).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from kompend.agents.base import UNSET_RETRIES, UNSET_TIMEOUT
from kompend.agents.deps import QueryDeps
from kompend.agents.failure import record_unit_failure
from kompend.agents.query import QueryAgent
from kompend.agents.query_agentic import make_agentic_query_agent
from kompend.agents.types import QueryAgentDeps, QuerySearchProtocol
from kompend.core.config import ReasoningEffort, RlmConfig
from kompend.core.config._configs import DisplayConfig, LLMConfig
from kompend.core.models import CompiledWiki, WikiNode
from kompend.logging import get_logger
from kompend.rag.vector_store import VectorStoreLike
from kompend.search.retrieval.assemble import extract_snippet, node_figures, result_from_hit
from kompend.search.retrieval.rank import TOKEN_SPLIT_RE, graph_expansion, keyword_search, tokenize
from kompend.search.retrieval.results import Citation, SearchResult, SynthesisResult
from kompend.search.retrieval.vector import vector_search

__all__ = [
    "TOKEN_SPLIT_RE",
    "Citation",
    "HybridRetriever",
    "KeywordRetriever",
    "SearchResult",
    "SynthesisResult",
    "extract_snippet",
    "node_figures",
    "run_agentic_query",
    "synthesize_results",
    "tokenize",
]

_TOKEN_SPLIT_RE = TOKEN_SPLIT_RE  # backward-compat alias


_SOURCE_MARKER_RE = re.compile(r"\[sources/[^\]]+\]|\[\^[^\]]+\]")

logger = get_logger(__name__)


class HybridRetriever:
    """Semantic search over compiled wikis via the vector store.

    Retrieval is purely vector-based. If no vector store is available,
    all search methods return empty lists.
    """

    _RRF_K: int = 60

    def __init__(
        self,
        wiki: CompiledWiki,
        vector_store: VectorStoreLike | None = None,
        display: DisplayConfig | None = None,
    ) -> None:
        self._wiki = wiki
        self._vector_store = vector_store
        self._node_by_id: dict[str, WikiNode] = {n.id: n for n in wiki.nodes}
        self._display = display if display is not None else DisplayConfig()

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        return self.search_wiki(
            query,
            top_k=top_k,
            tags_all=tags_all,
            tags_dnf=tags_dnf,
            attributes_all=attributes_all,
            attribute_keys=attribute_keys,
        )

    def read_page(self, slug: str) -> str | None:
        """Return the full page body for *slug* (a wiki node id), or None.

        Satisfies :class:`kompend.agents.types.QuerySearchProtocol` — the agentic
        query tools call this to fetch a whole page, not just a snippet.
        """
        node = self._node_by_id.get(slug)
        return node.content if node is not None else None

    def search_wiki(
        self,
        query: str,
        top_k: int = 5,
        *,
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        return self._wiki_vector_results(
            query,
            top_k,
            tags_all=tags_all,
            tags_dnf=tags_dnf,
            attributes_all=attributes_all,
            attribute_keys=attribute_keys,
        )

    def search_hybrid(
        self,
        query: str,
        sources_dir: Path | None = None,  # unused, kept for API compat
        top_k: int = 5,
        expand_graph: bool = False,
    ) -> list[SearchResult]:
        results = self.search_wiki(query, top_k=top_k)
        if expand_graph and results:
            results = graph_expansion(
                results,
                edges=self._wiki.edges,
                node_by_id=self._node_by_id,
                top_k=top_k,
                max_chars=self._display.search_snippet_chars,
            )
        return results

    @staticmethod
    def synthesize(
        query: str,
        results: list[SearchResult],
        *,
        verbosity: str = "standard",
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
            skills=skills,
            rlm=rlm,
        )

    def _hit_to_result(self, hit: dict[str, object], query_tokens: set[str]) -> SearchResult:
        """Convert one vector-store hit into a SearchResult, enriching from
        the compiled node when the id resolves.
        """
        node_id = str(hit.get("id", ""))
        return result_from_hit(
            self._node_by_id.get(node_id),
            hit,
            query_tokens,
            snippet_chars=self._display.search_snippet_chars,
        )

    def _wiki_vector_results(
        self,
        query: str,
        top_k: int,
        *,
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        hits = vector_search(
            self._vector_store,
            query,
            top_k=top_k,
            tags_all=tags_all,
            tags_dnf=tags_dnf,
            attributes_all=attributes_all,
            attribute_keys=attribute_keys,
        )
        if hits is None:
            return []
        query_tokens = set(tokenize(query))
        results: list[SearchResult] = []
        for hit in hits:
            node_id = str(hit.get("id", ""))
            if node_id not in self._node_by_id:
                # Markdown-only invariant: the loaded wiki is authoritative; a
                # hit whose id resolves to no page is stale store state.
                logger.debug("Skipping stale vector hit — %r not in loaded wiki", node_id)
                continue
            results.append(self._hit_to_result(hit, query_tokens))
        if not results:
            logger.warning("Vector index returned no matching pages (stale after recompile?). Falling back to keyword search over the wiki.")
            return KeywordRetriever(self._wiki).search(
                query,
                top_k=top_k,
                tags_all=tags_all,
                tags_dnf=tags_dnf,
                attributes_all=attributes_all,
                attribute_keys=attribute_keys,
            )
        return results


class KeywordRetriever:
    """Portable keyword search over a CompiledWiki — no vector store needed.

    Used when the ``.kompend/`` vector store is absent (e.g. a wiki handed over
    without internal state): title tokens score 3x, content tokens 1x.
    Implements the same surface as :class:`HybridRetriever` so the query
    dispatch is unchanged.
    """

    def __init__(self, wiki: CompiledWiki, display: DisplayConfig | None = None) -> None:
        self._nodes = {n.id: n for n in wiki.nodes}
        self._display = display if display is not None else DisplayConfig()

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        """Rank wiki pages by token overlap with *query*, honouring filters."""
        return keyword_search(
            self._nodes.values(),
            query,
            top_k=top_k,
            tags_all=tags_all,
            tags_dnf=tags_dnf,
            attributes_all=attributes_all,
            attribute_keys=attribute_keys,
            display=self._display,
        )

    def search_wiki(
        self,
        query: str,
        top_k: int = 5,
        *,
        tags_all: Sequence[str] | None = None,
        tags_dnf: Sequence[Sequence[str]] | None = None,
        attributes_all: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        """Alias of :meth:`search`."""
        return self.search(
            query,
            top_k=top_k,
            tags_all=tags_all,
            tags_dnf=tags_dnf,
            attributes_all=attributes_all,
            attribute_keys=attribute_keys,
        )

    def read_page(self, slug: str) -> str | None:
        """Return the full page body for *slug* (a wiki node id), or None."""
        node = self._nodes.get(slug)
        return node.content if node is not None else None

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
    skills: dict[str, str] | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    rlm: RlmConfig | None = None,
    query_agent: QueryAgent | None = None,
    llm: LLMConfig | None = None,
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
                    "figures": r.figures,
                }
            )
        return SynthesisResult(answer="", citations=citations)

    context_lines: list[str] = []
    for i, r in enumerate(results, 1):
        context_lines.append(f"[{i}] {r.title}")
        if r.source_file:
            context_lines.append(f"    Source: {r.source_file}")
        # Full page body when available — the snippet window (~300 chars)
        # is display-only and usually misses the passage that answers the
        # query. Fall back to the snippet for graph-expansion-only results.
        body = r.content.strip() or r.snippet
        # Strip inline source markers ([sources/foo.pdf]) — small models
        # mine them and emit citation arrays instead of prose.
        body = _SOURCE_MARKER_RE.sub("", body)
        context_lines.append(f"    {body}")
        context_lines.append("")

    context_text: str = "\n".join(context_lines)

    try:
        if query_agent is not None:
            agent = query_agent
        else:
            agent = QueryAgent(
                model_string=model,
                skills=skills or {},
                reasoning_effort=reasoning_effort,
                rlm=rlm,
                rlm_context_field="context",
                llm=llm,
            )
        deps = QueryDeps(
            question=query,
            context=context_text,
            source_id="query",
            verbosity=verbosity,
        )
        answer = agent.run(deps)
    except Exception as exc:
        # No fallback answer: an empty string would be returned as if it were
        # the model's answer. Every caller (CLI, MCP) already handles a
        # raised synthesis by showing the raw context instead.
        record_unit_failure(stage="search.synthesis", unit=query, exc=exc)
        raise

    citations_out: list[Citation] = []
    for r in results:
        citations_out.append(
            {
                "article": r.title,
                "article_id": r.article_id,
                "source": r.source_file,
                "snippet": r.snippet,
                "score": round(r.score, 3),
                "figures": r.figures,
            }
        )

    return SynthesisResult(answer=answer, citations=citations_out)


def run_agentic_query(
    question: str,
    *,
    retriever: QuerySearchProtocol,
    model: str,
    max_sub_questions: int = 5,
    timeout: float = UNSET_TIMEOUT,
    retries: int = UNSET_RETRIES,
    context_window: int | None = None,
    rlm: RlmConfig | None = None,
    tag_groups: list[list[str]] | None = None,
    attr_filters: list[tuple[str, str | None]] | None = None,
    llm: LLMConfig | None = None,
) -> str:
    """Answer *question* with the agentic query agent against *retriever*.

    Unlike :func:`synthesize_results` — which pre-retrieves and formats
    results for a single-shot :class:`QueryAgent` — this hands the agent the
    question plus the raw retrieval surface and lets it search/read pages via
    tools. Deliberately separate: ``synthesize_results`` builds
    :class:`QueryDeps`, which has no ``retriever``, so an agentic agent run
    there would crash the query tools.
    """
    agent = make_agentic_query_agent(
        model,
        timeout=timeout,
        retries=retries,
        context_window=context_window,
        rlm=rlm,
        llm=llm,
    )
    deps = QueryAgentDeps(
        question=question,
        sub_questions=[],
        retriever=retriever,
        max_sub_questions=max_sub_questions,
        tag_groups=tag_groups or [],
        attr_filters=attr_filters or [],
    )
    return agent.run(deps)


# ── Synthesis (context formatting + LLM call) ──────────────────────────
