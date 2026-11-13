"""Search entry point: the two retrievers and the public search surface.

All retrieval goes through the vector store. FTS5, token overlap, and BM25
raw-source fallbacks have been removed: ``HybridRetriever`` asks the store and
drops into ``KeywordRetriever`` only when no store is available or the index
answers nothing (stale after a recompile). The stages themselves live in
:mod:`kompend.search.retrieval` — ``vector`` (the store call), ``assemble``
(hit → result), ``rank`` (store-less scoring, graph expansion) and
``synthesis`` (context formatting, LLM call) — and this module re-exports
their public names so callers have one import site.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from kompend.core.config import ReasoningEffort, RlmConfig
from kompend.core.config._configs import DisplayConfig, LLMConfig
from kompend.core.models import CompiledWiki, WikiNode
from kompend.core.vocab import Verbosity
from kompend.logging import get_logger
from kompend.rag.vector_store import VectorStoreLike
from kompend.search.retrieval.assemble import extract_snippet, node_figures, result_from_hit
from kompend.search.retrieval.rank import TOKEN_SPLIT_RE, graph_expansion, keyword_search, tokenize
from kompend.search.retrieval.results import Citation, SearchResult, SynthesisResult
from kompend.search.retrieval.synthesis import synthesize_results
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
    "synthesize_results",
    "tokenize",
]

logger = get_logger(__name__)


class HybridRetriever:
    """Semantic search over compiled wikis via the vector store.

    Retrieval is purely vector-based. If no vector store is available,
    all search methods return empty lists.
    """

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
        verbosity: str = Verbosity.STANDARD,
        skills: dict[str, str] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        model: str = "",
        rlm: RlmConfig | None = None,
        llm: LLMConfig | None = None,
    ) -> SynthesisResult:
        """Synthesize an answer from search results.

        Forwards every knob ``synthesize_results`` accepts, so the MCP query
        tool and the CLI path honour the same ``reasoning_effort`` and
        ``llm`` settings.
        """
        return synthesize_results(
            query,
            results,
            model=model,
            verbosity=verbosity,
            skills=skills,
            reasoning_effort=reasoning_effort,
            rlm=rlm,
            llm=llm,
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
