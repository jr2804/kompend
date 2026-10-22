"""Context-budget-aware retrieval using the ChunkTree hierarchy.

Retrieval is purely vector-based. Wraps the zvec store with three
adaptive modes:
- ZOOM_IN: retrieve summaries -> identify hot sections -> expand to full leaves
- ZOOM_OUT: too many leaf hits -> collapse to section summaries
- TARGETED: query mentions table/equation -> direct structured query
"""

from __future__ import annotations

from kompend.core.config._configs import DisplayConfig
from kompend.core.models import CompiledWiki, MetadataField, WikiNode
from kompend.core.models.metadata import meta_str_list, meta_str_list_dict
from kompend.logging import get_logger
from kompend.rag.zvec_store import VectorStoreLike, ZvecField
from kompend.search.budget import BudgetPlan, ContextBudgetManager
from kompend.search.retriever import SearchResult, extract_snippet, tokenize

logger = get_logger(__name__)


class AdaptiveRetriever:
    """Context-budget-aware retrieval using the ChunkTree hierarchy.

    Wraps the existing HybridRetriever with three adaptive modes:
    - ZOOM_IN: retrieve summaries -> identify hot sections -> expand to full leaves
    - ZOOM_OUT: too many leaf hits -> collapse to section summaries
    - TARGETED: query mentions table/equation -> direct structured query
    """

    def __init__(
        self,
        wiki: CompiledWiki,
        *,
        vector_store: VectorStoreLike | None = None,
        max_input_tokens: int = 128_000,
        display: DisplayConfig | None = None,
    ) -> None:
        self._wiki = wiki
        self._vector_store = vector_store
        self._node_by_id: dict[str, WikiNode] = {n.id: n for n in wiki.nodes}
        self._budget = ContextBudgetManager(max_input_tokens)
        self._display = display if display is not None else DisplayConfig()

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        mode: str | None = None,
    ) -> list[SearchResult]:
        """Search with adaptive mode selection.

        Args:
            query: Natural language query.
            top_k: Maximum results to return.
            mode: Override retrieval mode (``"zoom_in"``, ``"zoom_out"``,
                ``"targeted"``, or ``None`` for auto-select).

        Returns ranked search results with context-budget-aware selection.
        """
        candidates = self._collect_candidates(query, top_k=top_k * 3)
        if not candidates:
            return []

        plan = self._budget.plan(candidates, query, top_k=top_k)
        if mode is not None:
            plan = self._override_mode(plan, candidates, query, mode, top_k=top_k)

        return self._execute_plan(plan, query, top_k=top_k)

    def _collect_candidates(self, query: str, top_k: int) -> list[WikiNode]:
        """Collect candidate nodes from vector search."""
        seen: set[str] = set()
        candidates: list[WikiNode] = []

        self._dedup_add(self._wiki_vector_results(query, top_k), seen, candidates, self._node_by_id)

        return candidates

    def _execute_plan(
        self,
        plan: BudgetPlan,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """Build SearchResults from a BudgetPlan node_ids. Handles all modes."""
        query_tokens = set(tokenize(query)) if query else set()
        results: list[SearchResult] = []
        for nid in plan.node_ids:
            node = self._node_by_id.get(nid)
            if node is None:
                continue
            snippet_limit = self._display.search_snippet_chars
            if node.content and query_tokens:
                snippet = extract_snippet(node.content, query_tokens, snippet_limit)
            elif node.content:
                snippet = node.content[:snippet_limit]
            else:
                snippet = node.title
            results.append(
                SearchResult(
                    article_id=node.id,
                    title=node.title,
                    snippet=snippet,
                    score=1.0,
                    source_file=str(node.metadata.get(MetadataField.SOURCE, "")) if node.metadata else None,
                    tags=set(node.metadata.get(MetadataField.TAGS, [])) if node.metadata else set(),
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _override_mode(
        self,
        plan: BudgetPlan,
        candidates: list[WikiNode],
        query: str,
        mode: str,
        top_k: int,
    ) -> BudgetPlan:
        if mode == "zoom_in":
            return self._budget.plan_zoom_in(candidates, query, top_k_summaries=top_k, top_k_expand=top_k)
        if mode == "zoom_out":
            return self._budget.plan_zoom_out(candidates, query, max_sections=top_k)
        if mode == "targeted":
            return self._budget.plan_targeted(candidates, query, top_k=top_k)
        return plan

    def _wiki_vector_results(self, query: str, top_k: int) -> list[SearchResult]:
        if self._vector_store is None or not self._vector_store.available:
            return []
        try:
            hits = self._vector_store.search(query, top_k=top_k)
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc, exc_info=True)
            return []
        query_tokens = set(tokenize(query))
        results: list[SearchResult] = []
        for hit in hits:
            node_id = hit.get("id", "")
            node = self._node_by_id.get(node_id)
            title = hit.get(ZvecField.TITLE) or (node.title if node else node_id)
            snippet = (
                extract_snippet(node.content, query_tokens, self._display.search_snippet_chars)
                if node and node.content
                else hit.get(ZvecField.CONTENT_PREVIEW, "") or title
            )
            source_file: str | None = hit.get(ZvecField.SOURCE_FILE)
            if source_file is None and node and node.metadata and node.metadata.get(MetadataField.SOURCE):
                source_file = str(node.metadata[MetadataField.SOURCE])
            results.append(
                SearchResult(
                    article_id=node_id,
                    title=title,
                    snippet=snippet,
                    score=float(hit.get("score", 0.0)),
                    source_file=source_file,
                    tags=set(meta_str_list(node.metadata.get(MetadataField.TAGS))) if node and node.metadata else set(),
                    attributes=meta_str_list_dict(node.metadata.get(MetadataField.ATTRIBUTES)) if node and node.metadata else {},
                )
            )
        return results

    @staticmethod
    def _dedup_add(
        results: list[SearchResult],
        seen: set[str],
        candidates: list[WikiNode],
        node_by_id: dict[str, WikiNode],
    ) -> None:
        """Add unique results to candidates (in-place)."""
        for r in results:
            if r.article_id not in seen:
                seen.add(r.article_id)
                node = node_by_id.get(r.article_id)
                if node is not None:
                    candidates.append(node)
