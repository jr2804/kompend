"""Enhanced retrieval pipeline with optional RAG techniques.

Wraps the existing :class:`HybridRetriever` with configurable plug-in steps:

1. **Multi-Query Expansion (RAG-Fusion)** — N query variants + RRF merge
2. **CRAG (Corrective RAG)** — LLM relevance grading, drop noise
3. **Neural Re-ranker** — cross-encoder post-fusion ranking
4. **HyDE** — hypothetical answer embeddings for sparse queries

Each step is optional and configurable via ``knox.toml`` under ``[search]``.
All default to **disabled** — opt-in only.
"""

from __future__ import annotations

from pathlib import Path

from sentence_transformers import CrossEncoder as _CrossEncoder

from knox.core.models import CompiledWiki
from knox.core.template_loader import TemplateLoader, get_template_loader
from knox.llm.client import JSON_RESPONSE_FORMAT, LLMClient
from knox.llm.roles import LLMRole
from knox.logging import get_logger
from knox.rag.zvec_store import ZvecWikiStore
from knox.search.retriever import (
    HybridRetriever,
    SearchResult,
    SynthesisResult,
    TokenOverlapRetriever,
)

logger = get_logger(__name__)


class EnhancedRetriever:
    """HybridRetriever wrapper with optional RAG plug-in steps.

    Each step is enabled/disabled via config. The pipeline order is:

    ``MultiQueryExpansion → HybridRetrieval → NeuralRerank → CRAG → Synthesis``
    """

    def __init__(
        self,
        wiki: CompiledWiki,
        *,
        index_file: Path | None = None,
        vector_store: ZvecWikiStore | None = None,
        sources_dir: Path | None = None,
        llm_client: LLMClient | None = None,
        # Multi-query expansion config
        expansion_enabled: bool = False,
        expansion_variants: int = 3,
        # CRAG config
        crag_enabled: bool = False,
        crag_threshold: float = 0.3,
        # Neural re-ranker config
        reranker_enabled: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_top_k: int = 20,
        # HyDE config
        hyde_enabled: bool = False,
    ) -> None:
        self._wiki = wiki
        self._llm_client = llm_client
        self._template_loader: TemplateLoader = get_template_loader()

        # Base retriever
        self._retriever = HybridRetriever(
            wiki,
            index_file=index_file,
            vector_store=vector_store,
            sources_dir=sources_dir,
        )

        # Expansion config
        self._expansion_enabled = expansion_enabled
        self._expansion_variants = expansion_variants

        # CRAG config
        self._crag_enabled = crag_enabled
        self._crag_threshold = crag_threshold

        # Neural re-ranker config
        self._reranker_enabled = reranker_enabled
        self._reranker_model = reranker_model
        self._reranker_top_k = reranker_top_k

        # HyDE config
        self._hyde_enabled = hyde_enabled

    # ── Public API ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        sources_dir: Path | None = None,
        top_k: int = 5,
        expand_graph: bool = False,
    ) -> list[SearchResult]:
        """Run the full enhanced retrieval pipeline.

        Steps (when enabled):
        1. Multi-query expansion (RAG-Fusion)
        2. HyDE hypothetical embedding
        3. Hybrid retrieval (FTS + vector + token-overlap + raw)
        4. Neural re-ranking
        5. CRAG relevance grading
        """
        queries = self._expand_queries(query) if self._expansion_enabled else [query]

        if self._hyde_enabled and self._llm_client is not None:
            hyde_query = self._generate_hypothetical(query)
            queries.append(hyde_query)

        all_results: list[list[SearchResult]] = []
        for q in queries:
            results = self._retriever.search_hybrid(q, sources_dir, top_k=top_k * 2, expand_graph=expand_graph)
            if results:
                all_results.append(results)

        if not all_results:
            return []

        # Fuse multi-query results via RRF
        results = self._fuse_multi_query(all_results, top_k=top_k * 2) if len(all_results) > 1 else all_results[0]

        if self._reranker_enabled and results:
            results = self._neural_rerank(query, results)

        if self._crag_enabled and self._llm_client is not None and results:
            results = self._crag_filter(query, results)

        return results[:top_k]

    def synthesize(
        self,
        query: str,
        results: list[SearchResult],
        *,
        verbosity: str = "standard",
        max_tokens: int | None = None,
        skills: dict[str, str] | None = None,
    ) -> SynthesisResult:
        """Synthesize an answer from search results."""
        synthesizer = TokenOverlapRetriever(self._wiki)
        synthesis = synthesizer.synthesize(
            query,
            results,
            llm_client=self._llm_client,
            verbosity=verbosity,
            max_tokens=max_tokens,
            skills=skills,
        )
        return synthesis

    # ── Multi-Query Expansion (RAG-Fusion) ────────────────────────────────

    def _expand_queries(self, query: str) -> list[str]:
        """Generate N query variants via LLM."""
        if self._llm_client is None:
            return [query]

        try:
            system_prompt = self._template_loader.render("search/expand_query.md.j2", query=query)
            result = self._llm_client.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                role=LLMRole.SEARCH_EXPAND,
                response_format=JSON_RESPONSE_FORMAT,
                temperature=0.0,
            )
            if result.json_data is None:
                return [query]

            expansions: list[str] = [query]
            for key in ("keyword_rewrites", "semantic_rewrites"):
                items = result.json_data.get(key, [])
                if isinstance(items, list):
                    expansions.extend(str(e) for e in items if isinstance(e, str))
            return expansions[: self._expansion_variants + 1]
        except Exception as exc:
            logger.warning("Query expansion failed: %s", exc, exc_info=True)
            return [query]

    # ── HyDE ──────────────────────────────────────────────────────────────

    def _generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical answer passage for sparse queries."""
        try:
            system_prompt = self._template_loader.render("search/hyde_generate.md.j2", query=query)
            result = self._llm_client.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                role=LLMRole.SEARCH_HYDE,
                temperature=0.7,
            )
            hyde_text = result.content.strip()
            if hyde_text:
                logger.debug("HyDE generated hypothetical passage (%d chars)", len(hyde_text))
                return hyde_text
        except Exception as exc:
            logger.warning("HyDE generation failed: %s", exc, exc_info=True)
        return query

    # ── Neural Re-ranker ──────────────────────────────────────────────────

    def _neural_rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        """Re-rank candidates using a cross-encoder model."""
        try:
            reranker = _CrossEncoder(self._reranker_model)
            pairs: list[tuple[str, str]] = []
            for c in candidates:
                text = c.snippet or c.title
                pairs.append((query, text))

            scores = reranker.predict(pairs)
            scored: list[tuple[float, SearchResult]] = []
            for i, c in enumerate(candidates):
                scored.append((float(scores[i]), c))

            scored.sort(key=lambda x: x[0], reverse=True)
            reranked = scored[: self._reranker_top_k]

            results: list[SearchResult] = []
            for score, r in reranked:
                results.append(
                    SearchResult(
                        article_id=r.article_id,
                        title=r.title,
                        snippet=r.snippet,
                        score=round(score, 6),
                        source_file=r.source_file,
                    )
                )
            logger.debug(
                "Neural re-ranker: %d → %d candidates (model=%s)",
                len(candidates),
                len(results),
                self._reranker_model,
            )
            return results
        except Exception as exc:
            logger.warning("Neural re-ranking failed: %s", exc, exc_info=True)
            return candidates

    # ── CRAG (Corrective RAG) ──────────────────────────────────────────────

    def _crag_filter(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        """Grade each candidate for relevance and drop low-scoring ones."""
        if self._llm_client is None:
            return candidates

        filtered: list[SearchResult] = []
        for c in candidates:
            relevance = self._grade_relevance(query, c)
            if relevance >= self._crag_threshold:
                filtered.append(c)
            else:
                logger.debug(
                    "CRAG dropped: %s (relevance=%.2f < threshold=%.2f)",
                    c.title,
                    relevance,
                    self._crag_threshold,
                )

        if not filtered:
            logger.warning("CRAG dropped ALL candidates — returning original set")
            return candidates

        return filtered

    def _grade_relevance(self, query: str, result: SearchResult) -> float:
        """Ask the LLM to grade a single result's relevance on 0-1 scale."""
        try:
            system_prompt = self._template_loader.render(
                "search/crag_grade.md.j2",
                query=query,
                title=result.title,
                snippet=result.snippet,
            )
            llm_result = self._llm_client.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                role=LLMRole.SEARCH_CRAG,
                response_format=JSON_RESPONSE_FORMAT,
                temperature=0.0,
            )
            if llm_result.json_data is not None:
                score = float(llm_result.json_data.get("relevance", 0.5))
                return max(0.0, min(1.0, score))
        except Exception as exc:
            logger.debug("CRAG grading failed for %s: %s", result.title, exc)
        return 0.5  # neutral fallback

    @staticmethod
    def _fuse_multi_query(result_lists: list[list[SearchResult]], top_k: int) -> list[SearchResult]:
        """Fuse multiple result lists via RRF."""
        rrf_k = 60
        scores: dict[str, float] = {}
        representatives: dict[str, SearchResult] = {}
        for result_list in result_lists:
            for rank, result in enumerate(result_list):
                scores[result.article_id] = scores.get(result.article_id, 0.0) + 1.0 / (rrf_k + rank)
                if result.article_id not in representatives:
                    representatives[result.article_id] = result

        fused: list[SearchResult] = []
        for article_id, score in scores.items():
            r = representatives[article_id]
            fused.append(
                SearchResult(
                    article_id=article_id,
                    title=r.title,
                    snippet=r.snippet,
                    score=round(score, 6),
                    source_file=r.source_file,
                )
            )
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:top_k]
