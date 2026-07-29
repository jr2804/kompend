"""Enhanced retrieval pipeline with optional RAG techniques.

Wraps the existing :class:`HybridRetriever` with configurable plug-in steps:

1. **Multi-Query Expansion (RAG-Fusion)** — N query variants + RRF merge
2. **CRAG (Corrective RAG)** — LLM relevance grading, drop noise
3. **Neural Re-ranker** — bi-encoder cosine-similarity post-fusion ranking
4. **HyDE** — hypothetical answer embeddings for sparse queries

Each step is optional and configurable via ``knox.toml`` under ``[search]``.
All default to **disabled** — opt-in only.
"""

from __future__ import annotations

from knox.agents.deps import PipelineAgentDeps
from knox.agents.search import CragGradeAgent, HydeAgent, QueryExpansionAgent
from knox.compilation.two_phase import cosine_similarity
from knox.core.models import CompiledWiki
from knox.core.templates import TemplateLoader, get_template_loader
from knox.logging import get_logger
from knox.rag.zvec_store import ZvecWikiStore
from knox.search.retriever import (
    HybridRetriever,
    SearchResult,
    SynthesisResult,
    synthesize_results,
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
        vector_store: ZvecWikiStore | None = None,
        expansion_agent: QueryExpansionAgent | None = None,
        crag_agent: CragGradeAgent | None = None,
        hyde_agent: HydeAgent | None = None,
        # Multi-query expansion config
        expansion_enabled: bool = False,
        expansion_variants: int = 3,
        # CRAG config
        crag_enabled: bool = False,
        crag_threshold: float = 0.3,
        # Neural re-ranker config
        reranker_enabled: bool = False,
        reranker_model: str = "",  # kept for config compat; bi-encoder uses vector store model
        reranker_top_k: int = 20,
        # HyDE config
        hyde_enabled: bool = False,
    ) -> None:
        self._wiki = wiki
        self._expansion_agent = expansion_agent
        self._crag_agent = crag_agent
        self._hyde_agent = hyde_agent
        self._template_loader: TemplateLoader = get_template_loader()

        # Base retriever
        self._retriever = HybridRetriever(
            wiki,
            vector_store=vector_store,
        )

        # Expansion config
        self._expansion_enabled = expansion_enabled
        self._expansion_variants = expansion_variants

        # CRAG config
        self._crag_enabled = crag_enabled
        self._crag_threshold = crag_threshold

        # Neural re-ranker config
        self._reranker_enabled = reranker_enabled
        self._reranker_top_k = reranker_top_k

        # HyDE config
        self._hyde_enabled = hyde_enabled
        # Model for synthesis (from any agent's model_string)
        self._synthesis_model = ""
        if expansion_agent is not None:
            self._synthesis_model = expansion_agent.model_string
        elif crag_agent is not None:
            self._synthesis_model = crag_agent.model_string
        elif hyde_agent is not None:
            self._synthesis_model = hyde_agent.model_string

    # ── Public API ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
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

        if self._hyde_enabled and self._hyde_agent is not None:
            hyde_query = self._generate_hypothetical(query)
            queries.append(hyde_query)

        all_results: list[list[SearchResult]] = []
        for q in queries:
            results = self._retriever.search_hybrid(q, top_k=top_k * 2, expand_graph=expand_graph)
            if results:
                all_results.append(results)

        if not all_results:
            return []

        # Fuse multi-query results via RRF
        results = self._fuse_multi_query(all_results, top_k=top_k * 2) if len(all_results) > 1 else all_results[0]

        if self._reranker_enabled and results:
            results = self._neural_rerank(query, results)

        if self._crag_enabled and self._crag_agent is not None and results:
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
        reasoning_effort: str | None = None,
    ) -> SynthesisResult:
        """Synthesize an answer from search results."""
        return synthesize_results(
            query,
            results,
            model=self._synthesis_model,
            verbosity=verbosity,
            max_tokens=max_tokens,
            skills=skills,
            reasoning_effort=reasoning_effort,
        )

    # ── Multi-Query Expansion (RAG-Fusion) ────────────────────────────────

    def _expand_queries(self, query: str) -> list[str]:
        """Generate N query variants via LLM."""
        if self._expansion_agent is None:
            return [query]

        try:
            prompt = self._template_loader.render("search/expand_query.md.j2", query=query)
            deps = PipelineAgentDeps(text=prompt, source_id="expand", timeout=120)
            result = self._expansion_agent.run(deps)

            expansions: list[str] = [query]
            expansions.extend(result.keyword_rewrites)
            expansions.extend(result.semantic_rewrites)
            return expansions[: self._expansion_variants + 1]
        except Exception as exc:
            logger.warning("Query expansion failed: %s", exc, exc_info=True)
            return [query]

    # ── HyDE ──────────────────────────────────────────────────────────────

    def _generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical answer passage for sparse queries."""
        try:
            prompt = self._template_loader.render("search/hyde_generate.md.j2", query=query)
            deps = PipelineAgentDeps(text=prompt, source_id="hyde", timeout=120)
            if self._hyde_agent is None:
                raise RuntimeError("HyDE agent not initialized")
            result = self._hyde_agent.run(deps)
            if result.strip():
                logger.debug("HyDE generated hypothetical passage (%d chars)", len(result))
                return result.strip()
        except Exception as exc:
            logger.warning("HyDE generation failed: %s", exc, exc_info=True)
        return query

    # ── Neural Re-ranker ──────────────────────────────────────────────────

    def _neural_rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        """Re-rank candidates using bi-encoder cosine similarity.

        Uses the vector store's embedding provider (pydantic-ai ``Embedder``)
        to embed the query and each candidate, then sorts by cosine similarity.
        Falls back to the original order if no vector store is available.
        """
        try:
            if self._retriever._vector_store is None:
                logger.debug("Neural re-rank skipped — no vector store")
                return candidates

            provider = self._retriever._vector_store._provider
            if provider is None:
                logger.debug("Neural re-rank skipped — no embedding provider")
                return candidates
            texts = [query] + [c.snippet or c.title for c in candidates]
            vectors = [provider.embed(t) for t in texts]
            query_vec = vectors[0]
            doc_vecs = vectors[1:]

            scored: list[tuple[float, SearchResult]] = []
            for vec, c in zip(doc_vecs, candidates, strict=True):
                score = cosine_similarity(query_vec, vec)
                scored.append((score, c))

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
                        tags=r.tags,
                    )
                )
            logger.debug(
                "Neural re-ranker: %d → %d candidates (bi-encoder cosine)",
                len(candidates),
                len(results),
            )
            return results
        except Exception as exc:
            logger.warning("Neural re-ranking failed: %s", exc, exc_info=True)
            return candidates

    # ── CRAG (Corrective RAG) ──────────────────────────────────────────────

    def _crag_filter(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        """Grade each candidate for relevance and drop low-scoring ones."""
        if self._crag_agent is None:
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
            prompt = self._template_loader.render(
                "search/crag_grade.md.j2",
                query=query,
                title=result.title,
                snippet=result.snippet,
            )
            deps = PipelineAgentDeps(text=prompt, source_id="crag", timeout=120)
            if self._crag_agent is None:
                raise RuntimeError("CRAG agent not initialized")
            grade = self._crag_agent.run(deps)
            return max(0.0, min(1.0, grade.relevance))
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
                    tags=r.tags,
                )
            )
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:top_k]
