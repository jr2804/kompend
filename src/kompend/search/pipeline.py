"""Enhanced retrieval pipeline with optional RAG techniques.

Wraps the existing :class:`HybridRetriever` with configurable plug-in steps:

1. **Multi-Query Expansion (RAG-Fusion)** — N query variants + RRF merge
2. **CRAG (Corrective RAG)** — LLM relevance grading, drop noise
3. **Neural Re-ranker** — post-fusion ranking, by bi-encoder or cross-encoder
4. **HyDE** — hypothetical answer embeddings for sparse queries

Each step is optional and configurable via ``kompend.toml`` under ``[search]``.
All default to **disabled** — opt-in only.
"""

from __future__ import annotations

from pathlib import Path

from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.failure import record_unit_failure
from kompend.agents.search import CragGradeAgent, HydeAgent, QueryExpansionAgent
from kompend.core.config import ReasoningEffort, RlmConfig
from kompend.core.config._configs import DisplayConfig, LLMConfig, RerankerBackend
from kompend.core.models import CompiledWiki
from kompend.core.templates import TemplateLoader, get_template_loader
from kompend.core.text.matching import cosine as cosine_similarity
from kompend.logging import get_logger
from kompend.rag.rerank import FlashRankReranker, Reranker
from kompend.rag.sqlite_vec_store import SqliteVecWikiStore
from kompend.search.retriever import (
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
        vector_store: SqliteVecWikiStore | None = None,
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
        reranker_backend: RerankerBackend = RerankerBackend.BI_ENCODER,
        reranker_model: str = "",  # the cross-encoder model; empty = the reranker's default
        reranker_top_k: int = 20,
        reranker_cache_dir: Path | None = None,
        # HyDE config
        hyde_enabled: bool = False,
        # LLM/budget config — applied to every agent this retriever owns
        llm: LLMConfig | None = None,
        display: DisplayConfig | None = None,
    ) -> None:
        self._wiki = wiki
        self._expansion_agent = expansion_agent
        self._crag_agent = crag_agent
        self._hyde_agent = hyde_agent
        self._llm = llm
        self._display = display
        self._template_loader: TemplateLoader = get_template_loader()

        # Base retriever
        self._retriever = HybridRetriever(
            wiki,
            vector_store=vector_store,
            display=display,
        )

        # Expansion config
        self._expansion_enabled = expansion_enabled
        self._expansion_variants = expansion_variants

        # CRAG config
        self._crag_enabled = crag_enabled
        self._crag_threshold = crag_threshold

        # Neural re-ranker config
        self._reranker_enabled = reranker_enabled
        self._reranker_backend = reranker_backend
        self._reranker_model = reranker_model
        self._reranker_top_k = reranker_top_k
        self._reranker_cache_dir = reranker_cache_dir
        self._reranker: Reranker | None = None

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
        skills: dict[str, str] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        rlm: RlmConfig | None = None,
    ) -> SynthesisResult:
        """Synthesize an answer from search results."""
        return synthesize_results(
            query,
            results,
            model=self._synthesis_model,
            verbosity=verbosity,
            skills=skills,
            reasoning_effort=reasoning_effort,
            rlm=rlm,
            llm=self._llm,
        )

    # ── Multi-Query Expansion (RAG-Fusion) ────────────────────────────────

    def _expand_queries(self, query: str) -> list[str]:
        """Generate N query variants via LLM."""
        if self._expansion_agent is None:
            return [query]

        try:
            prompt = self._template_loader.render("search/expand_query.md.j2", query=query)
            deps = PipelineAgentDeps(text=prompt, source_id="expand")
            result = self._expansion_agent.run(deps)

            expansions: list[str] = [query]
            expansions.extend(result.keyword_rewrites)
            expansions.extend(result.semantic_rewrites)
            return expansions[: self._expansion_variants + 1]
        except Exception as exc:
            # Best-effort enrichment: the query still has to run, so the
            # fallback value stays. The failure is recorded with its kind.
            record_unit_failure(stage="search.expansion", unit=query, exc=exc)
            return [query]

    # ── HyDE ──────────────────────────────────────────────────────────────

    def _generate_hypothetical(self, query: str) -> str:
        """Generate a hypothetical answer passage for sparse queries."""
        try:
            prompt = self._template_loader.render("search/hyde_generate.md.j2", query=query)
            deps = PipelineAgentDeps(text=prompt, source_id="hyde")
            if self._hyde_agent is None:
                raise RuntimeError("HyDE agent not initialized")
            result = self._hyde_agent.run(deps)
            if result.strip():
                logger.debug("HyDE generated hypothetical passage (%d chars)", len(result))
                return result.strip()
        except Exception as exc:
            record_unit_failure(stage="search.hyde", unit=query, exc=exc)
        return query

    # ── Neural Re-ranker ──────────────────────────────────────────────────

    def _neural_rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        """Re-rank the retrieved candidates.

        The backend decides how: a bi-encoder re-scores with the vector
        store's own embedding model, a cross-encoder (flashrank) reads the
        query and the passage together. The stage is best-effort
        enrichment: on failure it keeps the candidates unchanged.
        """
        if self._reranker_backend == RerankerBackend.NONE:
            return candidates
        if self._reranker_backend == RerankerBackend.FLASHRANK:
            return self._cross_encoder_rerank(query, candidates)
        return self._bi_encoder_rerank(query, candidates)

    def _bi_encoder_rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        """Re-rank candidates using bi-encoder cosine similarity.

        Uses the vector store's embedding provider (pydantic-ai ``Embedder``)
        to embed the query and each candidate, then sorts by cosine similarity.
        Falls back to the original order if no vector store is available.
        """
        try:
            store = self._retriever._vector_store
            provider = store.embedding_provider if isinstance(store, SqliteVecWikiStore) else None
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
            logger.debug(
                "Neural re-ranker: %d → %d candidates (bi-encoder cosine)",
                len(candidates),
                len(reranked),
            )
            return [_reranked_result(score, r) for score, r in reranked]
        except Exception as exc:
            record_unit_failure(stage="search.rerank", unit=query, exc=exc)
            return candidates

    def _cross_encoder_rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        """Re-rank candidates with a cross-encoder (flashrank).

        The cross-encoder reads the query and the page body together, so it
        judges a pair rather than two independent vectors. The passage is the
        page body: the model applies its own context limit, we do not trim.
        """
        try:
            reranker = self._ensure_reranker()
            passages = [r.content or r.snippet or r.title for r in candidates]
            scores = reranker.rerank(query, passages)
            scored = list(zip(scores, candidates, strict=True))
            scored.sort(key=lambda item: item[0], reverse=True)
            reranked = scored[: self._reranker_top_k]
            logger.debug(
                "Neural re-ranker: %d → %d candidates (cross-encoder)",
                len(candidates),
                len(reranked),
            )
            return [_reranked_result(score, r) for score, r in reranked]
        except Exception as exc:
            record_unit_failure(stage="search.rerank", unit=query, exc=exc)
            return candidates

    def _ensure_reranker(self) -> Reranker:
        """Build the cross-encoder once; a missing model is a stage failure."""
        if self._reranker is None:
            self._reranker = FlashRankReranker(
                self._reranker_model,
                cache_dir=self._reranker_cache_dir,
            )
        return self._reranker

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
            deps = PipelineAgentDeps(text=prompt, source_id="crag")
            if self._crag_agent is None:
                raise RuntimeError("CRAG agent not initialized")
            grade = self._crag_agent.run(deps)
            return max(0.0, min(1.0, grade.relevance))
        except Exception as exc:
            # Retrieval enhancement must stay visible: a silent fallback left CRAG
            # returning its neutral 0.5 unobserved. Degrade, but record the failure.
            record_unit_failure(stage="search.crag", unit=result.article_id or result.title, exc=exc)
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
                    attributes=r.attributes,
                    content=r.content,
                    figures=r.figures,
                )
            )
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:top_k]


def _reranked_result(score: float, result: SearchResult) -> SearchResult:
    """A re-ranked copy of *result* — with the page body and figures kept.

    A stage that rebuilds a result forwards ``content`` and ``figures``: a
    stage that drops them hands the synthesis a snippet where the page body
    was meant.
    """
    return SearchResult(
        article_id=result.article_id,
        title=result.title,
        snippet=result.snippet,
        score=round(score, 6),
        source_file=result.source_file,
        tags=result.tags,
        attributes=result.attributes,
        content=result.content,
        figures=result.figures,
    )
