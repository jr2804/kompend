"""How the query command finds candidates.

Retrieval helpers: opening the vector store and running
keyword/semantic/adaptive search.
"""

from __future__ import annotations

from pathlib import Path

from kompend.agents.search import CragGradeAgent, HydeAgent, QueryExpansionAgent
from kompend.cli.args.query import SearchMode
from kompend.core.config import ResolvedConfig
from kompend.core.config._configs import DisplayConfig
from kompend.core.models import CompiledWiki, MetadataField
from kompend.core.models.metadata import meta_str
from kompend.core.paths import CACHE_SUBDIR, VECTORS_DB_FILE
from kompend.core.verbosity import is_verbose
from kompend.logging import get_err_console, get_logger
from kompend.rag import build_vector_index
from kompend.rag.sqlite_vec_store import SqliteVecWikiStore
from kompend.search.adaptive_retriever import AdaptiveRetriever
from kompend.search.filters import matches_attr_filter, matches_tag_filter
from kompend.search.pipeline import EnhancedRetriever
from kompend.search.retriever import (
    HybridRetriever,
    KeywordRetriever,
    SearchResult,
)

logger = get_logger(__name__)


def open_vector_store(
    kompend_dir: Path,
    wiki: CompiledWiki,
    *,
    embed_model: str | None = None,
    embed_attachments: bool = False,
) -> SqliteVecWikiStore | None:
    """Open the sidecar vector index built during compile, if it exists.

    The embedding model comes from wiki metadata when available (cache
    path) or falls back to *embed_model* (config value) — a files-loaded
    wiki carries no embedding metadata of its own.

    *embed_attachments* must be the value the index was built with: it is part
    of which nodes the index covers, so a mismatch would rebuild on every query.
    """
    db_file = kompend_dir / VECTORS_DB_FILE
    if not db_file.exists():
        return None

    model = meta_str(wiki.metadata.get(str(MetadataField.EMBEDDING_MODEL))) or embed_model
    if not model:
        logger.debug("No embedding model available; skipping vector store")
        return None

    # Sidecar index can go stale when a wiki is recompiled (new/changed pages)
    # without a matching re-index. Detect via the build-time id manifest and
    # rebuild transparently so queries never silently fall back to keyword.
    if SqliteVecWikiStore(db_file=db_file).is_stale(wiki, embed_attachments=embed_attachments):
        logger.info("Vector index stale (wiki changed) — rebuilding with %s", model)
        try:
            build_vector_index(
                wiki,
                kompend_dir,
                embed_model=model,
                base_dir=kompend_dir.parent,
                embed_attachments=embed_attachments,
            )
        except Exception as exc:
            logger.warning(
                "Could not rebuild stale vector index (%s); queries may fall back to keyword search: %s",
                db_file,
                exc,
                exc_info=is_verbose(),
            )
    try:
        return SqliteVecWikiStore.open(db_file, embedding_model=model)
    except Exception as exc:
        logger.warning("Could not open vector store at %s: %s", db_file, exc, exc_info=is_verbose())
        return None


def execute_query_search(  # noqa: PLR0917
    retriever: HybridRetriever | KeywordRetriever,
    query: str,
    search_mode: str,
    sources_dir: Path | None,
    top_k: int,
    tag_groups: list[list[str]] | None = None,
    attr_filters: list[tuple[str, str | None]] | None = None,
) -> tuple[list[SearchResult], bool]:
    """Dispatch to the requested search mode and return results + raw flag."""
    raw = False
    expand_graph = search_mode == SearchMode.GRAPH
    mode = SearchMode.HYBRID if search_mode == SearchMode.GRAPH else search_mode

    if mode == SearchMode.RAW:
        if sources_dir is None:
            get_err_console().print("No sources directory found. Use --sources-dir to specify.")
            raise SystemExit(1)
        results = retriever.search(query, top_k=top_k)
        raw = True
    elif mode == SearchMode.HYBRID:
        results = retriever.search_hybrid(query, sources_dir, top_k=top_k, expand_graph=expand_graph)
    else:
        results = retriever.search_wiki(query, top_k=top_k)

    if tag_groups:
        results = [r for r in results if matches_tag_filter(getattr(r, "tags", set()), tag_groups)]
    if attr_filters:
        results = [r for r in results if matches_attr_filter(r.attributes, attr_filters)]
    return results, raw


def handle_adaptive_search(  # noqa: PLR0917
    query: str,
    top_k: int,
    search_mode: str,
    vector_store: SqliteVecWikiStore | None,
    sources_dir: Path | None,
    wiki: CompiledWiki,
    state_dir: Path,
    tag_groups: list[list[str]] | None = None,
    attr_filters: list[tuple[str, str | None]] | None = None,
    display: DisplayConfig | None = None,
) -> tuple[list[SearchResult], bool]:
    """Search using adaptive retriever."""
    retriever = AdaptiveRetriever(wiki, vector_store=vector_store, display=display)
    mode = search_mode if search_mode != SearchMode.HYBRID else None
    results = retriever.search(query, top_k=top_k, mode=mode)
    if tag_groups:
        results = [r for r in results if matches_tag_filter(getattr(r, "tags", set()), tag_groups)]
    if attr_filters:
        results = [r for r in results if matches_attr_filter(r.attributes, attr_filters)]
    return results, False


def handle_enhanced_search(  # noqa: PLR0917
    query: str,
    top_k: int,
    search_mode: str,
    sources_dir: Path | None,
    vector_store: SqliteVecWikiStore | None,
    state_dir: Path,
    wiki: CompiledWiki,
    cfg: ResolvedConfig,
    model: str,
    tag_groups: list[list[str]] | None = None,
    attr_filters: list[tuple[str, str | None]] | None = None,
) -> list[SearchResult] | None:
    """Search using enhanced retriever with RAG features. Returns results or None."""
    enhanced = build_enhanced_retriever(wiki, vector_store, state_dir, cfg, model)

    has_enhanced = (
        cfg.kompend.search.expansion.enabled or cfg.kompend.search.crag.enabled or cfg.kompend.search.reranker.enabled or cfg.kompend.search.hyde.enabled
    )

    if has_enhanced:
        expand_graph = search_mode == SearchMode.GRAPH
        results = enhanced.search(query, top_k=top_k, expand_graph=expand_graph)
        if tag_groups:
            results = [r for r in results if matches_tag_filter(getattr(r, "tags", set()), tag_groups)]
        if attr_filters:
            results = [r for r in results if matches_attr_filter(r.attributes, attr_filters)]
        return results

    return None


def build_enhanced_retriever(
    wiki: CompiledWiki,
    vector_store: SqliteVecWikiStore | None,
    state_dir: Path,
    cfg: ResolvedConfig,
    model: str,
) -> EnhancedRetriever:
    """Build an EnhancedRetriever from resolved config."""
    _llm = cfg.kompend.compile.llm
    _budget = cfg.kompend.compile.budget
    expansion_agent = QueryExpansionAgent(model, timeout=_llm.timeout, llm=_llm) if model else None
    crag_agent = CragGradeAgent(model, timeout=_llm.timeout, llm=_llm) if model else None
    hyde_agent = HydeAgent(model, timeout=_llm.timeout, llm=_llm) if model else None
    return EnhancedRetriever(
        wiki,
        vector_store=vector_store,
        expansion_agent=expansion_agent,
        crag_agent=crag_agent,
        hyde_agent=hyde_agent,
        expansion_enabled=cfg.kompend.search.expansion.enabled,
        expansion_variants=cfg.kompend.search.expansion.variants,
        crag_enabled=cfg.kompend.search.crag.enabled,
        crag_threshold=cfg.kompend.search.crag.threshold,
        reranker_enabled=cfg.kompend.search.reranker.enabled,
        reranker_backend=cfg.kompend.search.reranker.backend,
        reranker_model=cfg.kompend.search.reranker.model,
        reranker_top_k=cfg.kompend.search.reranker.top_k,
        reranker_cache_dir=state_dir / CACHE_SUBDIR,
        hyde_enabled=cfg.kompend.search.hyde.enabled,
        llm=_llm,
        display=cfg.kompend.display,
    )
