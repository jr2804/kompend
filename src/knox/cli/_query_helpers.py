"""Search infrastructure and output formatting for the query command.

Extracted from :mod:`knox.cli.cmds.query` to reduce module size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from knox.agents.search import CragGradeAgent, HydeAgent, QueryExpansionAgent
from knox.cli.args.query import SearchMode
from knox.core.config import ResolvedConfig
from knox.core.config._configs import LlmDataFormat, Verbosity  # type: ignore[attr-defined]
from knox.core.config.keys import (
    COMPILE_EMBEDDING_MODEL,
    COMPILE_LLM_OUTPUT_RETRIES,
    COMPILE_LLM_REASONING_EFFORT,
    COMPILE_LLM_RLM,
    COMPILE_LLM_RLM_MAX_DEPTH,
    COMPILE_LLM_RLM_SUB_MODEL,
    SEARCH_CRAG_ENABLED,
    SEARCH_CRAG_THRESHOLD,
    SEARCH_EXPANSION_ENABLED,
    SEARCH_EXPANSION_VARIANTS,
    SEARCH_HYDE_ENABLED,
    SEARCH_RERANKER_ENABLED,
    SEARCH_RERANKER_MODEL,
    SEARCH_RERANKER_TOP_K,
    SEARCH_SYNTHESIS_TOKENS_BRIEF,
    SEARCH_SYNTHESIS_TOKENS_COMPREHENSIVE,
    SEARCH_SYNTHESIS_TOKENS_STANDARD,
    SEARCH_SYNTHESIS_VERBOSITY,
)
from knox.core.models import CompiledWiki, MetadataField
from knox.core.paths import CAPTURED_SUBDIR, DEFAULT_OUTPUT_DIR_NAME, VECTORS_DIR
from knox.core.trust.actor import resolve_actor
from knox.llm.context_window import ContextWindow
from knox.llm.format import encode_for_llm
from knox.logging import get_console, get_err_console, get_logger
from knox.memory import MemoryStore
from knox.rag.zvec_store import ZvecWikiStore
from knox.search.adaptive_retriever import AdaptiveRetriever
from knox.search.pipeline import EnhancedRetriever
from knox.search.retriever import HybridRetriever, SearchResult, SynthesisResult, synthesize_results

_SYNTHESIS_TOKEN_KEYS: dict[str, str] = {
    Verbosity.BRIEF.value: SEARCH_SYNTHESIS_TOKENS_BRIEF,
    Verbosity.STANDARD.value: SEARCH_SYNTHESIS_TOKENS_STANDARD,
    Verbosity.COMPREHENSIVE.value: SEARCH_SYNTHESIS_TOKENS_COMPREHENSIVE,
}

logger = get_logger(__name__)


def open_vector_store(knox_dir: Path, wiki: CompiledWiki) -> ZvecWikiStore | None:
    """Open the zvec sidecar index built during compile, if it exists."""
    db_file = knox_dir / VECTORS_DIR
    if not db_file.exists():
        return None

    embed_model = wiki.metadata.get(str(MetadataField.EMBEDDING_MODEL))
    if not embed_model:
        logger.debug("No embedding model recorded in wiki metadata; skipping vector store")
        return None
    try:
        return ZvecWikiStore.open(db_file, embedding_model=embed_model)
    except Exception as exc:
        logger.warning("Could not open vector store at %s: %s", db_file, exc, exc_info=True)
        return None


def execute_query_search(
    retriever: HybridRetriever,
    query: str,
    search_mode: str,
    sources_dir: Path | None,
    top_k: int,
    tag_groups: list[list[str]] | None = None,
) -> tuple[list[SearchResult], bool]:
    """Dispatch to the requested search mode and return results + raw flag."""
    raw = False
    expand_graph = search_mode == SearchMode.GRAPH
    mode = SearchMode.HYBRID if search_mode == SearchMode.GRAPH else search_mode

    if mode == SearchMode.RAW:
        if sources_dir is None:
            get_err_console().print("No sources directory found. Use --sources-dir to specify.")
            raise typer.Exit(1)
        results = retriever.search(query, top_k=top_k)
        raw = True
    elif mode == SearchMode.HYBRID:
        results = retriever.search_hybrid(query, sources_dir, top_k=top_k, expand_graph=expand_graph)
    else:
        results = retriever.search_wiki(query, top_k=top_k)

    if tag_groups:
        results = [r for r in results if matches_tag_filter(getattr(r, "tags", set()), tag_groups)]
    return results, raw


def handle_adaptive_search(
    query: str,
    top_k: int,
    search_mode: str,
    vector_store: ZvecWikiStore | None,
    sources_dir: Path | None,
    wiki: CompiledWiki,
    state_dir: Path,
    tag_groups: list[list[str]] | None = None,
) -> tuple[list[SearchResult], bool]:
    """Search using adaptive retriever."""
    retriever = AdaptiveRetriever(wiki, vector_store=vector_store)
    mode = search_mode if search_mode != SearchMode.HYBRID else None
    results = retriever.search(query, top_k=top_k, mode=mode)
    if tag_groups:
        results = [r for r in results if matches_tag_filter(getattr(r, "tags", set()), tag_groups)]
    return results, False


def handle_enhanced_search(
    query: str,
    top_k: int,
    search_mode: str,
    sources_dir: Path | None,
    vector_store: ZvecWikiStore | None,
    state_dir: Path,
    wiki: CompiledWiki,
    cfg: ResolvedConfig,
    model: str,
    tag_groups: list[list[str]] | None = None,
) -> list[SearchResult] | None:
    """Search using enhanced retriever with RAG features. Returns results or None."""
    enhanced = build_enhanced_retriever(wiki, vector_store, state_dir, cfg, model)

    has_enhanced = (
        cfg.get(SEARCH_EXPANSION_ENABLED, False)
        or cfg.get(SEARCH_CRAG_ENABLED, False)
        or cfg.get(SEARCH_RERANKER_ENABLED, False)
        or cfg.get(SEARCH_HYDE_ENABLED, False)
    )

    if has_enhanced:
        expand_graph = search_mode == SearchMode.GRAPH
        results = enhanced.search(query, top_k=top_k, expand_graph=expand_graph)
        if tag_groups:
            results = [r for r in results if matches_tag_filter(getattr(r, "tags", set()), tag_groups)]
        return results

    return None


def matches_tag_filter(node_tags: set[str], tag_groups: list[list[str]]) -> bool:
    """Check if node satisfies at least one AND-group (OR of ANDs)."""
    if not tag_groups:
        return True
    return any(group and all(tag in node_tags for tag in group) for group in tag_groups)


def build_enhanced_retriever(
    wiki: CompiledWiki,
    vector_store: ZvecWikiStore | None,
    state_dir: Path,
    cfg: ResolvedConfig,
    model: str,
) -> EnhancedRetriever:
    """Build an EnhancedRetriever from resolved config."""
    _retries = cfg.get(COMPILE_LLM_OUTPUT_RETRIES)
    expansion_agent = QueryExpansionAgent(model, timeout=120, retries=_retries) if model else None
    crag_agent = CragGradeAgent(model, timeout=120, retries=_retries) if model else None
    hyde_agent = HydeAgent(model, timeout=120, retries=_retries) if model else None
    return EnhancedRetriever(
        wiki,
        vector_store=vector_store,
        expansion_agent=expansion_agent,
        crag_agent=crag_agent,
        hyde_agent=hyde_agent,
        expansion_enabled=cfg.get(SEARCH_EXPANSION_ENABLED, False),
        expansion_variants=cfg.get(SEARCH_EXPANSION_VARIANTS, 3),
        crag_enabled=cfg.get(SEARCH_CRAG_ENABLED, False),
        crag_threshold=cfg.get(SEARCH_CRAG_THRESHOLD, 0.3),
        reranker_enabled=cfg.get(SEARCH_RERANKER_ENABLED, False),
        reranker_model=cfg.get(SEARCH_RERANKER_MODEL, ""),
        reranker_top_k=cfg.get(SEARCH_RERANKER_TOP_K, 20),
        hyde_enabled=cfg.get(SEARCH_HYDE_ENABLED, False),
    )


def synthesize_and_output(
    query: str,
    results: list[SearchResult],
    format: LlmDataFormat,
    has_enhanced: bool,
    enhanced: EnhancedRetriever | None,
    wiki: CompiledWiki,
    model: str,
    cw: ContextWindow | None,
    remember: bool,
    state_dir: Path,
    cfg: ResolvedConfig,
) -> None:
    """Synthesize results and output the answer."""
    results = [r for r in results if not is_captured_source(r.source_file)]
    if not results:
        get_console().print("No relevant content found.")
        raise typer.Exit(0)

    verbosity: str = cfg.get(SEARCH_SYNTHESIS_VERBOSITY, Verbosity.STANDARD.value)
    token_key = _SYNTHESIS_TOKEN_KEYS.get(verbosity, SEARCH_SYNTHESIS_TOKENS_STANDARD)
    target_tokens: int = cfg.get(token_key, 800)
    max_tokens: int = min(target_tokens, cw.max_output) if cw is not None else target_tokens

    try:
        reasoning_effort: str | None = cfg.get(COMPILE_LLM_REASONING_EFFORT)
        rlm: str | bool = cfg.get(COMPILE_LLM_RLM, False)
        rlm_sub_model: str | None = cfg.get(COMPILE_LLM_RLM_SUB_MODEL)
        rlm_depth: int = cfg.get(COMPILE_LLM_RLM_MAX_DEPTH, 0) if rlm else 0
        if has_enhanced and enhanced is not None:
            synthesis = enhanced.synthesize(
                query,
                results,
                verbosity=verbosity,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                rlm=rlm,
                rlm_sub_model=rlm_sub_model,
                rlm_depth=rlm_depth,
            )
        else:
            synthesis = synthesize_results(
                query,
                results,
                model=model,
                verbosity=verbosity,
                max_tokens=max_tokens,
                rlm=rlm,
                rlm_sub_model=rlm_sub_model,
                rlm_depth=rlm_depth,
            )
    except Exception as exc:
        get_err_console().print(f"LLM synthesis failed: {exc}")
        get_err_console().print("Showing raw context instead:")
        print_raw_results(results, LlmDataFormat.MARKDOWN)
        raise typer.Exit(1)

    if format == LlmDataFormat.MARKDOWN:
        get_console().print(synthesis.answer)
        get_console().print("")
        get_console().print("Sources:")
        node_types = node_type_index(wiki)
        for i, c in enumerate(synthesis.citations, 1):
            link = wiki_link(c.get("article_id"), node_types)
            ref = link or f"{DEFAULT_OUTPUT_DIR_NAME}/{c['article']}"
            get_console().print(f"  [{i}] {c['article']} — {ref}", markup=False)
    else:
        get_console().print(encode_structured(structured_answer(synthesis, wiki), format))

    if remember:
        save_query_memory(state_dir, query, synthesis.answer, cfg.get(COMPILE_EMBEDDING_MODEL), memory_by=resolve_actor(cfg))


def is_captured_source(source_file: str | None) -> bool:
    """Return True if *source_file* is a learn/capture note (``_captured/…``)."""
    return bool(source_file) and CAPTURED_SUBDIR in source_file.replace("\\", "/")


def structured_answer(synthesis: SynthesisResult, wiki: CompiledWiki | None = None) -> dict[str, Any]:
    """Build the structured ``{answer, citations}`` payload for agent output."""
    node_types = node_type_index(wiki)
    return {
        "answer": synthesis.answer,
        "citations": [
            {
                "article": c["article"],
                "wiki_link": wiki_link(c.get("article_id"), node_types),
                "source": c.get("source"),
                "snippet": c["snippet"],
                "score": c["score"],
            }
            for c in synthesis.citations
        ],
    }


def node_type_index(wiki: CompiledWiki | None) -> dict[str, str]:
    """Map every wiki node id → its ``node_type`` (the wiki sub-directory)."""
    if wiki is None:
        return {}
    return {n.id: n.node_type for n in wiki.nodes}


def wiki_link(article_id: str | None, node_types: dict[str, str]) -> str | None:
    """Build a relative path into the compiled wiki."""
    if not article_id or article_id not in node_types:
        return None
    return f"{DEFAULT_OUTPUT_DIR_NAME}/{node_types[article_id]}/{article_id}.md"


def print_raw_results(results: list[SearchResult], format: LlmDataFormat) -> None:
    """Print raw search results to console."""
    if format == LlmDataFormat.MARKDOWN:
        for r in results:
            get_console().print(f"  [{r.score:.3f}] {r.title}")
            if r.source_file:
                get_console().print(f"        Source: {r.source_file}")
            get_console().print(f"        {r.snippet}")
            get_console().print("")
        return

    get_console().print(encode_raw_results(results, format))


def encode_raw_results(results: list[SearchResult], fmt: LlmDataFormat) -> str:
    """Build the raw-results payload and encode it for agent consumers."""
    payload = [
        {
            "article": r.title,
            "source": r.source_file,
            "snippet": r.snippet,
            "score": round(r.score, 3),
        }
        for r in results
    ]
    return encode_structured(payload, fmt)


def encode_structured(payload: Any, fmt: LlmDataFormat) -> str:  # noqa: ANN401
    """Encode a structured payload as json/toon/ison for agent consumers."""
    if fmt == LlmDataFormat.JSON:
        return json.dumps(payload, indent=2)
    if fmt == LlmDataFormat.ISON:
        payload = {"answer": [{"text": payload["answer"]}], "citations": payload["citations"]}
    return encode_for_llm(payload, fmt)


def save_query_memory(state_dir: Path, query: str, answer: str, embedding_model: str | None, memory_by: str = "") -> None:
    """Persist a query/answer pair to the memory store."""
    try:
        mem_store = MemoryStore(base_dir=state_dir, embedding_model=embedding_model)
        mem_store.add_entry(query=query, answer=answer, memory_by=memory_by)
        mem_store.close()
        get_err_console().print("\n✓ Query and answer saved to memory.")
    except Exception as exc:
        get_err_console().print(f"\nWarning: Failed to save to memory: {exc}")
