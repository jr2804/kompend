"""Query command and helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from knox.cli._shared import build_cli_overrides, resolve_project_paths
from knox.cli.app import app
from knox.cli.args import (
    ConfigPath,
    QueryAdaptive,
    QueryFormatOption,
    QueryNoClassify,
    QueryRaw,
    QueryRemember,
    QuerySearchMode,
    QueryText,
    QueryTopK,
    QueryVerbosity,
)
from knox.cli.args.query import SearchMode
from knox.core.config import resolve_config
from knox.core.config._configs import LlmDataFormat, QueryFormat, Verbosity
from knox.core.config.keys import (
    COMPILE_EMBEDDING_MODEL,
    QUERY_FORMAT,
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
from knox.core.models import CompiledWiki, load_cached_wiki
from knox.core.paths import CAPTURED_SUBDIR, DEFAULT_OUTPUT_DIR_NAME, SEARCH_DB_FILE, VECTORS_DIR
from knox.core.user import resolve_actor
from knox.llm.client import LLMClient
from knox.llm.format import encode_for_llm
from knox.logging import get_console, get_err_console, get_logger
from knox.memory import MemoryStore
from knox.rag.zvec_store import ZvecWikiStore
from knox.search.adaptive_retriever import AdaptiveRetriever
from knox.search.classifier import classify
from knox.search.pipeline import EnhancedRetriever
from knox.search.retriever import HybridRetriever, TokenOverlapRetriever

logger = get_logger(__name__)

_SYNTHESIS_TOKEN_KEYS: dict[str, str] = {
    Verbosity.BRIEF.value: SEARCH_SYNTHESIS_TOKENS_BRIEF,
    Verbosity.STANDARD.value: SEARCH_SYNTHESIS_TOKENS_STANDARD,
    Verbosity.COMPREHENSIVE.value: SEARCH_SYNTHESIS_TOKENS_COMPREHENSIVE,
}


def _is_captured_source(source_file: str | None) -> bool:
    """Return True if *source_file* is a learn/capture note (``_captured/…``).

    Captured notes remain in the wiki but are not authoritative sources, so
    they are excluded from query citations.
    """
    return bool(source_file) and CAPTURED_SUBDIR in source_file.replace("\\", "/")


def _node_type_index(wiki: CompiledWiki | None) -> dict[str, str]:
    """Map every wiki node id → its ``node_type`` (the wiki sub-directory)."""
    if wiki is None:
        return {}
    return {n.id: n.node_type for n in wiki.nodes}


def _wiki_link(article_id: str | None, node_types: dict[str, str]) -> str | None:
    """Build a relative path into the compiled wiki, e.g.
    ``wiki/concepts/my-concept.md``. Returns ``None`` when the node is unknown.
    """
    if not article_id or article_id not in node_types:
        return None
    return f"{DEFAULT_OUTPUT_DIR_NAME}/{node_types[article_id]}/{article_id}.md"


def _structured_answer(synthesis: Any, wiki: CompiledWiki | None = None) -> dict[str, Any]:  # noqa: ANN401
    """Build the structured ``{answer, citations}`` payload for agent output."""
    node_types = _node_type_index(wiki)
    return {
        "answer": synthesis.answer,
        "citations": [
            {
                "article": c["article"],
                "wiki_link": _wiki_link(c.get("article_id"), node_types),
                "source": c.get("source"),
                "snippet": c["snippet"],
                "score": c["score"],
            }
            for c in synthesis.citations
        ],
    }


def _encode_structured(payload: Any, fmt: QueryFormat) -> str:  # noqa: ANN401
    """Encode a structured payload as json/toon/ison for agent consumers."""
    if fmt == QueryFormat.JSON:
        return json.dumps(payload, indent=2)
    # ISON is table-oriented and drops scalar string fields — wrap the answer
    # in a single-row table so it survives encoding.
    if fmt == QueryFormat.ISON:
        payload = {"answer": [{"text": payload["answer"]}], "citations": payload["citations"]}
    return encode_for_llm(payload, LlmDataFormat(fmt.value))


def _encode_raw_results(results: list[Any], fmt: QueryFormat) -> str:
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
    return _encode_structured(payload, fmt)


def _try_build_llm_client(config: str | None = None) -> LLMClient | None:
    """Build an LLMClient from config if a default model is set."""
    cfg = resolve_config(config_path=config)
    return LLMClient.from_config(cfg)


def _open_vector_store(knox_dir: Path, wiki: CompiledWiki) -> ZvecWikiStore | None:
    """Open the zvec sidecar index built during compile, if it exists."""
    db_file = knox_dir / VECTORS_DIR
    if not db_file.exists():
        return None

    embed_model = wiki.metadata.get("embedding_model")
    if not embed_model:
        logger.debug("No embedding model recorded in wiki metadata; skipping vector store")
        return None
    try:
        return ZvecWikiStore.open(db_file, embedding_model=embed_model)
    except Exception as exc:
        logger.warning("Could not open vector store at %s: %s", db_file, exc, exc_info=True)
        return None


def _print_raw_results(results: list[Any], format: QueryFormat) -> None:
    if format == QueryFormat.TEXT:
        for r in results:
            get_console().print(f"  [{r.score:.3f}] {r.title}")
            if r.source_file:
                get_console().print(f"        Source: {r.source_file}")
            get_console().print(f"        {r.snippet}")
            get_console().print("")
        return

    get_console().print(_encode_raw_results(results, format))


def _classify_intent(query: str, no_classify: bool, format: QueryFormat) -> bool:
    """Run intent classification and respond directly if classified.

    Returns True if the classifier produced a response (caller should return).
    """
    if no_classify:
        return False

    result = classify(query)
    if result.response is None:
        return False
    if format == QueryFormat.TEXT:
        get_console().print(result.response)
    else:
        get_console().print(_encode_structured({"intent": result.intent.value, "response": result.response}, format))
    return True


def _execute_query_search(
    retriever: HybridRetriever,
    query: str,
    search_mode: str,
    sources_dir: Path | None,
    top_k: int,
) -> tuple[list[Any], bool]:
    """Dispatch to the requested search mode and return results + raw flag."""
    raw = False
    expand_graph = search_mode == SearchMode.GRAPH
    mode = SearchMode.HYBRID if search_mode == SearchMode.GRAPH else search_mode

    if mode == SearchMode.RAW:
        if sources_dir is None:
            get_err_console().print("No sources directory found. Use --sources-dir to specify.")
            raise typer.Exit(1)
        results = retriever.search_raw(query, sources_dir, top_k=top_k)
        raw = True
    elif mode == SearchMode.HYBRID:
        results = retriever.search_hybrid(query, sources_dir, top_k=top_k, expand_graph=expand_graph)
    else:
        results = retriever.search_wiki(query, top_k=top_k)
    return results, raw


def _save_query_memory(state_dir: Path, query: str, answer: str, embedding_model: str | None, memory_by: str = "") -> None:
    """Persist a query/answer pair to the memory store."""
    try:
        mem_store = MemoryStore(base_dir=state_dir, embedding_model=embedding_model)
        mem_store.add_entry(query=query, answer=answer, memory_by=memory_by)
        mem_store.close()
        get_err_console().print("\n✓ Query and answer saved to memory.")
    except Exception as exc:
        get_err_console().print(f"\nWarning: Failed to save to memory: {exc}")


def _build_enhanced_retriever(
    wiki: CompiledWiki,
    vector_store: ZvecWikiStore | None,
    sources_dir: Path | None,
    state_dir: Path,
    cfg: Any,
    llm_client: LLMClient | None,
) -> EnhancedRetriever:
    """Build an EnhancedRetriever from resolved config."""
    return EnhancedRetriever(
        wiki,
        index_file=state_dir / SEARCH_DB_FILE,
        vector_store=vector_store,
        sources_dir=sources_dir,
        llm_client=llm_client,
        # Multi-query expansion
        expansion_enabled=cfg.get(SEARCH_EXPANSION_ENABLED, False),
        expansion_variants=cfg.get(SEARCH_EXPANSION_VARIANTS, 3),
        # CRAG
        crag_enabled=cfg.get(SEARCH_CRAG_ENABLED, False),
        crag_threshold=cfg.get(SEARCH_CRAG_THRESHOLD, 0.3),
        # Neural re-ranker
        reranker_enabled=cfg.get(SEARCH_RERANKER_ENABLED, False),
        reranker_model=cfg.get(SEARCH_RERANKER_MODEL, "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        reranker_top_k=cfg.get(SEARCH_RERANKER_TOP_K, 20),
        # HyDE
        hyde_enabled=cfg.get(SEARCH_HYDE_ENABLED, False),
    )


def _handle_adaptive_search(
    query: str,
    top_k: int,
    search_mode: str,
    vector_store: ZvecWikiStore | None,
    sources_dir: Path | None,
    wiki: CompiledWiki,
    state_dir: Path,
) -> tuple[list[Any], bool]:
    """Search using adaptive retriever."""
    retriever = AdaptiveRetriever(
        wiki,
        index_file=state_dir / SEARCH_DB_FILE,
        vector_store=vector_store,
        sources_dir=sources_dir,
    )
    mode = search_mode if search_mode != SearchMode.HYBRID else None
    results = retriever.search(query, top_k=top_k, mode=mode)
    return results, False


def _handle_enhanced_search(
    query: str,
    top_k: int,
    search_mode: str,
    sources_dir: Path | None,
    vector_store: ZvecWikiStore | None,
    state_dir: Path,
    wiki: CompiledWiki,
    cfg: Any,
    llm_client: LLMClient | None,
) -> list[Any] | None:
    """Search using enhanced retriever with RAG features. Returns results or None."""
    enhanced = _build_enhanced_retriever(wiki, vector_store, sources_dir, state_dir, cfg, llm_client)

    has_enhanced = (
        cfg.get(SEARCH_EXPANSION_ENABLED, False)
        or cfg.get(SEARCH_CRAG_ENABLED, False)
        or cfg.get(SEARCH_RERANKER_ENABLED, False)
        or cfg.get(SEARCH_HYDE_ENABLED, False)
    )

    if has_enhanced:
        expand_graph = search_mode == SearchMode.GRAPH
        results = enhanced.search(query, sources_dir, top_k=top_k, expand_graph=expand_graph)
        return results

    return None


def _synthesize_and_output(
    query: str,
    results: list[Any],
    format: QueryFormat,
    has_enhanced: bool,
    enhanced: EnhancedRetriever | None,
    wiki: CompiledWiki,
    llm_client: LLMClient | None,
    remember: bool,
    state_dir: Path,
    cfg: Any,
) -> None:
    """Synthesize results and output the answer."""
    # Exclude learn/capture notes from citations — they are not authoritative
    # sources, so we drop them before synthesis (keeps citation numbering
    # consistent). The notes remain in the wiki for browsing/compile.
    results = [r for r in results if not _is_captured_source(r.source_file)]
    if not results:
        get_console().print("No relevant content found.")
        raise typer.Exit(0)
    verbosity: str = cfg.get(SEARCH_SYNTHESIS_VERBOSITY, Verbosity.STANDARD.value)
    token_key = _SYNTHESIS_TOKEN_KEYS.get(verbosity, SEARCH_SYNTHESIS_TOKENS_STANDARD)
    target_tokens: int = cfg.get(token_key, 800)
    cw = llm_client.context_window if llm_client is not None else None
    max_tokens: int = min(target_tokens, cw.max_output) if cw is not None else target_tokens

    try:
        if has_enhanced and enhanced is not None:
            synthesis = enhanced.synthesize(
                query,
                results,
                verbosity=verbosity,
                max_tokens=max_tokens,
            )
        else:
            synthesizer = TokenOverlapRetriever(wiki)
            synthesis = synthesizer.synthesize(
                query,
                results,
                llm_client=llm_client,
                verbosity=verbosity,
                max_tokens=max_tokens,
            )
    except Exception as exc:
        get_err_console().print(f"LLM synthesis failed: {exc}")
        get_err_console().print("Showing raw context instead:")
        _print_raw_results(results, QueryFormat.TEXT)
        raise typer.Exit(1)

    if format == QueryFormat.TEXT:
        get_console().print(synthesis.answer)
        get_console().print("")
        get_console().print("Sources:")
        node_types = _node_type_index(wiki)
        c: dict[str, Any]
        for i, c in enumerate(synthesis.citations, 1):
            link = _wiki_link(c.get("article_id"), node_types)
            ref = link or f"{DEFAULT_OUTPUT_DIR_NAME}/{c['article']}"
            get_console().print(f"  [{i}] {c['article']} — {ref}", markup=False)
    else:
        get_console().print(_encode_structured(_structured_answer(synthesis, wiki), format))

    if remember:
        _save_query_memory(state_dir, query, synthesis.answer, cfg.get(COMPILE_EMBEDDING_MODEL), memory_by=resolve_actor(cfg))


@app.command(name="query")
def query_cmd(
    ctx: typer.Context,
    query: QueryText,
    top_k: QueryTopK = 5,
    raw: QueryRaw = False,
    format: QueryFormatOption = None,
    config: ConfigPath = None,
    no_classify: QueryNoClassify = False,
    search_mode: QuerySearchMode = SearchMode.HYBRID,
    adaptive: QueryAdaptive = False,
    remember: QueryRemember = False,
    verbosity: QueryVerbosity = None,
) -> None:
    """Ask a natural language question against the compiled wiki.

    Retrieves relevant context from the last compiled wiki and optionally
    synthesizes an answer using the configured LLM. Use --raw to skip LLM
    synthesis and see the source context directly.

    Enhanced RAG features (multi-query expansion, CRAG,
    neural re-ranker, HyDE) are configurable via knox.toml under [search].
    All default to disabled.

    ``--verbosity`` controls answer length (brief|standard|comprehensive);
    token budgets per level are configurable under ``[search.synthesis]``.
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    no_user_tracking = ctx.obj.get("no_user_tracking") if ctx.obj else None
    cli_overrides = build_cli_overrides(
        wiki_dir=wiki_dir_override,
        no_user_tracking=no_user_tracking,
        synthesis_verbosity=verbosity.value if verbosity is not None else None,
    )
    cfg = resolve_config(cli_overrides=cli_overrides, config_path=config)
    if format is None:
        format = QueryFormat(cfg.get(QUERY_FORMAT, QueryFormat.TEXT.value))
    sources_dir, output_dir, _cache_dir, state_dir = resolve_project_paths(cfg)

    if _classify_intent(query, no_classify, format):
        return

    wiki = load_cached_wiki(state_dir.parent)
    if wiki is None:
        get_err_console().print(
            f"No compiled wiki found in {output_dir}. Run 'knox compile' first.",
        )
        raise typer.Exit(1)

    vector_store = _open_vector_store(state_dir, wiki)
    llm_client = _try_build_llm_client(config)

    try:
        if adaptive:
            results, raw = _handle_adaptive_search(query, top_k, search_mode, vector_store, sources_dir, wiki, state_dir)
            has_enhanced = False
        else:
            enhanced_result = _handle_enhanced_search(query, top_k, search_mode, sources_dir, vector_store, state_dir, wiki, cfg, llm_client)
            if enhanced_result is not None:
                results = enhanced_result
                raw = False
                has_enhanced = True
            else:
                retriever = HybridRetriever(
                    wiki,
                    index_file=state_dir / SEARCH_DB_FILE,
                    vector_store=vector_store,
                    sources_dir=sources_dir,
                )
                results, raw = _execute_query_search(retriever, query, search_mode, sources_dir, top_k)
                has_enhanced = False
    finally:
        if vector_store is not None:
            vector_store.close()

    if not results:
        get_console().print("No relevant content found.")
        raise typer.Exit(0)

    if raw:
        _print_raw_results(results, format)
        return

    if llm_client is None:
        get_err_console().print("Warning: No LLM configured. Showing raw context instead.")
        _print_raw_results(results, QueryFormat.TEXT)
        return

    enhanced = _build_enhanced_retriever(wiki, vector_store, sources_dir, state_dir, cfg, llm_client) if has_enhanced else None
    _synthesize_and_output(query, results, format, has_enhanced, enhanced, wiki, llm_client, remember, state_dir, cfg)
