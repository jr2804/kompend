"""Query command and helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer

from knox.cli._shared import build_cli_overrides, load_hooks, resolve_project_paths
from knox.cli.app import app
from knox.cli.args import (
    ConfigPath,
    QueryAdaptive,
    QueryFormat,
    QueryNoClassify,
    QueryRaw,
    QuerySearchMode,
    QueryText,
    QueryTopK,
)
from knox.core.config import resolve_config
from knox.core.config.keys import (
    COMPILE_EMBEDDING_MODEL,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_FALLBACK_CHAIN,
    COMPILE_LLM_ROLE_MODELS,
    SEARCH_CACHE_ENABLED,
    SEARCH_CACHE_THRESHOLD,
    SEARCH_CACHE_MAX_ENTRIES,
    SEARCH_EXPANSION_ENABLED,
    SEARCH_EXPANSION_VARIANTS,
    SEARCH_CRAG_ENABLED,
    SEARCH_CRAG_THRESHOLD,
    SEARCH_RERANKER_ENABLED,
    SEARCH_RERANKER_MODEL,
    SEARCH_RERANKER_TOP_K,
    SEARCH_HYDE_ENABLED,
    SEARCH_CONTEXTUAL_ENABLED,
    SEARCH_CONTEXTUAL_MODEL,
)
from knox.core.hooks import HookEvent, fire_hook
from knox.core.models import CompiledWiki, load_cached_wiki
from knox.core.paths import SEARCH_DB_FILE, VECTORS_DIR
from knox.core.user import resolve_actor
from knox.llm.client import LLMClient
from knox.memory import MemoryStore
from knox.rag.zvec_store import ZvecWikiStore
from knox.search.adaptive_retriever import AdaptiveRetriever
from knox.search.classifier import classify
from knox.search.pipeline import EnhancedRetriever
from knox.search.retriever import HybridRetriever, TokenOverlapRetriever

logger = logging.getLogger(__name__)


def format_json_output(synthesis: Any, results: list[Any]) -> str:  # noqa: ANN401
    """Format query result as JSON string."""
    output = {
        "answer": synthesis.answer,
        "citations": [
            {
                "article": c["article"],
                "source": c.get("source"),
                "snippet": c["snippet"],
                "score": c["score"],
            }
            for c in synthesis.citations
        ],
    }
    return json.dumps(output, indent=2)


def _try_build_llm_client(config: str | None = None) -> LLMClient | None:
    """Build an LLMClient from config if a default model is set."""
    cfg = resolve_config(config_path=config)
    default_model: str | None = cfg.get(COMPILE_LLM_DEFAULT_MODEL)
    if not default_model:
        return None

    return LLMClient(
        default_model=default_model,
        fallback_chain=list(cfg.get(COMPILE_LLM_FALLBACK_CHAIN, [])),
        role_models=dict(cfg.get(COMPILE_LLM_ROLE_MODELS, {})),
    )


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


def _print_raw_results(results: list[Any], format: str) -> None:
    if format == "json":
        output = [
            {
                "article": r.title,
                "source": r.source_file,
                "snippet": r.snippet,
                "score": round(r.score, 3),
            }
            for r in results
        ]
        typer.echo(json.dumps(output, indent=2))
    else:
        for r in results:
            typer.echo(f"  [{r.score:.3f}] {r.title}")
            if r.source_file:
                typer.echo(f"        Source: {r.source_file}")
            typer.echo(f"        {r.snippet}")
            typer.echo("")


def _classify_intent(query: str, no_classify: bool, format: str) -> bool:
    """Run intent classification and respond directly if classified.

    Returns True if the classifier produced a response (caller should return).
    """
    if no_classify:
        return False

    result = classify(query)
    if result.response is None:
        return False
    if format == "json":
        typer.echo(
            json.dumps(
                {
                    "intent": result.intent.value,
                    "response": result.response,
                },
                indent=2,
            )
        )
    else:
        typer.echo(result.response)
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
    expand_graph = search_mode == "graph"
    mode = "hybrid" if search_mode == "graph" else search_mode

    if mode == "raw":
        if sources_dir is None:
            typer.echo("No sources directory found. Use --sources-dir to specify.", err=True)
            raise typer.Exit(1)
        results = retriever.search_raw(query, sources_dir, top_k=top_k)
        raw = True
    elif mode == "hybrid":
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
        typer.echo("\n✓ Query and answer saved to memory.", err=True)
    except Exception as exc:
        typer.echo(f"\nWarning: Failed to save to memory: {exc}", err=True)


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
        # Semantic cache
        cache_enabled=cfg.get(SEARCH_CACHE_ENABLED, True),
        cache_threshold=cfg.get(SEARCH_CACHE_THRESHOLD, 0.92),
        cache_max_entries=cfg.get(SEARCH_CACHE_MAX_ENTRIES, 500),
        cache_dir=state_dir,
        cache_embedding_model=cfg.get(COMPILE_EMBEDDING_MODEL),
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


@app.command(name="query")
def query_cmd(
    ctx: typer.Context,
    query: QueryText,
    top_k: QueryTopK = 5,
    raw: QueryRaw = False,
    format: QueryFormat = "text",
    config: ConfigPath = None,
    no_classify: QueryNoClassify = False,
    search_mode: QuerySearchMode = "hybrid",
    adaptive: QueryAdaptive = False,
    remember: bool = typer.Option(False, "--remember", help="Save this query and synthesized answer to the memory store."),
) -> None:
    """Ask a natural language question against the compiled wiki.

    Retrieves relevant context from the last compiled wiki and optionally
    synthesizes an answer using the configured LLM. Use --raw to skip LLM
    synthesis and see the source context directly.

    Enhanced RAG features (semantic cache, multi-query expansion, CRAG,
    neural re-ranker, HyDE) are configurable via knox.toml under [search].
    All default to disabled except semantic cache.
    """
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    no_user_tracking = ctx.obj.get("no_user_tracking") if ctx.obj else None
    cli_overrides = build_cli_overrides(wiki_dir=wiki_dir_override, no_user_tracking=no_user_tracking)
    cfg = resolve_config(cli_overrides=cli_overrides, config_path=config)
    sources_dir, output_dir, _cache_dir, state_dir = resolve_project_paths(cfg)
    load_hooks()

    if _classify_intent(query, no_classify, format):
        return

    wiki = load_cached_wiki(state_dir.parent)
    if wiki is None:
        typer.echo(
            f"No compiled wiki found in {output_dir}. Run 'knox compile' first.",
            err=True,
        )
        raise typer.Exit(1)

    vector_store = _open_vector_store(state_dir, wiki)
    llm_client = _try_build_llm_client(config)

    try:
        if adaptive:
            retriever = AdaptiveRetriever(
                wiki,
                index_file=state_dir / SEARCH_DB_FILE,
                vector_store=vector_store,
                sources_dir=sources_dir,
            )
            results = retriever.search(query, top_k=top_k, mode=search_mode if search_mode != "hybrid" else None)
            raw = False
        else:
            # Use EnhancedRetriever when RAG features are enabled
            enhanced = _build_enhanced_retriever(
                wiki, vector_store, sources_dir, state_dir, cfg, llm_client
            )

            # Check semantic cache first
            if not raw:
                cached = enhanced.check_cache(query)
                if cached is not None:
                    if format == "json":
                        typer.echo(format_json_output(cached, []))
                    else:
                        typer.echo(cached.answer)
                        typer.echo("")
                        typer.echo("Sources:")
                        for i, c in enumerate(cached.citations, 1):
                            source_str: str = c.get("source") or "wiki"
                            typer.echo(f"  [{i}] {c['article']} ({source_str})")
                    if remember:
                        _save_query_memory(state_dir, query, cached.answer, cfg.get(COMPILE_EMBEDDING_MODEL), memory_by=resolve_actor(cfg))
                    return

            # Check if any enhanced features are enabled
            has_enhanced = (
                cfg.get(SEARCH_EXPANSION_ENABLED, False)
                or cfg.get(SEARCH_CRAG_ENABLED, False)
                or cfg.get(SEARCH_RERANKER_ENABLED, False)
                or cfg.get(SEARCH_HYDE_ENABLED, False)
            )

            if has_enhanced:
                expand_graph = search_mode == "graph"
                results = enhanced.search(query, sources_dir, top_k=top_k, expand_graph=expand_graph)
                raw = False
            else:
                retriever = HybridRetriever(
                    wiki,
                    index_file=state_dir / SEARCH_DB_FILE,
                    vector_store=vector_store,
                    sources_dir=sources_dir,
                )
                results, raw = _execute_query_search(retriever, query, search_mode, sources_dir, top_k)
    finally:
        if vector_store is not None:
            vector_store.close()

    fire_hook(HookEvent.ON_QUERY, query=query, results=results, search_mode=search_mode)

    if not results:
        typer.echo("No relevant content found.")
        raise typer.Exit(0)

    if raw:
        _print_raw_results(results, format)
        return

    if llm_client is None:
        typer.echo("Warning: No LLM configured. Showing raw context instead.", err=True)
        _print_raw_results(results, "text")
        return

    try:
        if has_enhanced:
            synthesis = enhanced.synthesize(query, results)
        else:
            synthesizer = TokenOverlapRetriever(wiki)
            synthesis = synthesizer.synthesize(query, results, llm_client=llm_client)
    except Exception as exc:
        typer.echo(f"LLM synthesis failed: {exc}", err=True)
        typer.echo("Showing raw context instead:", err=True)
        _print_raw_results(results, "text")
        raise typer.Exit(1)

    if format == "json":
        typer.echo(format_json_output(synthesis, results))
    else:
        typer.echo(synthesis.answer)
        typer.echo("")
        typer.echo("Sources:")
        c: dict[str, Any]
        for i, c in enumerate(synthesis.citations, 1):
            source_str: str = c.get("source") or "wiki"
            typer.echo(f"  [{i}] {c['article']} ({source_str})")

    if remember:
        _save_query_memory(state_dir, query, synthesis.answer, cfg.get(COMPILE_EMBEDDING_MODEL), memory_by=resolve_actor(cfg))
