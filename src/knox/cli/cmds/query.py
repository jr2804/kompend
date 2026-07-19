"""Query command.

Search infrastructure and output formatting helpers live in
:mod:`knox.cli._query_helpers`.
"""

from __future__ import annotations

import typer

from knox.cli._query_helpers import (  # noqa: F401  # re-exported for backward compat
    _build_enhanced_retriever,
    _encode_raw_results,
    _encode_structured,
    _execute_query_search,
    _handle_adaptive_search,
    _handle_enhanced_search,
    _is_captured_source,
    _matches_tag_filter,
    _node_type_index,
    _open_vector_store,
    _print_raw_results,
    _save_query_memory,
    _structured_answer,
    _synthesize_and_output,
    _wiki_link,
)
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
    SourceTags,
)
from knox.cli.args.query import SearchMode
from knox.core.config import resolve_config
from knox.core.config._configs import QueryFormat
from knox.core.config.keys import (
    COMPILE_LLM_CONTEXT_WINDOW_CAP,
    QUERY_FORMAT,
)
from knox.core.models import load_cached_wiki
from knox.llm.context_window import discover_context_window
from knox.logging import get_console, get_err_console, get_logger
from knox.search.classifier import classify
from knox.search.retriever import HybridRetriever

logger = get_logger(__name__)


def _parse_tag_groups(tags: list[str] | None) -> list[list[str]]:
    """Parse --tags values into AND-groups for OR-filtering.

    Each --tags VALUE is split by comma (AND within a value).
    Multiple --tags values are OR'd together.
    Returns groups like [["a","b"],["c"]] meaning "(a AND b) OR c".
    """
    if not tags:
        return []
    groups: list[list[str]] = []
    for value in tags:
        group = [t.strip() for t in value.split(",") if t.strip()]
        if group:
            groups.append(group)
    return groups


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
    tags: SourceTags = None,
) -> None:
    """Ask a natural language question against the compiled wiki."""
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

    tag_groups = _parse_tag_groups(tags)

    wiki = load_cached_wiki(state_dir.parent)
    if wiki is None:
        get_err_console().print(
            f"No compiled wiki found in {output_dir}. Run 'knox compile' first.",
        )
        raise typer.Exit(1)

    vector_store = _open_vector_store(state_dir, wiki)
    model = cfg.get("compile.llm.default_model", "")
    cw = discover_context_window(model, context_window_cap=cfg.get(COMPILE_LLM_CONTEXT_WINDOW_CAP)) if model else None

    try:
        if adaptive:
            results, raw = _handle_adaptive_search(
                query, top_k, search_mode, vector_store, sources_dir, wiki, state_dir, tag_groups
            )
            has_enhanced = False
        else:
            enhanced_result = _handle_enhanced_search(
                query, top_k, search_mode, sources_dir, vector_store, state_dir, wiki, cfg, model, tag_groups
            )
            if enhanced_result is not None:
                results = enhanced_result
                raw = False
                has_enhanced = True
            else:
                retriever = HybridRetriever(wiki, vector_store=vector_store)
                results, raw = _execute_query_search(
                    retriever, query, search_mode, sources_dir, top_k, tag_groups
                )
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

    if not model:
        get_err_console().print("Warning: No LLM configured. Showing raw context instead.")
        _print_raw_results(results, QueryFormat.TEXT)
        return

    enhanced = _build_enhanced_retriever(wiki, vector_store, state_dir, cfg, model) if has_enhanced else None
    _synthesize_and_output(query, results, format, has_enhanced, enhanced, wiki, model, cw, remember, state_dir, cfg)
