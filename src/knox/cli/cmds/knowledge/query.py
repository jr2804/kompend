"""Query command.

Search infrastructure and output formatting helpers live in
:mod:`knox.cli._query_helpers`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from knox.cli._query_helpers import (  # noqa: F401  # re-exported for backward compat
    build_enhanced_retriever,
    encode_raw_results,
    encode_structured,
    execute_query_search,
    handle_adaptive_search,
    handle_enhanced_search,
    is_captured_source,
    matches_tag_filter,
    node_type_index,
    open_vector_store,
    print_raw_results,
    save_query_memory,
    structured_answer,
    synthesize_and_output,
    wiki_link,
)
from knox.cli._shared import build_cli_overrides, vault_paths_from_ctx
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
from knox.core.config._configs import LlmDataFormat
from knox.core.config.keys import (
    COMPILE_LLM_CONTEXT_WINDOW_CAP,
    QUERY_FORMAT,
    EnvVar,
)
from knox.core.models import load_cached_wiki, load_wiki_from_files
from knox.core.paths import vault_paths
from knox.llm.context_window import discover_context_window
from knox.logging import get_console, get_err_console, get_logger
from knox.search.classifier import classify
from knox.search.retriever import HybridRetriever, KeywordRetriever

logger = get_logger(__name__)


@app.command(name="query")
def query_cmd(
    ctx: typer.Context,
    query: QueryText,
    wiki_dir: str | None = typer.Option(
        None,
        "--wiki-dir",
        "-w",
        help="Project root directory. Default: current working directory or KNOX_WIKI_DIR.",
        envvar=EnvVar.WIKI_DIR,
    ),
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
    vp = vault_paths_from_ctx(ctx)
    # If explicit --wiki-dir was passed to the command (not via global callback),
    # re-resolve vp from it.
    if wiki_dir:
        vp = vault_paths(wiki_dir)
    no_user_tracking = ctx.obj.get("no_user_tracking") if ctx.obj else None
    cli_overrides = build_cli_overrides(
        no_user_tracking=no_user_tracking,
        synthesis_verbosity=verbosity.value if verbosity is not None else None,
    )
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=cli_overrides, config_file=config)
    if format is None:
        format = LlmDataFormat(cfg.get(QUERY_FORMAT, LlmDataFormat.MARKDOWN.value))
    sources_dir = vp.sources_dir
    output_dir = vp.output_dir
    state_dir = vp.knox_dir

    if _classify_intent(query, no_classify, format):
        return

    tag_groups = _parse_tag_groups(tags)

    wiki = load_cached_wiki(vp.base_dir)
    portable = wiki is None
    if portable:
        # No .knox cache — build the wiki from the generated markdown files
        # (portable mode: the wiki folder alone is the source of truth).
        wiki = load_wiki_from_files(vp.output_dir)
    if wiki is None:
        get_err_console().print(
            f"No compiled wiki found in {output_dir}. Run 'knox compile' first.",
        )
        raise typer.Exit(1)

    vector_store = open_vector_store(state_dir, wiki) if not portable else None
    if portable:
        get_console().print("Portable mode: no .knox cache — searching the wiki files directly.")
    model = cfg.get("compile.llm.default_model", "")
    cw = discover_context_window(model, context_window_cap=cfg.get(COMPILE_LLM_CONTEXT_WINDOW_CAP)) if model else None

    try:
        results, raw, has_enhanced = _execute_search(
            wiki=wiki,
            query=query,
            top_k=top_k,
            search_mode=search_mode,
            sources_dir=sources_dir,
            state_dir=state_dir,
            vector_store=vector_store,
            tag_groups=tag_groups,
            adaptive=adaptive,
            cfg=cfg,
            model=model,
        )
    finally:
        if vector_store is not None:
            vector_store.close()

    if not results:
        get_console().print("No relevant content found.")
        raise typer.Exit(0)

    if raw:
        print_raw_results(results, format)
        return

    if not model:
        get_err_console().print("Warning: No LLM configured. Showing raw context instead.")
        print_raw_results(results, LlmDataFormat.MARKDOWN)
        return

    enhanced = build_enhanced_retriever(wiki, vector_store, state_dir, cfg, model) if has_enhanced else None
    synthesize_and_output(query, results, format, has_enhanced, enhanced, wiki, model, cw, remember, state_dir, cfg)


def _execute_search(
    *,
    wiki: Any,
    query: str,
    top_k: int,
    search_mode: str,
    sources_dir: Path | None,
    state_dir: Path,
    vector_store: Any,
    tag_groups: list[list[str]] | None,
    adaptive: bool,
    cfg: Any,
    model: str,
) -> tuple[list[Any], bool, bool]:
    """Dispatch query search; returns (results, raw, has_enhanced).

    Without a vector store (portable mode — no ``.knox/`` state), falls back
    to keyword search over the wiki pages.
    """
    if vector_store is None:
        retriever: HybridRetriever | KeywordRetriever = KeywordRetriever(wiki)
        results, raw = execute_query_search(retriever, query, search_mode, sources_dir, top_k, tag_groups)
        return results, raw, False
    if adaptive:
        results, raw = handle_adaptive_search(query, top_k, search_mode, vector_store, sources_dir, wiki, state_dir, tag_groups)
        return results, raw, False
    enhanced_result = handle_enhanced_search(query, top_k, search_mode, sources_dir, vector_store, state_dir, wiki, cfg, model, tag_groups)
    if enhanced_result is not None:
        return enhanced_result, False, True
    retriever = HybridRetriever(wiki, vector_store=vector_store)
    results, raw = execute_query_search(retriever, query, search_mode, sources_dir, top_k, tag_groups)
    return results, raw, False


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


def _classify_intent(query: str, no_classify: bool, format: LlmDataFormat) -> bool:
    """Run intent classification and respond directly if classified.

    Returns True if the classifier produced a response (caller should return).
    """
    if no_classify:
        return False

    result = classify(query)
    if result.response is None:
        return False
    if format == LlmDataFormat.MARKDOWN:
        get_console().print(result.response)
    else:
        get_console().print(encode_structured({"intent": result.intent.value, "response": result.response}, format))
    return True
