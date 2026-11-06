"""How the query command turns hits into an answer.

Running the agentic pipeline, synthesizing results, and recording query
memory.
"""

from __future__ import annotations

from pathlib import Path

from kompend.cli.query.render import (
    encode_structured,
    figure_line,
    is_captured_source,
    node_type_index,
    print_raw_results,
    structured_answer,
    wiki_link,
)
from kompend.core.config import ResolvedConfig, RlmConfig
from kompend.core.config._configs import DisplayConfig, LlmDataFormat
from kompend.core.models import CompiledWiki
from kompend.core.paths import DEFAULT_OUTPUT_DIR_NAME
from kompend.core.trust.actor import resolve_actor
from kompend.llm.context_window import ContextWindow
from kompend.logging import get_console, get_err_console, get_logger
from kompend.memory import MemoryStore
from kompend.rag.sqlite_vec_store import SqliteVecWikiStore
from kompend.search.pipeline import EnhancedRetriever
from kompend.search.retriever import (
    HybridRetriever,
    KeywordRetriever,
    SearchResult,
    SynthesisResult,
    run_agentic_query,
    synthesize_results,
)

logger = get_logger(__name__)


def run_agentic_query_output(
    query: str,
    wiki: CompiledWiki,
    vector_store: SqliteVecWikiStore | None,
    format: LlmDataFormat,
    cfg: ResolvedConfig,
    cw: ContextWindow | None,
    remember: bool,
    state_dir: Path,
    tag_groups: list[list[str]] | None = None,
    attr_filters: list[tuple[str, str | None]] | None = None,
) -> None:
    """Run the agentic query agent and print its answer.

    Does NOT close *vector_store* — the caller owns its lifetime (the store
    must stay open while the agent's tools search).
    """
    model = cfg.kompend.search.agentic.model or cfg.kompend.compile.llm.default_model
    if not model:
        get_err_console().print("error: No LLM model configured for agentic query.")
        raise SystemExit(1)
    retriever = (
        HybridRetriever(wiki, vector_store=vector_store, display=cfg.kompend.display)
        if vector_store is not None
        else KeywordRetriever(wiki, display=cfg.kompend.display)
    )
    try:
        answer = run_agentic_query(
            query,
            retriever=retriever,
            model=model,
            max_sub_questions=cfg.kompend.search.agentic.max_sub_questions,
            context_window=cw.max_input if cw is not None else None,
            tag_groups=tag_groups or [],
            attr_filters=attr_filters or [],
            llm=cfg.kompend.compile.llm,
        )
    except Exception as exc:
        get_err_console().print(f"Agentic query failed: {exc}")
        raise SystemExit(1)
    synthesis = SynthesisResult(answer=answer, citations=[])
    if format == LlmDataFormat.MARKDOWN:
        get_console().print(synthesis.answer)
    else:
        get_console().print(encode_structured(structured_answer(synthesis, wiki), format))
    if remember:
        save_query_memory(
            state_dir,
            query,
            synthesis.answer,
            cfg.kompend.compile.embedding.model,
            memory_by=resolve_actor(cfg),
            display=cfg.kompend.display,
        )


def synthesize_and_output(  # noqa: PLR0917
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
        raise SystemExit(0)

    verbosity: str = cfg.kompend.search.synthesis.verbosity.value

    try:
        reasoning_effort: str | None = cfg.kompend.compile.llm.reasoning_effort
        rlm_cfg = RlmConfig.model_validate(cfg.kompend.compile.llm.rlm or {})
        if has_enhanced and enhanced is not None:
            synthesis = enhanced.synthesize(
                query,
                results,
                verbosity=verbosity,
                reasoning_effort=reasoning_effort,
                rlm=rlm_cfg,
            )
        else:
            synthesis = synthesize_results(
                query,
                results,
                model=model,
                verbosity=verbosity,
                rlm=rlm_cfg,
                llm=cfg.kompend.compile.llm,
            )
    except Exception as exc:
        get_err_console().print(f"LLM synthesis failed: {exc}")
        get_err_console().print("Showing raw context instead:")
        print_raw_results(results, LlmDataFormat.MARKDOWN)
        raise SystemExit(1)

    if format == LlmDataFormat.MARKDOWN:
        get_console().print(synthesis.answer)
        get_console().print("")
        get_console().print("Sources:")
        node_types = node_type_index(wiki)
        for i, c in enumerate(synthesis.citations, 1):
            link = wiki_link(c.get("article_id"), node_types)
            ref = link or f"{DEFAULT_OUTPUT_DIR_NAME}/{c['article']}"
            get_console().print(f"  [{i}] {c['article']} — {ref}", markup=False)
            for figure in c.get("figures", []):
                get_console().print(f"        Figure: {figure_line(figure)}", markup=False)
    else:
        get_console().print(encode_structured(structured_answer(synthesis, wiki), format))

    if remember:
        save_query_memory(
            state_dir,
            query,
            synthesis.answer,
            cfg.kompend.compile.embedding.model,
            memory_by=resolve_actor(cfg),
            display=cfg.kompend.display,
        )


def save_query_memory(
    state_dir: Path,
    query: str,
    answer: str,
    embedding_model: str | None,
    memory_by: str = "",
    display: DisplayConfig | None = None,
) -> None:
    """Persist a query/answer pair to the memory store."""
    try:
        mem_store = MemoryStore(base_dir=state_dir, embedding_model=embedding_model, display=display)
        mem_store.add_entry(query=query, answer=answer, memory_by=memory_by)
        mem_store.close()
        get_err_console().print("\n✓ Query and answer saved to memory.")
    except Exception as exc:
        get_err_console().print(f"\nWarning: Failed to save to memory: {exc}")
