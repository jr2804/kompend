"""How the query command turns hits into an answer.

Synthesizing results through the unified query session, and recording
query memory.
"""

from __future__ import annotations

from pathlib import Path

from kompend.agents.types import QuerySearchProtocol
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
from kompend.search.pipeline import EnhancedRetriever
from kompend.search.retriever import (
    SearchResult,
    synthesize_results,
)

logger = get_logger(__name__)


def synthesize_and_output(  # noqa: PLR0917
    query: str,
    results: list[SearchResult],
    format: LlmDataFormat,
    enhanced: EnhancedRetriever | None,
    retriever: QuerySearchProtocol | None,
    wiki: CompiledWiki,
    model: str,
    cw: ContextWindow | None,
    remember: bool,
    state_dir: Path,
    cfg: ResolvedConfig,
) -> None:
    """Synthesize results and output the answer.

    *retriever* is the raw search surface the query session's tools use
    for follow-up investigation; the caller keeps its backing store open
    for the duration of the call.
    """
    results = [r for r in results if not is_captured_source(r.source_file)]
    if not results:
        get_console().print("No relevant content found.")
        raise SystemExit(0)

    verbosity: str = cfg.kompend.search.synthesis.verbosity.value

    try:
        reasoning_effort: str | None = cfg.kompend.compile.llm.reasoning_effort
        rlm_cfg = RlmConfig.model_validate(cfg.kompend.compile.llm.rlm or {})
        cw_max = cw.max_input if cw is not None else None
        if enhanced is not None:
            synthesis = enhanced.synthesize(
                query,
                results,
                verbosity=verbosity,
                reasoning_effort=reasoning_effort,
                rlm=rlm_cfg,
                retriever=retriever,
                context_window=cw_max,
            )
        else:
            synthesis = synthesize_results(
                query,
                results,
                model=model,
                verbosity=verbosity,
                rlm=rlm_cfg,
                retriever=retriever,
                context_window=cw_max,
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
