"""Search infrastructure and output formatting for the query command.

Extracted from :mod:`kompend.cli.cmds.knowledge.query` to reduce module size.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kompend.agents.search import CragGradeAgent, HydeAgent, QueryExpansionAgent
from kompend.cli.args.query import SearchMode
from kompend.core.config import ResolvedConfig, RlmConfig
from kompend.core.config._configs import LlmDataFormat, Verbosity
from kompend.core.models import CompiledWiki, MetadataField
from kompend.core.paths import DEFAULT_OUTPUT_DIR_NAME, PROTECTED_SOURCE_SUBDIRS, VECTORS_DIR
from kompend.core.trust.actor import resolve_actor
from kompend.core.verbosity import is_verbose
from kompend.llm.context_window import ContextWindow
from kompend.llm.format import encode_for_llm
from kompend.logging import get_console, get_err_console, get_logger
from kompend.memory import MemoryStore
from kompend.output.vault import VaultWriter
from kompend.rag import build_vector_index
from kompend.rag.zvec_store import ZvecWikiStore
from kompend.search.adaptive_retriever import AdaptiveRetriever
from kompend.search.filters import matches_attr_filter, matches_tag_filter
from kompend.search.pipeline import EnhancedRetriever
from kompend.search.retriever import (
    HybridRetriever,
    KeywordRetriever,
    SearchResult,
    SynthesisResult,
    run_agentic_query,
    synthesize_results,
)
from kompend.types import JSONDict, JSONList

#: On-disk folder per node kind — mirrors ``VaultWriter._KIND_FOLDERS`` plus
#: the overview kind, which stays at the bundle root.
_KIND_FOLDERS: dict[str, str] = {**VaultWriter._KIND_FOLDERS, "overview": ""}

_SYNTHESIS_TOKEN_KEYS: dict[str, str] = {
    Verbosity.BRIEF.value: "brief",
    Verbosity.STANDARD.value: "standard",
    Verbosity.COMPREHENSIVE.value: "comprehensive",
}

logger = get_logger(__name__)


def open_vector_store(
    kompend_dir: Path,
    wiki: CompiledWiki,
    *,
    embed_model: str | None = None,
    embed_attachments: bool = False,
) -> ZvecWikiStore | None:
    """Open the zvec sidecar index built during compile, if it exists.

    The embedding model comes from wiki metadata when available (cache
    path) or falls back to *embed_model* (config value) — a files-loaded
    wiki carries no embedding metadata of its own.

    *embed_attachments* must be the value the index was built with: it is part
    of which nodes the index covers, so a mismatch would rebuild on every query.
    """
    db_file = kompend_dir / VECTORS_DIR
    if not db_file.exists():
        return None

    model = wiki.metadata.get(str(MetadataField.EMBEDDING_MODEL)) or embed_model
    if not model:
        logger.debug("No embedding model available; skipping vector store")
        return None

    # Sidecar index can go stale when a wiki is recompiled (new/changed pages)
    # without a matching re-index. Detect via the build-time id manifest and
    # rebuild transparently so queries never silently fall back to keyword.
    if ZvecWikiStore(db_file=db_file).is_stale(wiki, embed_attachments=embed_attachments):
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
        return ZvecWikiStore.open(db_file, embedding_model=model)
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
            raise typer.Exit(1)
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
    vector_store: ZvecWikiStore | None,
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
    vector_store: ZvecWikiStore | None,
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


def should_use_agentic(cfg: ResolvedConfig, *, raw: bool, model: str | None) -> bool:
    """Whether this query should take the opt-in agentic path.

    Disabled by default. The agentic tools honour tag/attribute filters
    (pushed down through the retriever), so filters no longer disqualify the
    path.
    """
    agentic = cfg.kompend.search.agentic
    return bool(agentic.enabled and not raw and model)


def run_agentic_query_output(
    query: str,
    wiki: CompiledWiki,
    vector_store: ZvecWikiStore | None,
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
        raise typer.Exit(1)
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
            budget=cfg.kompend.compile.budget,
        )
    except Exception as exc:
        get_err_console().print(f"Agentic query failed: {exc}")
        raise typer.Exit(1)
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


def build_enhanced_retriever(
    wiki: CompiledWiki,
    vector_store: ZvecWikiStore | None,
    state_dir: Path,
    cfg: ResolvedConfig,
    model: str,
) -> EnhancedRetriever:
    """Build an EnhancedRetriever from resolved config."""
    _llm = cfg.kompend.compile.llm
    _budget = cfg.kompend.compile.budget
    _retries = _llm.output_retries if _llm.output_retries_enabled else 0
    expansion_agent = QueryExpansionAgent(model, timeout=_llm.timeout, retries=_retries, llm=_llm, budget=_budget) if model else None
    crag_agent = CragGradeAgent(model, timeout=_llm.timeout, retries=_retries, llm=_llm, budget=_budget) if model else None
    hyde_agent = HydeAgent(model, timeout=_llm.timeout, retries=_retries, llm=_llm, budget=_budget) if model else None
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
        reranker_model=cfg.kompend.search.reranker.model,
        reranker_top_k=cfg.kompend.search.reranker.top_k,
        hyde_enabled=cfg.kompend.search.hyde.enabled,
        llm=_llm,
        budget=_budget,
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
        raise typer.Exit(0)

    verbosity: str = cfg.kompend.search.synthesis.verbosity.value
    token_attr = _SYNTHESIS_TOKEN_KEYS.get(verbosity, "standard")
    target_tokens: int = getattr(cfg.kompend.search.synthesis.tokens, token_attr)
    max_tokens: int = min(target_tokens, cw.max_output) if cw is not None else target_tokens

    try:
        reasoning_effort: str | None = cfg.kompend.compile.llm.reasoning_effort
        rlm_cfg = RlmConfig.model_validate(cfg.kompend.compile.llm.rlm or {})
        if has_enhanced and enhanced is not None:
            synthesis = enhanced.synthesize(
                query,
                results,
                verbosity=verbosity,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                rlm=rlm_cfg,
            )
        else:
            synthesis = synthesize_results(
                query,
                results,
                model=model,
                verbosity=verbosity,
                max_tokens=max_tokens,
                rlm=rlm_cfg,
                llm=cfg.kompend.compile.llm,
                budget=cfg.kompend.compile.budget,
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
        save_query_memory(
            state_dir,
            query,
            synthesis.answer,
            cfg.kompend.compile.embedding.model,
            memory_by=resolve_actor(cfg),
            display=cfg.kompend.display,
        )


def is_captured_source(source_file: str | None) -> bool:
    """Return True if *source_file* is a tool-managed note in the capture lifecycle.

    Covers ``_captured/``, ``_learned/``, ``_distilled/`` and ``_archive/``
    notes — raw and archived notes are hidden from query retrieval either
    because they feed compile directly (raw) or were retired by
    ``consolidate``/``distill`` (archived).
    """
    return bool(source_file) and any(sub in source_file.replace("\\", "/") for sub in PROTECTED_SOURCE_SUBDIRS)


def structured_answer(synthesis: SynthesisResult, wiki: CompiledWiki | None = None) -> JSONDict:
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
    """Build a relative path into the compiled wiki.

    The on-disk folder mirrors the writer's kind mapping (``concept``/\u2026
    -> ``concepts/``; ``overview`` -> bundle root), not the bare node type.
    """
    if not article_id or article_id not in node_types:
        return None
    folder = _KIND_FOLDERS.get(node_types[article_id], "concepts")
    if not folder:
        return f"{DEFAULT_OUTPUT_DIR_NAME}/{article_id}.md"
    return f"{DEFAULT_OUTPUT_DIR_NAME}/{folder}/{article_id}.md"


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
    payload: JSONList = []
    for r in results:
        row: JSONDict = {
            "article": r.title,
            "source": r.source_file,
            "snippet": r.snippet,
            "score": round(r.score, 3),
        }
        payload.append(row)
    return encode_for_llm(payload, fmt)


def encode_structured(payload: JSONDict, fmt: LlmDataFormat) -> str:
    """Encode a structured payload as json/toon/ison for agent consumers."""
    if fmt == LlmDataFormat.JSON:
        return json.dumps(payload, indent=2)
    if fmt == LlmDataFormat.ISON:
        payload = {"answer": [{"text": payload["answer"]}], "citations": payload["citations"]}
    return encode_for_llm(payload, fmt)


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
