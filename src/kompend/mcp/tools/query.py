"""Query and status MCP tools."""

import dataclasses
from typing import Any

from kompend.cli._query_helpers import matches_attr_filter
from kompend.cli.args.query import SearchMode
from kompend.cli.cmds.knowledge.query import _parse_attr_filters
from kompend.cli.cmds.pipeline.status import collect_stats
from kompend.core.config import RlmConfig, resolve_config
from kompend.core.models import load_wiki_from_files
from kompend.core.paths import VECTORS_DIR, vault_paths
from kompend.logging import get_logger
from kompend.rag.zvec_store import ZvecWikiStore
from kompend.search.retriever import HybridRetriever, KeywordRetriever, run_agentic_query

logger = get_logger(__name__)


def query_wiki(
    query: str,
    wiki_dir: str | None = None,
    top_k: int = 5,
    raw: bool = False,
    search_mode: SearchMode = SearchMode.HYBRID,
    attributes: list[str] | None = None,
) -> dict[str, Any]:
    """Ask a natural language question against the compiled wiki.

    Markdown-first: the wiki is built from the generated ``.md`` files.
    Vector embeddings (``.kompend/vectors/``) upgrade retrieval to semantic;
    without them, keyword-only results feed LLM synthesis but cannot produce
    a useful answer on their own — a model is required.

    *attributes*: optional ``KEY=VALUE`` or bare ``KEY`` filters (AND'd).
    """
    vp = vault_paths(wiki_dir)
    cfg = resolve_config(base_dir=vp.base_dir)
    sources_dir = vp.sources_dir
    output_dir = vp.output_dir
    state_dir = vp.kompend_dir

    wiki = load_wiki_from_files(output_dir)
    if wiki is None:
        return {"status": "error", "message": f"No wiki pages found in {output_dir}. Run 'wiki_compile' first."}

    vector_store: ZvecWikiStore | None = None
    if state_dir and (state_dir / VECTORS_DIR).exists():
        try:
            embed_model = cfg.kompend.compile.embedding.model
            vector_store = ZvecWikiStore.open(state_dir / VECTORS_DIR, embedding_model=embed_model or "hash")
        except Exception:
            vector_store = None

    try:
        retriever = (
            HybridRetriever(wiki, vector_store=vector_store, display=cfg.kompend.display)
            if vector_store is not None
            else KeywordRetriever(wiki, display=cfg.kompend.display)
        )
        model = cfg.kompend.compile.llm.default_model or ""
        agentic = cfg.kompend.search.agentic
        if agentic.enabled and not raw and model and search_mode != SearchMode.RAW:
            answer = run_agentic_query(
                query,
                retriever=retriever,
                model=agentic.model or model,
                max_sub_questions=agentic.max_sub_questions,
                rlm=RlmConfig.model_validate(cfg.kompend.compile.llm.rlm or {}),
                attr_filters=_parse_attr_filters(attributes) if attributes else [],
                llm=cfg.kompend.compile.llm,
            )
            return {"status": "ok", "results": [], "raw": False, "synthesis": {"answer": answer, "citations": []}}
        expand_graph = search_mode == SearchMode.GRAPH
        mode = SearchMode.HYBRID if search_mode == SearchMode.GRAPH else search_mode

        if mode == SearchMode.RAW:
            if not sources_dir.exists():
                return {"status": "error", "message": "No sources directory found for raw search."}
            results = retriever.search(query, top_k=top_k)
            raw = True
        elif mode == SearchMode.HYBRID:
            results = retriever.search_hybrid(query, sources_dir, top_k=top_k, expand_graph=expand_graph)
        else:
            results = retriever.search_wiki(query, top_k=top_k)
    finally:
        if vector_store is not None:
            vector_store.close()

    if attributes:
        attr_filters = _parse_attr_filters(attributes)
        results = [r for r in results if matches_attr_filter(r.attributes, attr_filters)]

    result_dicts = [dataclasses.asdict(r) for r in results] if results else []
    if not results:
        return {"status": "ok", "results": result_dicts, "synthesis": None, "raw": True}

    try:
        model = cfg.kompend.compile.llm.default_model or ""
    except Exception:
        model = ""

    # No model to synthesize an answer — only vector-store results are meaningful
    # enough to surface raw. Keyword-only results without synthesis don't produce
    # a useful answer.
    if not model and not raw:
        return {
            "status": "error",
            "message": "No LLM model configured (compile.llm.default_model)."
            " Query requires a model to synthesize an answer."
            " Use --raw for retrieval-only output, or configure an embedding model"
            " (compile.embedding.model) for raw semantic search.",
        }

    if model and not raw:
        try:
            synthesizer = HybridRetriever(wiki, display=cfg.kompend.display)
            synthesis = synthesizer.synthesize(
                query,
                results,
                model=model,
                rlm=RlmConfig.model_validate(cfg.kompend.compile.llm.rlm or {}),
            )
            return {
                "status": "ok",
                "results": result_dicts,
                "raw": False,
                "synthesis": {
                    "answer": synthesis.answer,
                    "citations": [{"article": c.get("article", ""), "source": c.get("source", "wiki")} for c in synthesis.citations],
                },
            }
        except Exception:
            logger.warning("LLM synthesis failed for query", exc_info=True)

    return {"status": "ok", "results": result_dicts, "synthesis": None, "raw": True}


def wiki_status(
    wiki_dir: str | None = None,
) -> dict[str, Any]:
    """Show per-wiki statistics."""
    try:
        vp = vault_paths(wiki_dir)
        resolve_config(base_dir=vp.base_dir)
        sources_dir = vp.sources_dir
        output_dir = vp.output_dir
        stats = collect_stats(
            sources_dir=sources_dir if sources_dir.exists() else None,
            output_dir=output_dir if output_dir.exists() else None,
            config_file=vp.config_file if vp.config_file.exists() else None,
        )
        return {"status": "ok", "stats": stats}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
