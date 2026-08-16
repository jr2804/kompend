"""Query and status MCP tools."""

import dataclasses
from typing import Any

from knox.cli.args.query import SearchMode
from knox.cli.cmds.pipeline.status import collect_stats
from knox.core.config import RlmConfig, resolve_config
from knox.core.config.keys import (
    COMPILE_EMBEDDING_MODEL,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_RLM,
)
from knox.core.models import load_cached_wiki
from knox.core.paths import SEARCH_DB_FILE, VECTORS_DIR, vault_paths
from knox.logging import get_logger
from knox.rag.zvec_store import ZvecWikiStore
from knox.search.retriever import HybridRetriever

logger = get_logger(__name__)


def query_wiki(
    query: str,
    wiki_dir: str | None = None,
    top_k: int = 5,
    raw: bool = False,
    search_mode: SearchMode = SearchMode.HYBRID,
) -> dict[str, Any]:
    """Ask a natural language question against the compiled wiki."""
    vp = vault_paths(wiki_dir)
    cfg = resolve_config(base_dir=vp.base_dir)
    sources_dir = vp.sources_dir
    output_dir = vp.output_dir
    state_dir = vp.knox_dir
    base_dir = vp.base_dir

    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        return {"status": "error", "message": f"No compiled wiki found in {output_dir}. Run 'wiki_compile' first."}

    vector_store: ZvecWikiStore | None = None
    if state_dir and (state_dir / VECTORS_DIR).exists():
        try:
            embed_model = cfg.get(COMPILE_EMBEDDING_MODEL)
            vector_store = ZvecWikiStore.open(state_dir / VECTORS_DIR, embedding_model=embed_model or "hash")
        except Exception:
            vector_store = None

    try:
        retriever = HybridRetriever(
            wiki,
            index_file=state_dir / SEARCH_DB_FILE,  # ty:ignore[unknown-argument]
            vector_store=vector_store,
            sources_dir=sources_dir if sources_dir.exists() else None,  # ty:ignore[unknown-argument]
        )
        expand_graph = search_mode == SearchMode.GRAPH
        mode = SearchMode.HYBRID if search_mode == SearchMode.GRAPH else search_mode

        if mode == SearchMode.RAW:
            if not sources_dir.exists():
                return {"status": "error", "message": "No sources directory found for raw search."}
            results = retriever.search_raw(query, sources_dir, top_k=top_k)  # ty:ignore[unresolved-attribute]
            return {"status": "ok", "results": results, "synthesis": None, "raw": True}
        elif mode == SearchMode.HYBRID:
            results = retriever.search_hybrid(query, sources_dir, top_k=top_k, expand_graph=expand_graph)
        else:
            results = retriever.search_wiki(query, top_k=top_k)
    finally:
        if vector_store is not None:
            vector_store.close()

    if not results:
        result_dicts = [dataclasses.asdict(r) for r in results] if results else []
        return {"status": "ok", "results": result_dicts, "synthesis": None, "raw": True}

    try:
        model = cfg.get(COMPILE_LLM_DEFAULT_MODEL, "")
    except Exception:
        model = ""

    if model:
        try:
            synthesizer = HybridRetriever(wiki)
            synthesis = synthesizer.synthesize(
                query,
                results,
                model=model,
                rlm=RlmConfig.model_validate(cfg.get(COMPILE_LLM_RLM) or {}),
            )
            result_dicts = [dataclasses.asdict(r) for r in results]
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

    result_dicts = [dataclasses.asdict(r) for r in results]
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
        )
        return {"status": "ok", "stats": stats}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
