"""KNOX MCP server — thin wrapper that delegates to build_pipeline().

Uses the official ``mcp`` Python SDK (FastMCP) for stdio transport.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from knox.capture.capture import capture
from knox.capture.distill import distill
from knox.capture.learn import learn
from knox.cli._shared import resolve_project_paths
from knox.cli.args.query import SearchMode
from knox.cli.cmds.status import collect_stats
from knox.core.config import (
    DEFAULTS,
    EmbeddingConfig,
    LLMConfig,
    OutputConfig,
    coerce_config_value,
    get_global_config_path,
    get_project_config_path,
    resolve_config,
    set_config_file_value,
    validate_config_key,
)
from knox.core.config.keys import COMPILE_EMBEDDING_MODEL, DISPLAY_MCP_ELEMENT_CHARS, DISPLAY_MEMORY_ENTRIES, PATHS_WIKI_DIR
from knox.core.models import load_cached_wiki
from knox.core.paths import SEARCH_DB_FILE, VECTORS_DIR
from knox.core.pipeline.config import Compiler, ImageMode, OutputFormat, PipelineConfig
from knox.core.pipeline.orchestrator import build_pipeline
from knox.core.templates import get_template_loader
from knox.core.trust.actor import resolve_actor
from knox.llm.client import LLMClient
from knox.logging import get_logger
from knox.memory import MemoryStore
from knox.preprocessing.parsers.base import ParserRouter
from knox.preprocessing.parsers.docling_parser import DoclingParser
from knox.rag.zvec_store import ZvecWikiStore
from knox.scaffolding import ScaffoldError, scaffold_project
from knox.search.retriever import HybridRetriever, TokenOverlapRetriever

logger = get_logger(__name__)

server = FastMCP("KNOX")


def _make_memory_store(wiki_dir: str) -> tuple[MemoryStore, str]:
    """Create a MemoryStore and resolve the acting user from config."""
    cfg = resolve_config(cli_overrides={PATHS_WIKI_DIR: wiki_dir})
    embed_model = cfg.get(COMPILE_EMBEDDING_MODEL)
    return MemoryStore(base_dir=Path(wiki_dir), embedding_model=embed_model), resolve_actor(cfg)


@server.tool(name="parse_document")
def parse_document(file_path: str) -> dict[str, Any]:
    """Parse a document and return structured elements."""
    doc_file = Path(file_path)
    if not doc_file.exists():
        return {"status": "error", "message": f"File not found: {file_path}"}

    router = ParserRouter()
    try:
        router.register(DoclingParser())
    except Exception as exc:
        logger.warning("DoclingParser not available: %s", exc, exc_info=True)

    try:
        doc = router.parse(doc_file)
        return {
            "status": "ok",
            "title": doc.title or doc_file.name,
            "page_count": doc.page_count,
            "elements": [
                {
                    "type": str(el.type),
                    "content": str(el.content)[: DEFAULTS.get(DISPLAY_MCP_ELEMENT_CHARS, 500)],
                    "page": el.page,
                    "heading_level": el.heading_level,
                }
                for el in doc.elements
            ],
            "metadata": doc.metadata,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@server.tool(name="compile_wiki")
def compile_wiki(
    sources_dir: str,
    output: str = "./output",
    output_format: str = "markdown",
    image_mode: str = "file",
    urls: list[str] | None = None,
    compiler: str = "graph",
    vlm_model: str = "",
) -> dict[str, Any]:
    """Compile sources into a knowledge wiki.

    Parameters
    ----------
    compiler:
        Compiler backend: ``"graph"`` (default) or ``"wicer"``.
    """
    source_dir = Path(sources_dir)
    if not source_dir.exists():
        return {"status": "error", "message": f"Sources directory not found: {sources_dir}"}

    try:
        compiler_enum = Compiler(compiler)
    except ValueError:
        return {
            "status": "error",
            "message": f"Unknown compiler: {compiler!r}. Use graph or wicer.",
        }

    try:
        format_enum = OutputFormat(output_format)
    except ValueError:
        return {
            "status": "error",
            "message": f"Unknown format: {output_format}. Use markdown, json, or typst.",
        }

    try:
        image_mode_enum = ImageMode(image_mode)
    except ValueError:
        return {"status": "error", "message": f"Unknown image_mode: {image_mode!r}. Use file or base64."}

    config = PipelineConfig(
        source_dir=source_dir,
        output_dir=Path(output),
        llm=LLMConfig(default_model="", vlm_model=vlm_model),
        embedding=EmbeddingConfig(model=""),
        online_sources=urls or [],
        output=OutputConfig(format=format_enum),
        image_mode=image_mode_enum,
        compiler=compiler_enum,
    )

    pipeline = build_pipeline(config)

    try:
        wiki = pipeline.run()
        ext = "md" if output_format == "markdown" else output_format
        output_file = Path(output) / f"wiki.{ext}"
        return {
            "status": "ok",
            "node_count": len(wiki.nodes),
            "edge_count": len(wiki.edges),
            "toc_count": len(wiki.toc),
            "output_file": str(output_file.resolve()),
            "wiki": {
                "nodes": [dataclasses.asdict(n) for n in wiki.nodes],
                "edges": list(wiki.edges),
                "toc": list(wiki.toc),
                "quality_score": wiki.quality_score,
                "iteration": wiki.iteration,
                "metadata": dict(wiki.metadata),
            },
        }
    except Exception as exc:
        logger.exception("Compilation failed")
        return {"status": "error", "message": str(exc)}


@server.tool(name="wiki_capture")
def wiki_capture(
    text: str,
    sources_dir: str = "./sources",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Extract knowledge items from text and save as source files.

    Uses LLM to extract decisions, discoveries, corrections, and references
    from conversation text. Optional tags are attached to each item as a
    baseline (the LLM may add more). Results are written to sources/_captured/.
    """
    source_dir = Path(sources_dir)
    source_dir.mkdir(parents=True, exist_ok=True)

    try:
        cfg = resolve_config()
        llm_client = LLMClient.from_config(cfg)
    except Exception:
        llm_client = None

    try:
        paths = capture(text, sources_dir=source_dir, llm_client=llm_client, tags=tags)
        return {
            "status": "ok",
            "count": len(paths),
            "files": [str(p) for p in paths],
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@server.tool(name="wiki_learn")
def wiki_learn(
    text: str,
    sources_dir: str = "./sources",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Store a single fact directly as a source file.

    Unlike wiki_capture, this stores the text verbatim without LLM extraction.
    """
    source_dir = Path(sources_dir)
    source_dir.mkdir(parents=True, exist_ok=True)

    try:
        learned_file = learn(text, sources_dir=source_dir, tags=tags)
        return {
            "status": "ok",
            "file": str(learned_file),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@server.tool(name="memory_add")
def memory_add(
    query: str,
    answer: str,
    feedback: str | None = None,
    wiki_dir: str = ".",
) -> dict[str, Any]:
    """Add a query/answer pair to the memory store for long-term context."""
    try:
        store, actor = _make_memory_store(wiki_dir)
        store.add_entry(query=query, answer=answer, feedback=feedback, memory_by=actor)
        store.close()
        return {"status": "ok", "message": "Memory entry added successfully."}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@server.tool(name="memory_search")
def memory_search(
    query: str,
    top_k: int = 5,
    wiki_dir: str = ".",
) -> dict[str, Any]:
    """Search past memory entries semantically by query."""
    try:
        store, _actor = _make_memory_store(wiki_dir)
        results = store.search(query=query, top_k=top_k)
        store.close()
        return {"status": "ok", "results": results}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@server.tool(name="memory_synthesize")
def memory_synthesize(
    limit: int = 20,
    wiki_dir: str = ".",
) -> dict[str, Any]:
    """Synthesize recent memory entries into a system context file."""
    try:
        store, _actor = _make_memory_store(wiki_dir)
        entries = store.get_recent_entries(limit=limit)
        if not entries:
            store.close()
            return {"status": "ok", "message": "No memory entries found to synthesize."}

        base_dir = Path(wiki_dir)
        context_file = base_dir / "sources" / "_memory" / "system_context.md"
        context_file.parent.mkdir(parents=True, exist_ok=True)

        loader = get_template_loader()
        content = loader.render(
            "memory/system_context.md.j2",
            entries=entries[: DEFAULTS.get(DISPLAY_MEMORY_ENTRIES, 10)],
        )
        context_file.write_text(content, encoding="utf-8")

        store.close()
        return {"status": "ok", "message": f"Synthesized context written to {context_file}"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@server.tool(name="init_wiki")
def init_wiki(
    target: str = ".",
    bare: bool = False,
    force: bool = False,
    git: bool = False,
) -> dict[str, Any]:
    """Scaffold a new KNOX wiki project directory."""
    try:
        result = scaffold_project(Path(target), bare=bare, force=force, git=git)
        return {
            "status": "ok",
            "target_dir": str(result.target_dir),
            "created_dirs": [str(d) for d in result.created_dirs],
            "copied_files": [str(f) for f in result.copied_files],
            "skipped_files": list(result.skipped_files) if result.skipped_files else [],
            "git_initialised": result.git_initialised,
        }
    except ScaffoldError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@server.tool(name="query_wiki")
def query_wiki(
    query: str,
    wiki_dir: str = ".",
    top_k: int = 5,
    raw: bool = False,
    search_mode: SearchMode = SearchMode.HYBRID,
) -> dict[str, Any]:
    """Ask a natural language question against the compiled wiki."""
    cfg = resolve_config(cli_overrides={PATHS_WIKI_DIR: wiki_dir})
    sources_dir, output_dir, _cache_dir, state_dir = resolve_project_paths(cfg)

    wiki = load_cached_wiki(output_dir.parent)
    if wiki is None:
        return {"status": "error", "message": f"No compiled wiki found in {output_dir}. Run 'compile_wiki' first."}

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
            index_file=state_dir / SEARCH_DB_FILE,
            vector_store=vector_store,
            sources_dir=sources_dir if sources_dir.exists() else None,
        )
        expand_graph = search_mode == SearchMode.GRAPH
        mode = SearchMode.HYBRID if search_mode == SearchMode.GRAPH else search_mode

        if mode == SearchMode.RAW:
            if not sources_dir.exists():
                return {"status": "error", "message": "No sources directory found for raw search."}
            results = retriever.search_raw(query, sources_dir, top_k=top_k)
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
        llm_client = LLMClient.from_config(cfg)
    except Exception:
        llm_client = None

    if llm_client:
        try:
            synthesizer = TokenOverlapRetriever(wiki)
            synthesis = synthesizer.synthesize(query, results, llm_client=llm_client)
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


@server.tool(name="wiki_status")
def wiki_status(
    wiki_dir: str = ".",
) -> dict[str, Any]:
    """Show per-wiki statistics."""
    try:
        cfg = resolve_config(cli_overrides={PATHS_WIKI_DIR: wiki_dir})
        sources_dir, output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)
        stats = collect_stats(
            sources_dir=sources_dir if sources_dir.exists() else None,
            output_dir=output_dir if output_dir.exists() else None,
        )
        return {"status": "ok", "stats": stats}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@server.tool(name="wiki_distill")
def wiki_distill(
    wiki_dir: str = ".",
) -> dict[str, Any]:
    """Synthesize captured notes into coherent wiki pages."""
    try:
        cfg = resolve_config(cli_overrides={PATHS_WIKI_DIR: wiki_dir})
        sources_dir, _output_dir, _cache_dir, _state_dir = resolve_project_paths(cfg)

        try:
            llm_client = LLMClient.from_config(cfg)
        except Exception:
            llm_client = None

        paths = distill(sources_dir, llm_client=llm_client)
        return {
            "status": "ok",
            "count": len(paths),
            "files": [str(p) for p in paths],
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@server.tool(name="config_get")
def config_get(
    key: str,
) -> dict[str, Any]:
    """Print a resolved config value."""
    if not validate_config_key(key):
        return {"status": "error", "message": f"Unknown config key: {key}"}
    try:
        cfg = resolve_config()
        value = cfg.get(key)
        return {"status": "ok", "key": key, "value": value}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@server.tool(name="config_set")
def config_set(
    key: str,
    value: str,
    global_: bool = False,
) -> dict[str, Any]:
    """Set a config value."""
    if not validate_config_key(key):
        return {"status": "error", "message": f"Unknown config key: {key}"}
    try:
        coerced = coerce_config_value(value, key)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    if global_:
        config_file = get_global_config_path(create=True)
    else:
        config_file = get_project_config_path(create=False)
        if not config_file.exists():
            return {"status": "error", "message": "Not in a KNOX vault — run `knox init` first or use `--global`"}
    try:
        set_config_file_value(config_file, key, coerced)
        return {"status": "ok", "key": key, "value": coerced, "file": str(config_file)}
    except (OSError, ValueError) as exc:
        return {"status": "error", "message": f"Failed to write {config_file}: {exc}"}


def main() -> None:
    """Run the KNOX MCP server over stdio transport."""
    server.run(transport="stdio")


def create_mcp_server(port: int = 8000) -> FastMCP:
    """Create a new FastMCP instance with all KNOX tools registered.

    Used by both ``knox mcp`` (stdio) and the ACP agent's companion mode
    (HTTP on the given *port*).
    """
    srv = FastMCP("KNOX", port=port)

    # Register all tools on the new instance.
    # The tool functions are already defined in this module; we re-register
    # them here so the new FastMCP instance has the same tool set.
    srv.tool(name="parse_document")(parse_document)
    srv.tool(name="compile_wiki")(compile_wiki)
    srv.tool(name="wiki_capture")(wiki_capture)
    srv.tool(name="wiki_learn")(wiki_learn)
    srv.tool(name="memory_add")(memory_add)
    srv.tool(name="memory_search")(memory_search)
    srv.tool(name="memory_synthesize")(memory_synthesize)
    srv.tool(name="init_wiki")(init_wiki)
    srv.tool(name="query_wiki")(query_wiki)
    srv.tool(name="wiki_status")(wiki_status)
    srv.tool(name="wiki_distill")(wiki_distill)
    srv.tool(name="config_get")(config_get)
    srv.tool(name="config_set")(config_set)

    return srv
