"""KNOX MCP server — thin wrapper that delegates to build_pipeline().

Uses the official ``mcp`` Python SDK (FastMCP) for stdio transport.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from knox.capture.capture import capture
from knox.capture.learn import learn
from knox.core.config import EmbeddingConfig, LLMConfig, OutputConfig, resolve_config
from knox.core.pipeline import build_pipeline
from knox.core.pipeline_defs import Compiler, ImageMode, OutputFormat, PipelineConfig
from knox.llm.client import LLMClient
from knox.memory import MemoryStore
from knox.preprocessing.parsers.base import ParserRouter
from knox.preprocessing.parsers.docling_parser import DoclingParser

logger = logging.getLogger(__name__)

server = FastMCP("KNOX")


@server.tool(name="parse_document")
def parse_document(file_path: str) -> dict[str, Any]:
    """Parse a document and return structured elements."""
    path = Path(file_path)
    if not path.exists():
        return {"status": "error", "message": f"File not found: {file_path}"}

    router = ParserRouter()
    try:
        router.register(DoclingParser())
    except Exception as exc:
        logger.warning("DoclingParser not available: %s", exc)

    try:
        doc = router.parse(path)
        return {
            "status": "ok",
            "title": doc.title or path.name,
            "page_count": doc.page_count,
            "elements": [
                {
                    "type": str(el.type),
                    "content": str(el.content)[:500],
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
    chunk_size: int = 512,
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
    source_path = Path(sources_dir)
    if not source_path.exists():
        return {"status": "error", "message": f"Sources directory not found: {sources_dir}"}

    # Validate and convert string params to enums
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
        source_dir=source_path,
        output_dir=Path(output),
        llm=LLMConfig(default_model="", vlm_model=vlm_model),
        embedding=EmbeddingConfig(model=""),
        online_sources=urls or [],
        output=OutputConfig(format=format_enum),
        chunk_size=chunk_size,
        image_mode=image_mode_enum,
        compiler=compiler_enum,
    )

    pipeline = build_pipeline(config)

    try:
        wiki = pipeline.run()
        ext = "md" if output_format == "markdown" else output_format
        output_path = Path(output) / f"wiki.{ext}"
        return {
            "status": "ok",
            "node_count": len(wiki.nodes),
            "edge_count": len(wiki.edges),
            "toc_count": len(wiki.toc),
            "output_path": str(output_path.resolve()),
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
) -> dict[str, Any]:
    """Extract knowledge items from text and save as source files.

    Uses LLM to extract decisions, discoveries, corrections, and references
    from conversation text. Results are written to sources/_captured/.
    """
    source_path = Path(sources_dir)
    source_path.mkdir(parents=True, exist_ok=True)

    try:
        cfg = resolve_config()
        default_model = cfg.get("compile.llm.default_model")
        llm_client = LLMClient(default_model=default_model)
    except Exception:
        llm_client = None

    try:
        paths = capture(text, sources_dir=source_path, llm_client=llm_client)
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
    source_path = Path(sources_dir)
    source_path.mkdir(parents=True, exist_ok=True)

    try:
        path = learn(text, sources_dir=source_path, tags=tags)
        return {
            "status": "ok",
            "file": str(path),
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
        store = MemoryStore(base_dir=Path(wiki_dir))
        store.add_entry(query=query, answer=answer, feedback=feedback)
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
        store = MemoryStore(base_dir=Path(wiki_dir))
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
        store = MemoryStore(base_dir=Path(wiki_dir))
        entries = store.get_recent_entries(limit=limit)
        if not entries:
            store.close()
            return {"status": "ok", "message": "No memory entries found to synthesize."}

        base_dir = Path(wiki_dir)
        context_path = base_dir / "sources" / "_memory" / "system_context.md"
        context_path.parent.mkdir(parents=True, exist_ok=True)

        with open(context_path, "w", encoding="utf-8") as f:
            f.write("# System Context (Auto-Generated)\n\n")
            f.write("This file is automatically generated based on recent user queries and feedback.\n\n")
            f.write("## Recent Query Themes\n\n")
            for e in entries[:10]:
                f.write(f"- {e['query']}\n")

        store.close()
        return {"status": "ok", "message": f"Synthesized context written to {context_path}"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def main() -> None:
    """Run the KNOX MCP server over stdio transport."""
    server.run(transport="stdio")
