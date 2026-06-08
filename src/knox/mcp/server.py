"""KNOX MCP server — thin wrapper that delegates to build_pipeline().

Uses the official ``mcp`` Python SDK (FastMCP) for stdio transport.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from knox.core.pipeline import PipelineConfig, build_pipeline

logger = logging.getLogger(__name__)

server = FastMCP("KNOX")


@server.tool(name="parse_document")
def parse_document(file_path: str) -> dict[str, Any]:
    """Parse a document and return structured elements."""
    from knox.preprocessing.parsers.base import ParserRouter
    from knox.preprocessing.parsers.docling_parser import DoclingParser

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

    valid_formats = {"markdown", "json", "typst"}
    if output_format not in valid_formats:
        return {
            "status": "error",
            "message": f"Unknown format: {output_format}. Use markdown, json, or typst.",
        }

    valid_compilers = {"graph", "wicer"}
    if compiler not in valid_compilers:
        return {
            "status": "error",
            "message": f"Unknown compiler: {compiler}. Use graph or wicer.",
        }

    config = PipelineConfig(
        source_dir=source_path,
        output_dir=Path(output),
        online_sources=urls or [],
        compile_target=output_format,
        chunk_size=chunk_size,
        image_mode=image_mode,
        compiler=compiler,
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


def main() -> None:
    """Run the KNOX MCP server over stdio transport."""
    server.run(transport="stdio")
