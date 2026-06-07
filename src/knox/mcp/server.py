"""KNOX MCP server — thin wrapper that delegates to build_pipeline()."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from knox.core.models import CompiledWiki
from knox.core.pipeline import PipelineConfig, build_pipeline
from knox.preprocessing.parsers.base import ParserRouter
from knox.preprocessing.parsers.docling_parser import DoclingParser

logger = logging.getLogger(__name__)


def _wiki_to_dict(wiki: CompiledWiki) -> dict[str, Any]:
    """Convert a CompiledWiki to a JSON-serializable dict."""
    import dataclasses

    return {
        "nodes": [dataclasses.asdict(n) for n in wiki.nodes],
        "edges": wiki.edges,
        "toc": wiki.toc,
        "quality_score": wiki.quality_score,
        "iteration": wiki.iteration,
        "metadata": wiki.metadata,
    }


class KnoxMCPServer:
    """MCP server exposing KNOX pipeline tools via stdio transport.

    Zero business logic — all pipeline construction is delegated to
    :func:`knox.core.pipeline.build_pipeline`.
    """

    def __init__(self) -> None:
        self.router = ParserRouter()
        try:
            self.router.register(DoclingParser())
        except Exception as exc:
            logger.warning("DoclingParser not available: %s", exc)

    async def parse_document(self, file_path: str) -> dict[str, Any]:
        """MCP tool: parse a document and return structured elements."""
        path = Path(file_path)
        if not path.exists():
            return {"status": "error", "message": f"File not found: {file_path}"}

        try:
            doc = self.router.parse(path)
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

    async def compile_wiki(
        self,
        sources_dir: str,
        output: str = "./output",
        format: str = "markdown",
        chunk_size: int = 512,
        image_mode: str = "file",
        urls: list[str] | None = None,
    ) -> dict[str, Any]:
        """MCP tool: compile sources into a knowledge wiki."""
        source_path = Path(sources_dir)
        if not source_path.exists():
            return {"status": "error", "message": f"Sources directory not found: {sources_dir}"}

        valid_formats = {"markdown", "json", "typst"}
        if format not in valid_formats:
            return {"status": "error", "message": f"Unknown format: {format}. Use markdown, json, or typst."}

        config = PipelineConfig(
            source_dir=source_path,
            output_dir=Path(output),
            online_sources=urls or [],
            compile_target=format,
            chunk_size=chunk_size,
            image_mode=image_mode,
        )

        pipeline = build_pipeline(config)

        try:
            wiki = pipeline.run()
            output_path = Path(output) / f"wiki.{'md' if format == 'markdown' else format}"
            return {
                "status": "ok",
                "node_count": len(wiki.nodes),
                "edge_count": len(wiki.edges),
                "toc_count": len(wiki.toc),
                "output_path": str(output_path.resolve()),
                "wiki": _wiki_to_dict(wiki),
            }
        except Exception as exc:
            logger.exception("Compilation failed")
            return {"status": "error", "message": str(exc)}


def _handle_request(request: dict[str, Any], server: KnoxMCPServer) -> dict[str, Any]:
    """Route and execute an MCP request synchronously using asyncio."""
    import asyncio

    method = request.get("method", "")
    params = request.get("params", {})

    if method == "parse_document":
        return asyncio.run(server.parse_document(**params))
    elif method == "compile_wiki":
        return asyncio.run(server.compile_wiki(**params))
    else:
        return {"status": "error", "message": f"Unknown method: {method}"}


def main() -> None:
    """Run the KNOX MCP server over stdio transport.

    Reads JSON-RPC-style requests from stdin and writes responses to stdout.
    Each line is a single JSON request; each response is a single JSON line.
    """
    server = KnoxMCPServer()
    logger.info("KNOX MCP server started on stdio")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            request_id = request.get("id", None)
            result = _handle_request(request, server)

            response = {"id": request_id, "result": result}
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError as exc:
            error_response = {"id": None, "error": {"message": f"Invalid JSON: {exc}"}}
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            error_response = {"id": None, "error": {"message": str(exc)}}
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()