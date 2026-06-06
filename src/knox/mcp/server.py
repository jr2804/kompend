"""KNOX MCP server."""

from __future__ import annotations

from knox.core.models import CompiledWiki
from knox.preprocessing.parsers.base import ParserRouter
from knox.preprocessing.parsers.docling_parser import DoclingParser


class KnoxMCPServer:
    """MCP server exposing KNOX pipeline tools."""

    def __init__(self) -> None:
        self.router = ParserRouter()
        self.router.register(DoclingParser())

    async def parse_document(self, file_path: str) -> dict:
        """MCP tool: parse a document and return structured elements."""
        return {"status": "stub", "file": file_path}

    async def compile_wiki(self, sources_dir: str) -> dict:
        """MCP tool: compile sources into a knowledge wiki."""
        return {"status": "stub", "sources_dir": sources_dir}
