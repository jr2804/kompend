"""Tests for KNOX MCP server — FastMCP-based tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from knox.mcp.server import compile_wiki, parse_document


# ---------------------------------------------------------------------------
# parse_document
# ---------------------------------------------------------------------------


class TestParseDocument:
    def test_returns_error_on_missing_file(self) -> None:
        result = parse_document("/nonexistent/file.pdf")
        assert result["status"] == "error"
        assert "not found" in result["message"]


# ---------------------------------------------------------------------------
# compile_wiki
# ---------------------------------------------------------------------------


class TestCompileWiki:
    def test_returns_error_on_missing_dir(self) -> None:
        result = compile_wiki(sources_dir="/nonexistent/sources")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_returns_error_on_unknown_format(self) -> None:
        with patch.object(Path, "exists", return_value=True):
            result = compile_wiki(
                sources_dir="/tmp/fake",
                output_format="latex",
            )
        assert result["status"] == "error"
        assert "Unknown format" in result["message"]
