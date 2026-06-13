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
                sources_dir="/tmp/fake",  # noqa: S108
                output_format="latex",
            )
        assert result["status"] == "error"
        assert "Unknown format" in result["message"]

    def test_vlm_model_passed_to_pipeline(self) -> None:
        with patch.object(Path, "exists", return_value=True), patch("knox.mcp.server.build_pipeline") as mock_build:
            mock_pipeline = mock_build.return_value
            mock_pipeline.run.return_value = type(
                "Wiki",
                (),
                {
                    "nodes": [],
                    "edges": [],
                    "toc": [],
                    "quality_score": 0.0,
                    "iteration": 0,
                    "metadata": {},
                },
            )()
            compile_wiki(
                sources_dir="/tmp/fake",  # noqa: S108
                vlm_model="openai/gpt-4o",
            )
        assert mock_build.called
        passed_config = mock_build.call_args.args[0]
        assert passed_config.llm.vlm_model == "openai/gpt-4o"
