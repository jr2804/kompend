"""Tests for KNOX MCP server — FastMCP-based tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from knox.mcp.server import compile_wiki, parse_document
from knox.mcp.tools.config import config_check

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
        with patch.object(Path, "exists", return_value=True), patch("knox.mcp.tools.knowledge.build_pipeline") as mock_build:
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


# ---------------------------------------------------------------------------
# config_check
# ---------------------------------------------------------------------------


class TestConfigCheck:
    def test_unknown_key_returns_error_status(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text("[compile]\nlegacy_removed = 'oops'\n", encoding="utf-8")
        result = config_check(paths=[str(cfg)])
        assert result["status"] == "error"
        assert result["totals"]["error"] >= 1

    def test_clean_config_returns_ok_status(self, tmp_path: Path) -> None:
        cfg = tmp_path / "empty.toml"
        cfg.write_text("", encoding="utf-8")
        result = config_check(paths=[str(cfg)])
        assert result["status"] == "ok"
        assert result["totals"]["error"] == 0

    def test_merge_true_combines_files(self, tmp_path: Path) -> None:
        a = tmp_path / "a.toml"
        a.write_text("", encoding="utf-8")
        b = tmp_path / "b.toml"
        b.write_text("[compile]\nworkers = 8\n", encoding="utf-8")
        result = config_check(paths=[str(a), str(b)], merge=True)
        assert result["status"] == "ok"
        # One merged report.
        assert len(result["reports"]) == 1

    def test_merge_false_returns_per_file_reports(self, tmp_path: Path) -> None:
        a = tmp_path / "a.toml"
        a.write_text("", encoding="utf-8")
        b = tmp_path / "b.toml"
        b.write_text("[compile]\nworkers = 8\n", encoding="utf-8")
        result = config_check(paths=[str(a), str(b)], merge=False)
        assert len(result["reports"]) == 2
