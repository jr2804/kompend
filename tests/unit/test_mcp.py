"""Tests for KNOX MCP server — FastMCP-based tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from knox.cli.app import app as cli_app
from knox.cli.app import register_all_commands
from knox.mcp.server import compile_wiki, parse_document
from knox.mcp.server import server as mcp_server
from knox.mcp.tools.config import config_check
from knox.mcp.tools.query import query_wiki

# ---------------------------------------------------------------------------
# CLI ↔ MCP parity
# ---------------------------------------------------------------------------
#
# These commands are server-bootstrap / interactive and intentionally have no
# MCP equivalent. They are excluded from the parity assertion.
_EXCLUDED_CLI_COMMANDS = {"mcp", "serve", "agent", "agents"}
_MCP_TO_CLI_ALIASES = {"wiki_list_attachments": "attachments_list"}

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


def test_cli_mcp_parity() -> None:
    """Every functional CLI command has a matching MCP tool."""
    register_all_commands()
    cli_cmds: set[str] = set()
    for cmd in cli_app.registered_commands:
        cli_cmds.add(_normalize_cli(cmd.name))
    for group in cli_app.registered_groups:
        sub_app = group.typer_instance
        gname = sub_app.info.name or str(group.name)
        gnorm = _normalize_cli(gname)
        for cmd in sub_app.registered_commands:
            cli_cmds.add(f"{gnorm}_{_normalize_cli(cmd.name)}")

    mcp_tools = set(mcp_server._tool_manager._tools)
    mcp_norm = {_normalize_mcp(t) for t in mcp_tools}
    # Apply explicit aliases: remove the aliased CLI command from the
    # expected set (the MCP tool already covers it).
    for cli_cmd in _MCP_TO_CLI_ALIASES.values():
        cli_cmds.discard(cli_cmd)

    missing = sorted((cli_cmds - _EXCLUDED_CLI_COMMANDS) - mcp_norm)
    assert not missing, f"CLI commands missing MCP tools: {missing}"


def test_mcp_cli_parity_reverse() -> None:
    """Every MCP tool has a matching CLI command (reverse direction).

    Guards against MCP tools that drift out of the CLI surface — e.g. a
    tool added to the MCP server but never exposed as a CLI command, or a
    CLI command renamed without updating the MCP side.
    """
    register_all_commands()
    cli_cmds: set[str] = set()
    for cmd in cli_app.registered_commands:
        cli_cmds.add(_normalize_cli(cmd.name))
    for group in cli_app.registered_groups:
        sub_app = group.typer_instance
        gname = sub_app.info.name or str(group.name)
        gnorm = _normalize_cli(gname)
        for cmd in sub_app.registered_commands:
            cli_cmds.add(f"{gnorm}_{_normalize_cli(cmd.name)}")

    mcp_tools = set(mcp_server._tool_manager._tools)
    mcp_norm = {_normalize_mcp(t) for t in mcp_tools}
    # MCP tools that intentionally have no CLI equivalent (server bootstrap,
    # interactive, or pure-MCP conveniences).
    mcp_only = {"list_attachments", "read_attachment", "attachment_info"}
    orphan = sorted(mcp_norm - cli_cmds - mcp_only)
    assert not orphan, f"MCP tools without a CLI command: {orphan}"


def _normalize_cli(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").strip("_")


def _normalize_mcp(name: str) -> str:
    return name[len("wiki_") :] if name.startswith("wiki_") else name


def test_query_wiki_works_without_knox_state(tmp_path: Path) -> None:
    """wiki_query searches markdown without a vector index — vault root required."""
    vault = tmp_path / "vault"
    (vault / ".knox").mkdir(parents=True)
    (vault / ".knox" / "config.toml").write_text("", encoding="utf-8")
    (vault / "wiki" / "concepts").mkdir(parents=True)
    (vault / "wiki" / "index.md").write_text("# Wiki Index\n", encoding="utf-8")
    (vault / "wiki" / "concepts" / "transformers.md").write_text(
        "---\ntype: Concept\ntitle: Transformers\n---\n\nTransformers use self-attention mechanisms for language modeling.\n",
        encoding="utf-8",
    )

    # raw=True to skip the model requirement — this test validates that markup
    # files can be loaded and searched without .knox/ vector state.
    result = query_wiki("transformers attention", wiki_dir=str(vault), raw=True)

    assert result["status"] == "ok"
    assert result["results"], "keyword search must return hits without .knox/ state"
    assert any("transformers" in str(r.get("article_id", "")).lower() for r in result["results"])


def test_query_wiki_errors_when_no_pages(tmp_path: Path) -> None:
    """Querying a non-vault dir fails fast (LookupError) before page loading."""
    with pytest.raises(LookupError, match="not a KNOX vault root"):
        query_wiki("anything", wiki_dir=str(tmp_path / "empty"))
