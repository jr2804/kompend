"""Tests for KNOX MCP server (Milestone 6)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from knox.mcp.server import KnoxMCPServer, _handle_request, _wiki_to_dict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp_server() -> KnoxMCPServer:
    return KnoxMCPServer()


# ---------------------------------------------------------------------------
# _wiki_to_dict helper
# ---------------------------------------------------------------------------


def test_wiki_to_dict_empty() -> None:
    from knox.core.models import CompiledWiki

    wiki = CompiledWiki(nodes=[], edges=[], toc=[], metadata={})
    result = _wiki_to_dict(wiki)
    assert result == {
        "nodes": [],
        "edges": [],
        "toc": [],
        "quality_score": 0.0,
        "iteration": 0,
        "metadata": {},
    }


def test_wiki_to_dict_with_nodes() -> None:
    from knox.core.models import CompiledWiki, WikiNode

    node = WikiNode(
        id="n1", title="Test", content="Content",
        node_type="section", source_chunks=["c1"],
        metadata={"level": 1},
    )
    wiki = CompiledWiki(nodes=[node], edges=[], toc=[], metadata={"key": "val"})
    result = _wiki_to_dict(wiki)
    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["id"] == "n1"
    assert result["nodes"][0]["metadata"]["level"] == 1
    assert result["metadata"] == {"key": "val"}


# ---------------------------------------------------------------------------
# parse_document — stubbed with mock due to docling dependency
# ---------------------------------------------------------------------------


class TestParseDocument:
    async def test_returns_error_on_missing_file(self, mcp_server: KnoxMCPServer) -> None:
        result = await mcp_server.parse_document("/nonexistent/file.pdf")
        assert result["status"] == "error"
        assert "not found" in result["message"]


# ---------------------------------------------------------------------------
# compile_wiki — tests the logic indirectly via mock
# ---------------------------------------------------------------------------


class TestCompileWiki:
    async def test_returns_error_on_missing_dir(self, mcp_server: KnoxMCPServer) -> None:
        result = await mcp_server.compile_wiki(sources_dir="/nonexistent/sources")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    async def test_returns_error_on_unknown_format(self, mcp_server: KnoxMCPServer) -> None:
        with patch.object(Path, "exists", return_value=True):
            result = await mcp_server.compile_wiki(
                sources_dir="/tmp/fake",
                format="latex",
            )
        assert result["status"] == "error"
        assert "Unknown format" in result["message"]


# ---------------------------------------------------------------------------
# _handle_request routing
# ---------------------------------------------------------------------------


class TestHandleRequest:
    def test_unknown_method(self, mcp_server: KnoxMCPServer) -> None:
        result = _handle_request({"method": "unknown", "params": {}}, mcp_server)
        assert result["status"] == "error"
        assert "Unknown method" in result["message"]

    def test_parse_with_request_id(self, mcp_server: KnoxMCPServer) -> None:
        result = _handle_request(
            {"id": 1, "method": "parse_document", "params": {"file_path": "/nonexistent.pdf"}},
            mcp_server,
        )
        assert result["status"] == "error"
        assert "not found" in result["message"]