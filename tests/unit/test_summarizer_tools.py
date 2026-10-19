"""Tests for the agentic summarizer toolset (summarizer_tools.py)."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from pydantic_ai import ModelRetry, RunContext

from kompend.agents.types import SummarizerDeps
from kompend.compilation.summarizer_tools import next_node, summarize_node


class _Ctx:
    """Minimal stand-in for ``RunContext`` (tools only read ``deps``)."""

    def __init__(self, deps: SummarizerDeps) -> None:
        self.deps = deps


def _ctx(deps: SummarizerDeps) -> RunContext[SummarizerDeps]:
    return cast("RunContext[SummarizerDeps]", _Ctx(deps))


def _node(node_id: str) -> dict[str, Any]:
    return {"id": node_id, "heading_chain": "h", "content": "c"}


class TestNextNode:
    def test_returns_first_pending_node_with_an_explicit_id(self) -> None:
        deps = SummarizerDeps(nodes=[_node("a"), _node("b")], source_id="root")

        out = asyncio.run(next_node(_ctx(deps)))

        assert out.startswith("node_id: a")
        assert deps.current_node is not None
        assert deps.current_node["id"] == "a"

    def test_skips_nodes_already_recorded(self) -> None:
        deps = SummarizerDeps(nodes=[_node("a"), _node("b")], source_id="root", results={"a": "x"})

        out = asyncio.run(next_node(_ctx(deps)))

        assert "node_id: b" in out

    def test_reports_when_every_node_is_recorded(self) -> None:
        deps = SummarizerDeps(nodes=[_node("a")], source_id="root", results={"a": "x"})

        assert asyncio.run(next_node(_ctx(deps))) == "No more nodes to summarize."


class TestSummarizeNode:
    """Regression (p863 request_limit storm): a mangled node id used to be
    written to ``deps.results`` unvalidated, so the resume loop saw "progress"
    while the real payload never shrank and re-ran the same session forever.
    """

    def test_records_a_payload_node_and_strips_the_summary(self) -> None:
        deps = SummarizerDeps(nodes=[_node("a")], source_id="root")

        out = asyncio.run(summarize_node(_ctx(deps), "a", "  body  "))

        assert deps.results == {"a": "body"}
        assert deps.current_node is None
        assert "a" in out

    def test_tolerates_surrounding_whitespace_on_the_id(self) -> None:
        deps = SummarizerDeps(nodes=[_node("a")], source_id="root")

        asyncio.run(summarize_node(_ctx(deps), " a ", "body"))

        assert deps.results == {"a": "body"}

    def test_rejects_an_unknown_id_with_a_retry(self) -> None:
        deps = SummarizerDeps(nodes=[_node("a")], source_id="root")

        with pytest.raises(ModelRetry) as excinfo:
            asyncio.run(summarize_node(_ctx(deps), "a/mangled", "body"))

        assert "Unknown node id 'a/mangled'" in str(excinfo.value)
        assert "'a'" in str(excinfo.value)
        assert deps.results == {}
