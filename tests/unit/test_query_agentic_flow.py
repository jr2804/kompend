"""Agentic query flow — plumbing, real tool content, gating (no LLM).

Covers the opt-in ``search.agentic`` path end-to-end without any LLM call:
the flow builds deps with the raw retriever, the tools reach real content
through it, and the non-agentic default is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic_ai import RunContext

from knox.agents.types import QueryAgentDeps, QuerySearchProtocol
from knox.cli._query_helpers import should_use_agentic
from knox.compilation.query_tools import add_finding, read_page, search_wiki
from knox.core.config import ResolvedConfig
from knox.core.models import CompiledWiki, WikiNode
from knox.search.retriever import KeywordRetriever, SearchResult, run_agentic_query


@dataclass
class _Ctx:
    deps: QueryAgentDeps


class _FakeRetriever:
    """Raw retrieval surface — structurally satisfies QuerySearchProtocol."""

    def __init__(self, hits: list[SearchResult], pages: dict[str, str]) -> None:
        self._hits = hits
        self._pages = pages

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        return self._hits[:top_k]

    def read_page(self, slug: str) -> str | None:
        return self._pages.get(slug)


async def test_search_wiki_returns_real_retrieved_content() -> None:
    deps = _deps(_fake())
    out = await search_wiki(_ctx(deps), "P.863 question")
    assert "P.863" in out
    assert "Full body of p1" in out


async def test_read_page_reads_body_and_reports_missing() -> None:
    deps = _deps(_fake())
    assert await read_page(_ctx(deps), "p1") == "Full body of p1"
    assert "not found" in await read_page(_ctx(deps), "nope")


async def test_add_finding_records_by_index() -> None:
    deps = _deps(_fake())
    await add_finding(_ctx(deps), 0, "  answer fragment  ")
    assert deps.results == {0: "answer fragment"}


def _ctx(deps: QueryAgentDeps) -> RunContext[QueryAgentDeps]:
    return cast(RunContext[QueryAgentDeps], _Ctx(deps))


def _deps(retriever: QuerySearchProtocol) -> QueryAgentDeps:
    return QueryAgentDeps(question="What is P.863?", sub_questions=[], retriever=retriever)


def test_run_agentic_query_builds_deps_and_returns_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake()
    seen: dict[str, object] = {}

    class _Agent:
        def run(self, deps: QueryAgentDeps, *, message_history: object = None) -> str:
            seen["deps"] = deps
            assert deps.question == "What is P.863?"
            assert deps.sub_questions == []
            assert deps.retriever is fake
            assert deps.max_sub_questions == 7
            return "SYNTHESIZED ANSWER"

    def _factory(model: str, **kwargs: object) -> _Agent:
        seen["model"] = model
        return _Agent()

    monkeypatch.setattr("knox.search.retriever.make_agentic_query_agent", _factory)
    answer = run_agentic_query("What is P.863?", retriever=fake, model="ollama/x", max_sub_questions=7)
    assert answer == "SYNTHESIZED ANSWER"
    assert seen["model"] == "ollama/x"


def _fake() -> _FakeRetriever:
    hits = [
        SearchResult(article_id="p1", title="P.863", snippet="s", score=1.0, content="Full body of p1"),
        SearchResult(article_id="p2", title="MOS-LQO", snippet="s", score=0.5, content="Full body of p2"),
    ]
    return _FakeRetriever(hits, {"p1": "Full body of p1"})


def test_retriever_read_page_returns_node_content() -> None:
    node = WikiNode(id="p1", title="P.863", content="BODY", node_type="concept")
    wiki = CompiledWiki(nodes=[node], edges=[], toc=[])
    retriever = KeywordRetriever(wiki)
    assert retriever.read_page("p1") == "BODY"
    assert retriever.read_page("missing") is None


def test_agentic_disabled_by_default() -> None:
    assert should_use_agentic(_cfg(False), raw=False, model="m", tag_groups=[], attr_filters=[]) is False


def test_agentic_gating_requires_model_and_no_filters() -> None:
    assert should_use_agentic(_cfg(True), raw=False, model="m", tag_groups=[], attr_filters=[]) is True
    assert should_use_agentic(_cfg(True), raw=True, model="m", tag_groups=[], attr_filters=[]) is False
    assert should_use_agentic(_cfg(True), raw=False, model=None, tag_groups=[], attr_filters=[]) is False
    assert should_use_agentic(_cfg(True), raw=False, model="m", tag_groups=[["a"]], attr_filters=[]) is False
    assert should_use_agentic(_cfg(True), raw=False, model="m", tag_groups=[], attr_filters=[("k", "v")]) is False


def _cfg(enabled: bool) -> ResolvedConfig:
    return cast(
        ResolvedConfig,
        SimpleNamespace(knox=SimpleNamespace(search=SimpleNamespace(agentic=SimpleNamespace(enabled=enabled)))),
    )
