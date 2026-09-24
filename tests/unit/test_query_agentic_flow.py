"""Agentic query flow — plumbing, real tool content, filters, gating (no LLM).

Covers the opt-in ``search.agentic`` path end-to-end without any LLM call:
the flow builds deps with the raw retriever, the tools reach real content
through it and thread tag/attribute filters, and the non-agentic default is
untouched.
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
from knox.search.filters import matches_attr_filter, matches_tag_filter
from knox.search.retriever import KeywordRetriever, SearchResult, run_agentic_query


class _FakeRetriever:
    """Raw retrieval surface — structurally satisfies QuerySearchProtocol."""

    def __init__(self, hits: list[SearchResult], pages: dict[str, str]) -> None:
        self._hits = hits
        self._pages = pages
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        query: str,
        top_k: int,
        *,
        tags_all: object = None,
        tags_dnf: object = None,
        attributes_all: object = None,
        attribute_keys: object = None,
    ) -> list[SearchResult]:
        self.calls.append(
            {
                "query": query,
                "tags_all": tags_all,
                "tags_dnf": tags_dnf,
                "attributes_all": attributes_all,
                "attribute_keys": attribute_keys,
            }
        )
        return self._hits[:top_k]

    def read_page(self, slug: str) -> str | None:
        return self._pages.get(slug)


@dataclass
class _Ctx:
    deps: QueryAgentDeps


def _ctx(deps: QueryAgentDeps) -> RunContext[QueryAgentDeps]:
    return cast(RunContext[QueryAgentDeps], _Ctx(deps))


def _deps(retriever: QuerySearchProtocol, **kwargs: object) -> QueryAgentDeps:
    return QueryAgentDeps(question="What is P.863?", sub_questions=[], retriever=retriever, **kwargs)  # type: ignore[arg-type]


def _fake() -> _FakeRetriever:
    hits = [
        SearchResult(article_id="p1", title="P.863", snippet="s", score=1.0, content="Full body of p1"),
        SearchResult(article_id="p2", title="MOS-LQO", snippet="s", score=0.5, content="Full body of p2"),
    ]
    return _FakeRetriever(hits, {"p1": "Full body of p1"})


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


async def test_search_wiki_threads_deps_filters_to_retriever() -> None:
    fake = _fake()
    deps = _deps(
        fake,
        tag_groups=[["audio", "paper"], ["video"]],
        attr_filters=[("lang", "en"), ("version", None)],
    )
    await search_wiki(_ctx(deps), "q")
    call = fake.calls[-1]
    assert call["tags_dnf"] == [["audio", "paper"], ["video"]]
    assert call["attributes_all"] == ["lang=en"]
    assert call["attribute_keys"] == ["version"]


async def test_search_wiki_sends_no_filters_when_deps_empty() -> None:
    fake = _fake()
    await search_wiki(_ctx(_deps(fake)), "q")
    call = fake.calls[-1]
    assert call["tags_dnf"] is None
    assert call["attributes_all"] is None
    assert call["attribute_keys"] is None


def test_run_agentic_query_builds_deps_and_returns_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake()
    seen: dict[str, object] = {}

    class _Agent:
        def run(self, deps: QueryAgentDeps, *, message_history: object = None) -> str:
            seen["deps"] = deps
            assert deps.question == "What is P.863?"
            assert deps.retriever is fake
            assert deps.max_sub_questions == 7
            assert deps.tag_groups == [["audio"]]
            assert deps.attr_filters == [("lang", "en")]
            return "SYNTHESIZED ANSWER"

    def _factory(model: str, **kwargs: object) -> _Agent:
        seen["model"] = model
        return _Agent()

    monkeypatch.setattr("knox.search.retriever.make_agentic_query_agent", _factory)
    answer = run_agentic_query(
        "What is P.863?",
        retriever=fake,
        model="ollama/x",
        max_sub_questions=7,
        tag_groups=[["audio"]],
        attr_filters=[("lang", "en")],
    )
    assert answer == "SYNTHESIZED ANSWER"
    assert seen["model"] == "ollama/x"


def test_keyword_retriever_applies_filters() -> None:
    nodes = [
        WikiNode(id="a", title="Audio paper", content="audio codec", node_type="concept", metadata={"tags": ["audio", "paper"]}),
        WikiNode(id="b", title="Video paper", content="video codec", node_type="concept", metadata={"tags": ["video"]}),
        WikiNode(id="c", title="Audio note", content="audio note", node_type="concept", metadata={"tags": ["audio"]}),
    ]
    retriever = KeywordRetriever(CompiledWiki(nodes=nodes, edges=[], toc=[]))
    dnf = retriever.search("codec", top_k=5, tags_dnf=[["audio", "paper"]])
    assert {r.article_id for r in dnf} == {"a"}
    keys = retriever.search("note", top_k=5, attribute_keys=["missing"])
    assert keys == []


def test_retriever_read_page_returns_node_content() -> None:
    node = WikiNode(id="p1", title="P.863", content="BODY", node_type="concept")
    retriever = KeywordRetriever(CompiledWiki(nodes=[node], edges=[], toc=[]))
    assert retriever.read_page("p1") == "BODY"
    assert retriever.read_page("missing") is None


def _cfg(enabled: bool) -> ResolvedConfig:
    return cast(
        ResolvedConfig,
        SimpleNamespace(knox=SimpleNamespace(search=SimpleNamespace(agentic=SimpleNamespace(enabled=enabled)))),
    )


def test_agentic_disabled_by_default() -> None:
    assert should_use_agentic(_cfg(False), raw=False, model="m") is False


def test_agentic_gating_allows_filtered_queries() -> None:
    assert should_use_agentic(_cfg(True), raw=False, model="m") is True
    assert should_use_agentic(_cfg(True), raw=True, model="m") is False
    assert should_use_agentic(_cfg(True), raw=False, model=None) is False


def test_shared_matchers_unchanged() -> None:
    assert matches_tag_filter({"audio"}, [["audio"], ["paper"]]) is True
    assert matches_attr_filter({"lang": ["en"]}, [("lang", "en")]) is True
    assert matches_attr_filter({}, [("version", None)]) is False
