"""Tests for restored agentic agent wiring (0iv2.5 / .6 / .10).

Regression guard for commit ``d443c583``, which dropped the
``make_agentic_*`` factory shims for the summarizer, context-chunk, and
query agents. Each agent survived with class + tools + prompt, but with no
factory and no construction site — the same double bug distill had. The
system prompts still instruct the model to call tools, so each factory must
register exactly the tools its prompt declares.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import RunContext

from kompend.agents.context_chunk_agentic import AgenticContextChunkAgent, make_agentic_context_chunk_agent
from kompend.agents.distill_agentic import make_agentic_distill_agent
from kompend.agents.query_agentic import make_agentic_query_agent
from kompend.agents.summarizer_agentic import AgenticSummarizerAgent, make_agentic_summarizer_agent
from kompend.agents.types import ContextChunk, ContextChunkDeps, DistillAgentDeps, DistillGroup, QueryAgentDeps, SummarizerDeps, SummaryNode
from kompend.chunking.contextual import ContextualEnricher
from kompend.compilation.context_chunk_tools import next_chunk
from kompend.compilation.query_tools import read_page, search_wiki
from kompend.core.config._configs import BudgetConfig, SearchConfig, SearchContextualConfig
from kompend.core.models import Chunk, ChunkNode, ChunkTree, DocumentElement, ElementType, ParsedDocument
from kompend.core.pipeline.config import Compiler, EmbeddingConfig, LLMConfig, PipelineConfig
from kompend.core.pipeline.orchestrator import build_pipeline
from kompend.llm.context_window import ContextSource, ContextWindow
from tests.conftest import make_tmp_vault

# NOTE: system-prompt-tools == registered-tools is guarded centrally for every
# agent in tests/unit/test_agent_templates.py; only the per-agent tool lists
# remain here.


@dataclass
class _Hit:
    article_id: str
    title: str
    content: str


@dataclass
class _FakeRetriever:
    """Minimal QuerySearchProtocol stand-in — raw hits only, no LLM."""

    pages: dict[str, str]

    def search(
        self,
        query: str,
        top_k: int,
        *,
        tags_all: object = None,
        tags_dnf: object = None,
        attributes_all: object = None,
        attribute_keys: object = None,
    ) -> list[_Hit]:  # noqa: ARG002
        return [_Hit(article_id=slug, title=slug.title(), content=body) for slug, body in list(self.pages.items())[:top_k]]

    def read_page(self, slug: str) -> str | None:
        return self.pages.get(slug)


@dataclass
class _Ctx:
    deps: QueryAgentDeps


@dataclass
class _ChunkCtx:
    deps: ContextChunkDeps


class TestAgenticSummarizerAgentWiring:
    def test_factory_registers_prompt_declared_tools(self) -> None:
        agent = make_agentic_summarizer_agent(model_string="ollama/gpt-oss:20b")
        built = agent._build_agent()
        assert sorted(built._function_toolset.tools.keys()) == ["next_node", "summarize_node"]

    def test_factory_injects_harness_capabilities(self) -> None:
        agent = make_agentic_summarizer_agent(model_string="ollama/gpt-oss:20b", context_window=8192)
        assert agent.extra_capabilities

    def test_build_pipeline_wires_agentic_summarizer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: NodeSummarizer was built without `agent=`, so every
        summary silently fell back to the extractive path (0iv2.6).
        """
        monkeypatch.setattr(
            "kompend.core.pipeline.stages._factory.discover_context_window",
            lambda *a, **k: ContextWindow(max_input=8192, max_output=2048, source=ContextSource.FALLBACK),
        )
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(default_model="ollama/gpt-oss:20b", vlm_model="vlm"),
            embedding=EmbeddingConfig(model="hash"),
            compiler=Compiler.GRAPH,
        )
        pipeline = build_pipeline(config)
        assert pipeline.summarizer is not None
        assert isinstance(pipeline.summarizer._agent, AgenticSummarizerAgent)  # noqa: SLF001

    def test_build_pipeline_passes_summary_requests_per_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "kompend.core.pipeline.stages._factory.discover_context_window",
            lambda *a, **k: ContextWindow(max_input=8192, max_output=2048, source=ContextSource.FALLBACK),
        )
        config = PipelineConfig(
            source_dir=make_tmp_vault(),
            output_dir=make_tmp_vault(),
            llm=LLMConfig(default_model="ollama/gpt-oss:20b", vlm_model="vlm"),
            embedding=EmbeddingConfig(model="hash"),
            compiler=Compiler.GRAPH,
            budget=BudgetConfig(summary_requests_per_node=5),
        )
        pipeline = build_pipeline(config)
        assert pipeline.summarizer is not None
        assert isinstance(pipeline.summarizer._agent, AgenticSummarizerAgent)  # noqa: SLF001
        assert pipeline.summarizer._agent.requests_per_node == 5  # noqa: SLF001


class TestAgenticSummarizerUsageLimits:
    """Regression: pydantic-ai's implicit default caps a run at 50 requests.
    A tree session spends ~2 requests per eligible node (next_node +
    summarize_node) plus a fixed closing cost, so trees with >24 nodes died
    with "request_limit of 50" instead of summarizing (p863 P863E_root), and
    a one-node payload was budgeted at exactly 3 — the closing turn fell off
    the end ("request_limit of 3", p863 T-REC-P.10-201906).
    """

    def test_request_limit_scales_with_node_count(self) -> None:
        agent = make_agentic_summarizer_agent(model_string="ollama/gpt-oss:20b")
        nodes = [SummaryNode(id=str(i), heading_chain="", content="x") for i in range(40)]
        deps = SummarizerDeps(nodes=nodes, source_id="root")
        limits = agent._usage_limits_for(deps)  # noqa: SLF001
        assert limits is not None
        assert limits.request_limit == math.ceil((40 * 2 + 2) * 1.1)  # 91

    def test_headroom_params_are_overridable(self) -> None:
        agent = make_agentic_summarizer_agent(model_string="ollama/gpt-oss:20b")
        nodes = [SummaryNode(id=str(i), heading_chain="", content="x") for i in range(40)]
        deps = SummarizerDeps(nodes=nodes, source_id="root")
        limits = agent._usage_limits_for(deps, requests_per_node=1, headroom_frac=1.0)  # noqa: SLF001
        assert limits is not None
        assert limits.request_limit == 42

    def test_configured_requests_per_node_raises_the_budget(self) -> None:
        """``budget.summary_requests_per_node`` reaches the derived limit.

        Small local models burn prose turns and id retries before every tool
        call; the 2-request/node default starves them mid-batch (p863:
        request_limit of 47 hit four times in a row on one 20-node window).
        """
        agent = make_agentic_summarizer_agent(model_string="ollama/gpt-oss:20b", requests_per_node=4)
        nodes = [SummaryNode(id=str(i), heading_chain="", content="x") for i in range(10)]
        deps = SummarizerDeps(nodes=nodes, source_id="root")
        limits = agent._usage_limits_for(deps)  # noqa: SLF001
        assert limits is not None
        assert limits.request_limit == math.ceil((10 * 4 + 2) * 1.1)  # 47

    def test_single_node_session_covers_its_own_closing_turn(self) -> None:
        """A one-node payload needs next_node + summarize_node + the closing turn."""
        agent = make_agentic_summarizer_agent(model_string="ollama/gpt-oss:20b")
        deps = SummarizerDeps(nodes=[SummaryNode(id="a", heading_chain="", content="x")], source_id="root")
        limits = agent._usage_limits_for(deps)  # noqa: SLF001
        assert limits is not None
        assert limits.request_limit == 5

    def test_session_overhead_is_overridable(self) -> None:
        agent = make_agentic_summarizer_agent(model_string="ollama/gpt-oss:20b")
        deps = SummarizerDeps(nodes=[SummaryNode(id="a", heading_chain="", content="x")], source_id="root")
        limits = agent._usage_limits_for(deps, headroom_frac=1.0, session_overhead=0)  # noqa: SLF001
        assert limits is not None
        assert limits.request_limit == 2


class TestAgenticDistillUsageLimits:
    """Regression: pydantic-ai's implicit default caps a run at 50 requests.

    A multi-group distill session spends ~3 requests per group
    (get_group + save_page + slack) plus a fixed closing cost, so runs with
    >16 groups died with "request_limit of 50" instead of distilling
    (p863: 32 tag groups), and a one-group session was budgeted at exactly its
    own cost.
    """

    def test_request_limit_scales_with_group_count(self) -> None:
        agent = make_agentic_distill_agent(model_string="ollama/gpt-oss:20b")
        groups = [DistillGroup(tag=f"t{i}", notes=[]) for i in range(32)]
        deps = DistillAgentDeps(groups=groups, source_id="distill-all")
        limits = agent._usage_limits_for(deps)  # noqa: SLF001
        assert limits is not None
        assert limits.request_limit == math.ceil((32 * 3 + 2) * 1.1)  # 108

    def test_headroom_params_are_overridable(self) -> None:
        agent = make_agentic_distill_agent(model_string="ollama/gpt-oss:20b")
        groups = [DistillGroup(tag=f"t{i}", notes=[]) for i in range(10)]
        deps = DistillAgentDeps(groups=groups, source_id="distill-all")
        limits = agent._usage_limits_for(deps, requests_per_group=2, headroom_frac=1.0)  # noqa: SLF001
        assert limits is not None
        assert limits.request_limit == 22

    def test_single_group_session_covers_its_own_closing_turn(self) -> None:
        agent = make_agentic_distill_agent(model_string="ollama/gpt-oss:20b")
        deps = DistillAgentDeps(groups=[DistillGroup(tag="dll", notes=[])], source_id="distill-all")
        limits = agent._usage_limits_for(deps)  # noqa: SLF001
        assert limits is not None
        assert limits.request_limit == 6

    def test_session_overhead_is_overridable(self) -> None:
        agent = make_agentic_distill_agent(model_string="ollama/gpt-oss:20b")
        deps = DistillAgentDeps(groups=[DistillGroup(tag="dll", notes=[])], source_id="distill-all")
        limits = agent._usage_limits_for(deps, headroom_frac=1.0, session_overhead=0)  # noqa: SLF001
        assert limits is not None
        assert limits.request_limit == 3


class TestAgenticContextChunkAgentWiring:
    def test_factory_registers_prompt_declared_tools(self) -> None:
        agent = make_agentic_context_chunk_agent(model_string="ollama/gpt-oss:20b")
        built = agent._build_agent()
        assert sorted(built._function_toolset.tools.keys()) == ["next_chunk", "situate_chunk"]

    def test_factory_injects_harness_capabilities(self) -> None:
        agent = make_agentic_context_chunk_agent(model_string="ollama/gpt-oss:20b", context_window=8192)
        assert agent.extra_capabilities


class TestAgenticQueryAgentWiring:
    def test_factory_registers_prompt_declared_tools(self) -> None:
        agent = make_agentic_query_agent(model_string="ollama/gpt-oss:20b")
        built = agent._build_agent()
        assert sorted(built._function_toolset.tools.keys()) == ["add_finding", "read_page", "search_wiki"]

    def test_factory_injects_harness_capabilities(self) -> None:
        agent = make_agentic_query_agent(model_string="ollama/gpt-oss:20b", context_window=8192)
        assert agent.extra_capabilities


class TestQueryToolsReturnRealContent:
    async def test_search_wiki_returns_retrieved_hits(self) -> None:
        deps = QueryAgentDeps(question="q", sub_questions=[], retriever=_FakeRetriever({"p863": "P.863 body text"}))
        out = await search_wiki(_ctx(deps), "P.863", top_k=5)
        assert "P.863 body text" in out
        assert "p863" in out

    async def test_search_wiki_empty_result_is_explicit(self) -> None:
        deps = QueryAgentDeps(question="q", sub_questions=[], retriever=_FakeRetriever({}))
        out = await search_wiki(_ctx(deps), "nothing", top_k=5)
        assert "No wiki pages matched" in out

    async def test_read_page_returns_body_or_not_found(self) -> None:
        deps = QueryAgentDeps(question="q", sub_questions=[], retriever=_FakeRetriever({"a": "A body"}))
        assert await read_page(_ctx(deps), "a") == "A body"
        assert "Page not found" in await read_page(_ctx(deps), "missing")


class TestContextualEnricherAgenticPath:
    def test_agentic_agent_drives_enrich_tree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = make_agentic_context_chunk_agent(model_string="ollama/gpt-oss:20b")
        seen: dict[str, bool] = {}

        def fake_run(deps: ContextChunkDeps, **_: object) -> str:
            seen["ran"] = True
            for chunk in deps.chunks:
                deps.results[chunk["id"]] = f"ctx:{chunk['id']}"
            return ""

        monkeypatch.setattr(agent, "run", fake_run)
        tree = _make_tree(_make_leaf("a"), _make_leaf("b"))
        enriched = ContextualEnricher(agent=agent).enrich_tree(tree, _make_doc())

        assert seen.get("ran")
        contexts = {n.id for n in enriched.nodes.values() if n.contextual_context}
        assert contexts == {"a", "b"}, "agentic path must set contextual_context on each leaf"

    def test_no_agent_skips(self) -> None:
        tree = _make_tree(_make_leaf("a"))
        assert ContextualEnricher(agent=None).enrich_tree(tree, _make_doc()) is tree


class TestContextChunkHeadingContract:
    """The payload carries the heading *path*, never a pre-joined string.

    ``next_chunk`` joins the parts itself; a joined string here made it join
    characters, so the agent saw "S e c t i o n" instead of "Section A".
    """

    def test_payload_carries_the_heading_path_as_a_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = make_agentic_context_chunk_agent(model_string="ollama/gpt-oss:20b")
        captured: list[ContextChunk] = []

        def fake_run(deps: ContextChunkDeps, **_: object) -> str:
            captured.extend(deps.chunks)
            return ""

        monkeypatch.setattr(agent, "run", fake_run)
        ContextualEnricher(agent=agent).enrich_tree(_make_tree(_make_leaf("a")), _make_doc())

        assert captured[0]["heading_chain"] == ["Section A"]

    def test_next_chunk_renders_the_heading_path(self) -> None:
        deps = ContextChunkDeps(
            doc_title="Doc",
            doc_summary="Summary",
            chunks=[ContextChunk(id=7, heading_chain=["Chapter 1", "Section 1.1"], text="body")],
        )

        out = asyncio.run(next_chunk(cast("RunContext[ContextChunkDeps]", _ChunkCtx(deps))))

        assert "heading=Chapter 1 > Section 1.1" in out


def test_build_pipeline_enricher_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """TASK-20: `search.contextual.enabled` defaults false → no enricher, and
    today's chunk-cache output is unchanged.
    """
    monkeypatch.setattr(
        "kompend.core.pipeline.stages._factory.discover_context_window",
        lambda *a, **k: ContextWindow(max_input=8192, max_output=2048, source=ContextSource.FALLBACK),
    )
    config = PipelineConfig(
        source_dir=make_tmp_vault(),
        output_dir=make_tmp_vault(),
        llm=LLMConfig(default_model="ollama/gpt-oss:20b", vlm_model="vlm"),
        embedding=EmbeddingConfig(model="hash"),
        compiler=Compiler.GRAPH,
    )
    assert build_pipeline(config).enricher is None


def test_build_pipeline_enricher_wired_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """TASK-20: enabling the gate constructs a ContextualEnricher with an agent."""
    monkeypatch.setattr(
        "kompend.core.pipeline.stages._factory.discover_context_window",
        lambda *a, **k: ContextWindow(max_input=8192, max_output=2048, source=ContextSource.FALLBACK),
    )
    config = PipelineConfig(
        source_dir=make_tmp_vault(),
        output_dir=make_tmp_vault(),
        llm=LLMConfig(default_model="ollama/gpt-oss:20b", vlm_model="vlm"),
        embedding=EmbeddingConfig(model="hash"),
        compiler=Compiler.GRAPH,
        search=SearchConfig(contextual=SearchContextualConfig(enabled=True)),
    )
    pipeline = build_pipeline(config)
    assert pipeline.enricher is not None
    assert isinstance(pipeline.enricher._agent, AgenticContextChunkAgent)  # noqa: SLF001


def _ctx(deps: QueryAgentDeps) -> RunContext[QueryAgentDeps]:
    return cast("RunContext[QueryAgentDeps]", _Ctx(deps))


def _make_leaf(node_id: str) -> ChunkNode:
    return ChunkNode(
        id=node_id,
        heading_chain=("Section A",),
        chunk=Chunk(id=node_id, text="some text", element_types=[], source_elements=[0]),
    )


def _make_tree(*leaves: ChunkNode) -> ChunkTree:
    root = ChunkNode(id="root", children_ids=[leaf.id for leaf in leaves])
    nodes: dict[str, ChunkNode] = {"root": root}
    for leaf in leaves:
        nodes[leaf.id] = leaf
    return ChunkTree(root_id="root", nodes=nodes)


def _make_doc() -> ParsedDocument:
    return ParsedDocument(
        source_file=Path("doc.md"),
        title="Doc",
        elements=[DocumentElement(type=ElementType.HEADING, content="Introduction")],
    )
