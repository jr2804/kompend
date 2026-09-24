"""Tests for restored agentic agent wiring (0iv2.5 / .6 / .10).

Regression guard for commit ``d443c583``, which dropped the
``make_agentic_*`` factory shims for the summarizer, context-chunk, and
query agents. Each agent survived with class + tools + prompt, but with no
factory and no construction site — the same double bug distill had. The
system prompts still instruct the model to call tools, so each factory must
register exactly the tools its prompt declares.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from pydantic_ai import RunContext

from knox.agents.context_chunk_agentic import make_agentic_context_chunk_agent
from knox.agents.query_agentic import make_agentic_query_agent
from knox.agents.summarizer_agentic import AgenticSummarizerAgent, make_agentic_summarizer_agent
from knox.agents.types import ContextChunkDeps, QueryAgentDeps
from knox.chunking.contextual import ContextualEnricher
from knox.compilation.query_tools import read_page, search_wiki
from knox.core.models import Chunk, ChunkNode, ChunkTree, DocumentElement, ElementType, ParsedDocument
from knox.core.pipeline.config import Compiler, EmbeddingConfig, LLMConfig, PipelineConfig
from knox.core.pipeline.orchestrator import build_pipeline
from knox.llm.context_window import ContextSource, ContextWindow
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
            "knox.core.pipeline.stages._factory.discover_context_window",
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
