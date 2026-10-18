"""Tests for NodeSummarizer (summarizer.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded

from kompend.agents.summarizer_agentic import make_agentic_summarizer_agent
from kompend.agents.types import SummarizerDeps
from kompend.chunking.summarizer import NodeSummarizer
from kompend.core.models import (
    Chunk,
    ChunkNode,
    ChunkStrategy,
    ChunkTree,
    ElementType,
    StructuredTable,
)

# ── tests ────────────────────────────────────────────────────────────────


class TestSummarizerSkip:
    def test_single_pass_skipped(self, single_pass_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(agent=None)
        result = summarizer.summarize(single_pass_tree)
        root = result.nodes["root"]
        assert root.summary == ""

    def test_single_child_skipped_by_default(self, single_child_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(agent=None, skip_single_child=True)
        summarizer.summarize(single_child_tree)
        wrapper = single_child_tree.nodes["w"]
        assert wrapper.summary == ""

    def test_single_child_summarized_when_disabled(self, single_child_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(agent=None, skip_single_child=False)
        summarizer.summarize(single_child_tree)
        wrapper = single_child_tree.nodes["w"]
        assert wrapper.summary != ""


class TestSummarizerExtractiveFallback:
    def test_internal_nodes_get_summaries(self, sample_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(agent=None)
        summarizer.summarize(sample_tree)
        section1 = sample_tree.nodes["doc_h1"]
        assert section1.summary != ""
        assert "SNR" in section1.summary or "threshold" in section1.summary.lower()

    def test_root_gets_summary(self, sample_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(agent=None)
        summarizer.summarize(sample_tree)
        root = sample_tree.nodes["doc_root"]
        assert root.summary != ""

    def test_leaf_nodes_unchanged(self, sample_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(agent=None)
        summarizer.summarize(sample_tree)
        leaf = sample_tree.nodes["doc_t1"]
        assert leaf.summary == ""
        assert leaf.chunk is not None

    def test_extractive_max_sentences(self, sample_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(agent=None)
        summarizer.summarize(sample_tree)
        root = sample_tree.nodes["doc_root"]
        assert len(root.summary.split(". ")) <= 6


class TestSummarizerLLM:
    def test_llm_called_for_internal_nodes(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.model_string = "test-model"
        mock_llm.run.return_value = "LLM summary."

        summarizer = NodeSummarizer(agent=mock_llm)
        summarizer.summarize(sample_tree)

        section1 = sample_tree.nodes["doc_h1"]
        assert section1.summary == "LLM summary."
        assert mock_llm.run.called

    def test_llm_template_rendered(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.model_string = "test-model"
        mock_llm.run.return_value = "Summary."

        summarizer = NodeSummarizer(agent=mock_llm)
        summarizer.summarize(sample_tree)

        call_args = mock_llm.run.call_args
        deps = call_args[0][0]
        prompt = deps.text
        assert "SNR" in prompt or "threshold" in prompt.lower()

    def test_returns_same_tree_object(self, sample_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(agent=None)
        result = summarizer.summarize(sample_tree)
        assert result is sample_tree


class TestSummarizerCache:
    def test_cache_hit_skips_llm(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.model_string = "test-model"
        mock_llm.run.return_value = "Fresh summary."

        mock_cache = MagicMock()
        mock_cache.get.return_value = "Cached summary."

        summarizer = NodeSummarizer(agent=mock_llm, cache_store=mock_cache)
        summarizer.summarize(sample_tree)

        mock_llm.run.assert_not_called()

    def test_cache_miss_calls_llm(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.model_string = "test-model"
        mock_llm.run.return_value = "Fresh summary."

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        summarizer = NodeSummarizer(agent=mock_llm, cache_store=mock_cache)
        summarizer.summarize(sample_tree)

        assert mock_llm.run.called
        assert mock_cache.set.called

    def test_cache_includes_model_in_deps(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.model_string = "my-model"
        mock_llm.run.return_value = "Summary."

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        summarizer = NodeSummarizer(agent=mock_llm, cache_store=mock_cache)
        summarizer.summarize(sample_tree)

        for call in mock_cache.set.call_args_list:
            deps = call[0][1]
            assert deps["model"] == "my-model"


class TestSummarizerStructuredLeaf:
    def test_structured_leaf_content(self) -> None:
        root = ChunkNode(id="r", depth=0)
        section = ChunkNode(id="s", parent_id="r", depth=1, heading_chain=("Data",))
        table_leaf = ChunkNode(
            id="t",
            parent_id="s",
            depth=2,
            heading_chain=("Data",),
            structured=StructuredTable(
                columns=["A", "B"],
                rows=[{"A": "1", "B": "2"}],
                caption="Results table",
            ),
            chunk=_make_chunk("t", "Table: Results table\nA | B\n1 | 2"),
            token_count=10,
        )
        root.children_ids = ["s"]
        section.children_ids = ["t"]

        tree = ChunkTree(
            root_id="r",
            nodes={"r": root, "s": section, "t": table_leaf},
            source_file=Path("d.md"),
            strategy=ChunkStrategy.TWO_LEVEL,
        )

        summarizer = NodeSummarizer(agent=None, skip_single_child=False)
        summarizer.summarize(tree)

        section_node = tree.nodes["s"]
        assert section_node.summary != ""


class TestSummarizerAgenticResume:
    """Regression (p863 ``ITU-T_P_863__03_2018__root``): one agentic session
    that ran out of its request budget abandoned the rest of the tree.  A
    session that exhausts its budget must be resumed for the nodes it left
    unsummarized, and must stop instead of spinning when a round makes no
    progress.
    """

    def test_resumes_after_request_budget_exhaustion(
        self,
        agentic_tree: ChunkTree,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        agent = make_agentic_summarizer_agent(model_string="ollama/gpt-oss:20b")
        rounds: list[list[str]] = []

        def _run(deps: SummarizerDeps) -> None:
            ids = [entry["id"] for entry in deps.nodes]
            rounds.append(ids)
            if len(rounds) == 1:
                deps.results[ids[0]] = f"summary {ids[0]}"
                raise UsageLimitExceeded("The next request would exceed the request_limit of 3.")
            for node_id in ids:
                deps.results[node_id] = f"summary {node_id}"

        monkeypatch.setattr(agent, "run", _run)
        tree = NodeSummarizer(agent=agent).summarize(agentic_tree)

        assert len(rounds) == 2
        assert len(rounds[0]) == 4
        assert rounds[0][0] not in rounds[1]
        assert set(rounds[1]) == set(rounds[0]) - {rounds[0][0]}
        for node_id in rounds[0]:
            assert tree.nodes[node_id].summary == f"summary {node_id}"

    def test_stops_when_a_round_records_nothing(
        self,
        agentic_tree: ChunkTree,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        agent = make_agentic_summarizer_agent(model_string="ollama/gpt-oss:20b")
        rounds: list[list[str]] = []

        def _run(deps: SummarizerDeps) -> None:
            rounds.append([entry["id"] for entry in deps.nodes])
            raise UsageLimitExceeded("The next request would exceed the request_limit of 3.")

        monkeypatch.setattr(agent, "run", _run)
        tree = NodeSummarizer(agent=agent).summarize(agentic_tree)

        assert len(rounds) == 1
        for node_id in rounds[0]:
            assert not tree.nodes[node_id].summary

    def test_single_round_when_the_budget_suffices(
        self,
        agentic_tree: ChunkTree,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        agent = make_agentic_summarizer_agent(model_string="ollama/gpt-oss:20b")
        rounds: list[list[str]] = []

        def _run(deps: SummarizerDeps) -> None:
            rounds.append([entry["id"] for entry in deps.nodes])
            for node_id in rounds[-1]:
                deps.results[node_id] = f"summary {node_id}"

        monkeypatch.setattr(agent, "run", _run)
        tree = NodeSummarizer(agent=agent).summarize(agentic_tree)

        assert len(rounds) == 1
        for node_id in rounds[0]:
            assert tree.nodes[node_id].summary == f"summary {node_id}"


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_tree() -> ChunkTree:
    """Build a 5-node tree: root → section (leaf-a, leaf-b) + section-c."""
    root = ChunkNode(id="doc_root", depth=0)
    section1 = ChunkNode(
        id="doc_h1",
        parent_id="doc_root",
        depth=1,
        heading_chain=("Introduction",),
    )
    leaf_a = ChunkNode(
        id="doc_t1",
        parent_id="doc_h1",
        depth=2,
        heading_chain=("Introduction",),
        chunk=_make_chunk("doc_t1", "The SNR threshold is 20 dB for intelligibility."),
        token_count=10,
    )
    leaf_b = ChunkNode(
        id="doc_t2",
        parent_id="doc_h1",
        depth=2,
        heading_chain=("Introduction",),
        chunk=_make_chunk("doc_t2", "Results show improvement at higher levels."),
        token_count=8,
    )
    section2 = ChunkNode(
        id="doc_h2",
        parent_id="doc_root",
        depth=1,
        heading_chain=("Methodology",),
    )
    leaf_c = ChunkNode(
        id="doc_t3",
        parent_id="doc_h2",
        depth=2,
        heading_chain=("Methodology",),
        chunk=_make_chunk("doc_t3", "We used a standard test setup with calibration."),
        token_count=10,
    )

    root.children_ids = ["doc_h1", "doc_h2"]
    section1.children_ids = ["doc_t1", "doc_t2"]
    section2.children_ids = ["doc_t3"]

    nodes = {n.id: n for n in [root, section1, leaf_a, leaf_b, section2, leaf_c]}
    return ChunkTree(
        root_id="doc_root",
        nodes=nodes,
        source_file=Path("doc.md"),
        strategy=ChunkStrategy.TWO_LEVEL,
    )


@pytest.fixture
def agentic_tree() -> ChunkTree:
    """Root → 3 sections with 2 leaves each: 4 eligible internal nodes."""
    root = ChunkNode(id="r", depth=0)
    nodes: dict[str, ChunkNode] = {"r": root}
    for index in range(3):
        section_id = f"s{index}"
        section = ChunkNode(
            id=section_id,
            parent_id="r",
            depth=1,
            heading_chain=(f"Section {index}",),
        )
        for leaf in range(2):
            leaf_id = f"l{index}{leaf}"
            nodes[leaf_id] = ChunkNode(
                id=leaf_id,
                parent_id=section_id,
                depth=2,
                heading_chain=(f"Section {index}",),
                chunk=_make_chunk(leaf_id, f"body {leaf_id}"),
                token_count=4,
            )
            section.children_ids.append(leaf_id)
        nodes[section_id] = section
        root.children_ids.append(section_id)
    return ChunkTree(
        root_id="r",
        nodes=nodes,
        source_file=Path("doc.md"),
        strategy=ChunkStrategy.TWO_LEVEL,
    )


@pytest.fixture
def single_pass_tree() -> ChunkTree:
    """A SINGLE_PASS tree — should be skipped by summarizer."""
    root = ChunkNode(id="root", depth=0)
    leaf = ChunkNode(
        id="leaf",
        parent_id="root",
        depth=1,
        chunk=_make_chunk("leaf", "text"),
        token_count=1,
    )
    root.children_ids = ["leaf"]
    return ChunkTree(
        root_id="root",
        nodes={"root": root, "leaf": leaf},
        source_file=Path("d.md"),
        strategy=ChunkStrategy.SINGLE_PASS,
    )


@pytest.fixture
def single_child_tree() -> ChunkTree:
    """Tree with single-child internal nodes."""
    root = ChunkNode(id="r", depth=0)
    wrapper = ChunkNode(id="w", parent_id="r", depth=1, heading_chain=("Wrapper",))
    leaf = ChunkNode(
        id="l",
        parent_id="w",
        depth=2,
        heading_chain=("Wrapper",),
        chunk=_make_chunk("l", "content"),
        token_count=2,
    )
    root.children_ids = ["w"]
    wrapper.children_ids = ["l"]
    return ChunkTree(
        root_id="r",
        nodes={"r": root, "w": wrapper, "l": leaf},
        source_file=Path("d.md"),
        strategy=ChunkStrategy.TWO_LEVEL,
    )


def _make_chunk(cid: str, text: str, source: str = "doc.md") -> Chunk:
    """Build a Chunk with required fields."""
    return Chunk(
        id=cid,
        text=text,
        element_types=[ElementType.TEXT],
        source_elements=[],
        source_file=Path(source),
        token_count=max(1, len(text) // 4),
    )
