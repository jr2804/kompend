"""Tests for NodeSummarizer (summarizer.py)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded

from kompend.agents.summarizer import SummarizerAgent
from kompend.agents.summarizer_agentic import make_agentic_summarizer_agent
from kompend.agents.types import SummarizerDeps
from kompend.chunking.summarizer import (
    NodeSummarizer,
    SummarizerContentPolicy,
    SummarizerSessionPolicy,
)
from kompend.core.config._configs import BudgetConfig
from kompend.core.models import (
    Chunk,
    ChunkNode,
    ChunkStrategy,
    ChunkTree,
    ElementType,
    StructuredTable,
)


class _StubAgent:
    """Minimal stand-in for the summarizer agent: returns fixed text."""

    model_string = "test-model"

    def __init__(self, summary: str = "Summary.") -> None:
        self.summary = summary
        self.calls: list = []

    def run(self, deps: object) -> str:
        self.calls.append(deps)
        return self.summary


# ── tests ────────────────────────────────────────────────────────────────


class TestSummarizerSkip:
    def test_single_pass_skipped(self, single_pass_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(agent=_agent(), content_policy=_policy())
        result = summarizer.summarize(single_pass_tree)
        root = result.nodes["root"]
        assert root.summary == ""

    def test_single_child_skipped_by_default(self, single_child_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(agent=_agent(), content_policy=_policy(skip_single_child=True))
        summarizer.summarize(single_child_tree)
        wrapper = single_child_tree.nodes["w"]
        assert wrapper.summary == ""

    def test_single_child_summarized_when_disabled(self, single_child_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(agent=_agent(), content_policy=_policy(skip_single_child=False))
        summarizer.summarize(single_child_tree)
        wrapper = single_child_tree.nodes["w"]
        assert wrapper.summary != ""


class TestSummarizerRequiresAnAgent:
    def test_no_agent_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires an agent"):
            NodeSummarizer(agent=None, content_policy=_policy())

    def test_llm_failure_propagates(self, sample_tree: ChunkTree) -> None:
        failing = MagicMock()
        failing.model_string = "test-model"
        failing.run.side_effect = RuntimeError("model exploded")

        summarizer = NodeSummarizer(agent=failing, content_policy=_policy())

        with pytest.raises(RuntimeError, match="model exploded"):
            summarizer.summarize(sample_tree)


class TestSummarizerLLM:
    def test_llm_called_for_internal_nodes(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.model_string = "test-model"
        mock_llm.run.return_value = "LLM summary."

        summarizer = NodeSummarizer(agent=mock_llm, content_policy=_policy())
        summarizer.summarize(sample_tree)

        section1 = sample_tree.nodes["doc_h1"]
        assert section1.summary == "LLM summary."
        assert mock_llm.run.called

    def test_llm_template_rendered(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.model_string = "test-model"
        mock_llm.run.return_value = "Summary."

        summarizer = NodeSummarizer(agent=mock_llm, content_policy=_policy())
        summarizer.summarize(sample_tree)

        call_args = mock_llm.run.call_args
        deps = call_args[0][0]
        prompt = deps.text
        assert "SNR" in prompt or "threshold" in prompt.lower()

    def test_returns_same_tree_object(self, sample_tree: ChunkTree) -> None:
        summarizer = NodeSummarizer(agent=_agent(), content_policy=_policy())
        result = summarizer.summarize(sample_tree)
        assert result is sample_tree

    def test_target_length_reaches_the_prompt(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.model_string = "test-model"
        mock_llm.run.return_value = "Summary."

        policy = replace(_policy(), target_chars=321)
        summarizer = NodeSummarizer(agent=mock_llm, content_policy=policy)
        summarizer.summarize(sample_tree)

        prompt = mock_llm.run.call_args[0][0].text
        assert "321 characters" in prompt


class TestSummarizerCache:
    def test_cache_hit_skips_llm(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.model_string = "test-model"
        mock_llm.run.return_value = "Fresh summary."

        mock_cache = MagicMock()
        mock_cache.get.return_value = "Cached summary."

        summarizer = NodeSummarizer(agent=mock_llm, cache_store=mock_cache, content_policy=_policy())
        summarizer.summarize(sample_tree)

        mock_llm.run.assert_not_called()

    def test_cache_miss_calls_llm(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.model_string = "test-model"
        mock_llm.run.return_value = "Fresh summary."

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        summarizer = NodeSummarizer(agent=mock_llm, cache_store=mock_cache, content_policy=_policy())
        summarizer.summarize(sample_tree)

        assert mock_llm.run.called
        assert mock_cache.set.called

    def test_cache_includes_model_in_deps(self, sample_tree: ChunkTree) -> None:
        mock_llm = MagicMock()
        mock_llm.model_string = "my-model"
        mock_llm.run.return_value = "Summary."

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        summarizer = NodeSummarizer(agent=mock_llm, cache_store=mock_cache, content_policy=_policy())
        summarizer.summarize(sample_tree)

        for call in mock_cache.set.call_args_list:
            deps = call[0][1]
            assert deps["model"] == "my-model"

    def test_cache_deps_skip_model_when_ignoring(self, sample_tree: ChunkTree) -> None:
        """knox-wflz: ``cache_ignore_model`` reuses summaries across a model switch."""
        mock_llm = MagicMock()
        mock_llm.model_string = "my-model"
        mock_llm.run.return_value = "Summary."

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        summarizer = NodeSummarizer(agent=mock_llm, cache_store=mock_cache, content_policy=_policy(), cache_ignore_model=True)
        summarizer.summarize(sample_tree)

        assert mock_llm.run.called
        for call in mock_cache.set.call_args_list:
            deps = call[0][1]
            assert "model" not in deps
            assert deps["content_hash"]

    def test_cache_deps_exclude_the_agent_notebook(self, sample_tree: ChunkTree) -> None:
        """The agent notebook is not a cache input: the agent writes it mid-run."""
        mock_llm = MagicMock()
        mock_llm.model_string = "my-model"
        mock_llm.run.return_value = "Summary."

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        summarizer = NodeSummarizer(agent=mock_llm, cache_store=mock_cache, content_policy=_policy())
        summarizer.summarize(sample_tree)

        for call in mock_cache.set.call_args_list:
            deps = call[0][1]
            assert "memory" not in deps
            assert set(deps) <= {"content_hash", "heading_chain", "model"}


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

        summarizer = NodeSummarizer(agent=_agent(), content_policy=_policy(skip_single_child=False))
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
        tree = NodeSummarizer(agent=agent, content_policy=_policy()).summarize(agentic_tree)

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
        tree = NodeSummarizer(agent=agent, content_policy=_policy()).summarize(agentic_tree)

        assert len(rounds) == 1
        for node_id in rounds[0]:
            assert not tree.nodes[node_id].summary

    def test_junk_ids_do_not_count_as_progress(
        self,
        agentic_tree: ChunkTree,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A model that mangles ids must not freeze the payload and re-run forever."""
        agent = make_agentic_summarizer_agent(model_string="ollama/gpt-oss:20b")
        rounds: list[list[str]] = []

        def _run(deps: SummarizerDeps) -> None:
            rounds.append([entry["id"] for entry in deps.nodes])
            deps.results[f"not-a-node-{len(rounds)}"] = "junk"

        monkeypatch.setattr(agent, "run", _run)
        tree = NodeSummarizer(agent=agent, content_policy=_policy()).summarize(agentic_tree)

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
        tree = NodeSummarizer(agent=agent, content_policy=_policy()).summarize(agentic_tree)

        assert len(rounds) == 1
        for node_id in rounds[0]:
            assert tree.nodes[node_id].summary == f"summary {node_id}"


class TestSummarizerAgenticBatching:
    """Batch size and stall retries come from ``SummarizerSessionPolicy``.

    A session that records nothing must not abandon the whole tree: it only
    strands its own batch, and that batch is retried per the policy.
    """

    def test_sessions_are_bounded_by_the_policy(
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
        policy = SummarizerSessionPolicy(batch_size=2, stall_retries=0, payload_chars=100_000)
        tree = NodeSummarizer(agent=agent, content_policy=_policy(), session_policy=policy).summarize(agentic_tree)

        assert [len(round) for round in rounds] == [2, 2]
        for node_id in ("s0", "s1", "s2", "r"):
            assert tree.nodes[node_id].summary == f"summary {node_id}"

    def test_stalled_batch_does_not_abandon_the_rest(
        self,
        agentic_tree: ChunkTree,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        agent = make_agentic_summarizer_agent(model_string="ollama/gpt-oss:20b")
        rounds: list[list[str]] = []

        def _run(deps: SummarizerDeps) -> None:
            rounds.append([entry["id"] for entry in deps.nodes])
            if len(rounds) == 1:
                return  # model answers in prose: no summarize_node() calls
            for node_id in rounds[-1]:
                deps.results[node_id] = f"summary {node_id}"

        monkeypatch.setattr(agent, "run", _run)
        policy = SummarizerSessionPolicy(batch_size=2, stall_retries=0, payload_chars=100_000)
        tree = NodeSummarizer(agent=agent, content_policy=_policy(), session_policy=policy).summarize(agentic_tree)

        assert len(rounds) == 2
        for node_id in rounds[0]:
            assert not tree.nodes[node_id].summary
        for node_id in rounds[1]:
            assert tree.nodes[node_id].summary == f"summary {node_id}"

    def test_stalled_batch_is_retried_per_policy(
        self,
        agentic_tree: ChunkTree,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        agent = make_agentic_summarizer_agent(model_string="ollama/gpt-oss:20b")
        rounds: list[list[str]] = []

        def _run(deps: SummarizerDeps) -> None:
            rounds.append([entry["id"] for entry in deps.nodes])
            if len(rounds) == 1:
                return  # first attempt records nothing
            for node_id in rounds[-1]:
                deps.results[node_id] = f"summary {node_id}"

        monkeypatch.setattr(agent, "run", _run)
        policy = SummarizerSessionPolicy(batch_size=2, stall_retries=1, payload_chars=100_000)
        tree = NodeSummarizer(agent=agent, content_policy=_policy(), session_policy=policy).summarize(agentic_tree)

        assert [len(round) for round in rounds] == [2, 2, 2]
        for node_id in ("s0", "s1", "s2", "r"):
            assert tree.nodes[node_id].summary == f"summary {node_id}"


class TestSummarizerWindowing:
    """Content longer than the per-call budget is split — never truncated."""

    def test_long_content_is_summarized_window_by_window(self) -> None:
        mock_llm = MagicMock()
        mock_llm.model_string = "test-model"
        mock_llm.run.side_effect = lambda deps: f"summary of {len(deps.text)} chars"

        summarizer = NodeSummarizer(agent=mock_llm, content_policy=_policy(max_content_chars=100))
        tree = summarizer.summarize(_long_tree())

        prompts = [call[0][0].text for call in mock_llm.run.call_args_list]
        assert len(prompts) > 1
        # The tail of the content reaches the model — nothing is dropped.
        assert any("TAIL-MARKER" in prompt for prompt in prompts)
        assert tree.nodes["s"].summary != ""

    def test_agentic_payload_carries_every_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = make_agentic_summarizer_agent("test-model")
        seen: list[list[tuple[str, str]]] = []

        def _run(deps: SummarizerDeps) -> None:
            seen.append([(entry["id"], entry["content"]) for entry in deps.nodes])
            for entry in deps.nodes:
                deps.results[entry["id"]] = f"summary {entry['id']}"

        monkeypatch.setattr(agent, "run", _run)
        policy = SummarizerSessionPolicy(batch_size=10, stall_retries=1, payload_chars=100_000)
        tree = NodeSummarizer(agent=agent, content_policy=_policy(), session_policy=policy).summarize(_long_tree())

        entries = [entry for session in seen for entry in session]
        ids = [entry_id for entry_id, _ in entries]
        assert any(entry_id.startswith("s#w") for entry_id in ids)
        assert any("TAIL-MARKER" in content for _, content in entries)
        assert tree.nodes["s"].summary != ""

    def test_agentic_payload_is_bounded_by_the_session_payload_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A payload never exceeds its character budget; the rest is asked later."""
        agent = make_agentic_summarizer_agent("test-model")
        seen: list[list[str]] = []

        def _run(deps: SummarizerDeps) -> None:
            entries = list(deps.nodes)
            seen.append([entry["id"] for entry in entries])
            assert sum(len(entry["content"]) for entry in entries) <= 120
            for entry in entries:
                deps.results[entry["id"]] = f"summary {entry['id']}"

        monkeypatch.setattr(agent, "run", _run)
        policy = SummarizerSessionPolicy(batch_size=10, stall_retries=1, payload_chars=120)
        tree = NodeSummarizer(agent=agent, content_policy=_policy(window_chars=50), session_policy=policy).summarize(_long_tree())

        # The long node needs several windows, so it takes several sessions -
        # and every window is eventually summarised.
        assert len(seen) > 1
        assert tree.nodes["s"].summary != ""

    def test_agentic_window_size_comes_from_the_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = make_agentic_summarizer_agent("test-model")
        seen: list[list[str]] = []

        def _run(deps: SummarizerDeps) -> None:
            seen.append([entry["id"] for entry in deps.nodes])
            for entry in deps.nodes:
                deps.results[entry["id"]] = f"summary {entry['id']}"

        monkeypatch.setattr(agent, "run", _run)
        policy = SummarizerSessionPolicy(batch_size=100, stall_retries=1, payload_chars=100_000)
        NodeSummarizer(agent=agent, content_policy=_policy(window_chars=40), session_policy=policy).summarize(_long_tree())
        small_window_entries = [entry_id for session in seen for entry_id in session]

        seen.clear()
        NodeSummarizer(agent=agent, content_policy=_policy(window_chars=4000), session_policy=policy).summarize(_long_tree())
        large_window_entries = [entry_id for session in seen for entry_id in session]

        # A bigger window means fewer summarize calls — never fewer windows of
        # content.
        assert len(small_window_entries) > len(large_window_entries)


def _agent(summary: str = "Summary.") -> SummarizerAgent:
    return cast(SummarizerAgent, _StubAgent(summary))


def _policy(
    max_content_chars: int | None = None,
    window_chars: int | None = None,
    skip_single_child: bool | None = None,
) -> SummarizerContentPolicy:
    """Content policy from the config defaults, with optional overrides."""
    policy = SummarizerContentPolicy.from_config(BudgetConfig())
    if max_content_chars is not None:
        policy = replace(policy, max_content_chars=max_content_chars)
    if window_chars is not None:
        policy = replace(policy, window_chars=window_chars)
    if skip_single_child is not None:
        policy = replace(policy, skip_single_child=skip_single_child)
    return policy


# ── fixtures ─────────────────────────────────────────────────────────────


def _long_tree() -> ChunkTree:
    """Root → one section with four long leaves (the last one is the tail)."""
    root = ChunkNode(id="r", depth=0)
    section = ChunkNode(id="s", parent_id="r", depth=1, heading_chain=("Intro",))
    leaves: list[ChunkNode] = []
    for index in range(4):
        text = f"line {index} " + "x" * 700 + (". TAIL-MARKER" if index == 3 else "")
        leaves.append(
            ChunkNode(
                id=f"l{index}",
                parent_id="s",
                depth=2,
                heading_chain=("Intro",),
                chunk=_make_chunk(f"l{index}", text),
                token_count=30,
            )
        )
    section.children_ids = [leaf.id for leaf in leaves]
    root.children_ids = ["s"]
    nodes = {node.id: node for node in [root, section, *leaves]}
    return ChunkTree(
        root_id="r",
        nodes=nodes,
        source_file=Path("doc.md"),
        strategy=ChunkStrategy.TWO_LEVEL,
    )


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
