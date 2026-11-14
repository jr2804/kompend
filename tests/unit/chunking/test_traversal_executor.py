"""Tests for the traversal plan executor in NodeSummarizer (knox-u84v.4).

The executor consumes a ``TraversalPlanner`` (protocol defined in
``kompend.chunking.summarizer``) level by level, top-down, and summarizes
only the planned nodes bottom-up. These tests use stub planners — the real
planner adapter is wired in the pipeline factory and covered there.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from kompend.agents.summarizer import SummarizerAgent
from kompend.agents.types import SummarizerDeps
from kompend.chunking.outline import OutlineNode
from kompend.chunking.summarizer import (
    LevelPlan,
    NodeSummarizer,
    PlanContext,
    PlanEntry,
    PlanMode,
    SummarizerContentPolicy,
    SummarizerSessionPolicy,
    TraversalMode,
    TraversalPlanner,
)
from kompend.core.cache.store import CacheStore
from kompend.core.cancel import KompendCancelled
from kompend.core.config._configs import BudgetConfig
from kompend.core.models import Chunk, ChunkNode, ChunkStrategy, ChunkTree, ElementType
from kompend.core.reporting.handle import set_current_run_id
from kompend.core.store import vault_db_path
from kompend.core.types import ReasonCode, StageOutcome
from kompend.core.utils.exceptions import LLMContextOverflowError


class _StubAgent:
    """Minimal stand-in for the summarizer agent: returns fixed text."""

    model_string = "test-model"

    def __init__(self, summary: str = "Summary.") -> None:
        self.summary = summary
        self.calls: list[SummarizerDeps] = []

    def run(self, deps: SummarizerDeps) -> str:
        self.calls.append(deps)
        for entry in deps.nodes:
            deps.results[entry["id"]] = self.summary
        return self.summary


class _StubPlanner:
    """Return fixed LevelPlans keyed by the level's child-id tuple."""

    def __init__(self, plans: dict[tuple[str, ...], LevelPlan], fail: BaseException | None = None) -> None:
        self._plans = plans
        self.fail = fail
        self.calls: list[tuple[tuple[str, ...], int, PlanContext]] = []

    def plan_level(self, *, outline: Sequence[OutlineNode], depth: int, context: PlanContext) -> LevelPlan | None:
        if self.fail is not None:
            raise self.fail
        key = tuple(entry.node_id for entry in outline)
        self.calls.append((key, depth, context))
        return self._plans.get(key)


class _FakeCache:
    """Dict-backed cache matching the CacheStore get/set surface."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, layer: str, deps: dict[str, str]) -> str | None:
        return self._store.get(json.dumps(deps, sort_keys=True))

    def set(self, layer: str, deps: dict[str, str], value: str) -> None:
        self._store[json.dumps(deps, sort_keys=True)] = value


# ── tests ────────────────────────────────────────────────────────────────


class _DistinctStubAgent(_StubAgent):
    """Stub that gives every node its own summary text, keyed by payload id."""

    def run(self, deps: SummarizerDeps) -> str:
        self.calls.append(deps)
        for entry in deps.nodes:
            deps.results[entry["id"]] = f"summary[{entry['id']}]"
        return ""


class TestPlannedSkip:
    def test_skipped_subtree_is_never_summarized_and_inherits(self, caplog: pytest.LogCaptureFixture) -> None:
        plans = {
            ("s0", "s1", "s2"): _plan(("s0", "s1", "s2"), skip={"s2": "boilerplate"}),
            ("l00", "l01"): _plan(("l00", "l01")),
            ("l10", "l11"): _plan(("l10", "l11")),
        }
        agent = _DistinctStubAgent()
        planner = _StubPlanner(plans)
        tree = _tree()

        with caplog.at_level(logging.DEBUG, logger="kompend.chunking.summarizer"):
            _summarizer(agent, planner).summarize(tree)

        assert "s2" not in _payload_ids(agent)
        assert not any(entry_id.startswith("l2") for entry_id in _payload_ids(agent))
        # The skipped section inherits its nearest summarized ancestor (the
        # root) — its summary names the root, not itself.
        assert tree.nodes["s2"].summary == "summary[r]"
        assert tree.nodes["r"].summary != ""
        # The skip decision is recorded with its reason.
        assert any("Skipping s2 by plan (boilerplate)" in record.message for record in caplog.records)

    def test_summarize_branch_inlines_undecided_content(self) -> None:
        """A SUMMARIZE branch with undecided internal children inlines their leaves."""
        agent = _StubAgent()
        planner = _StubPlanner({("m",): _plan(("m",), focus={"m": "the middle part"})})
        tree = _deep_tree()

        _summarizer(agent, planner).summarize(tree)

        payload = [entry for deps in agent.calls for entry in deps.nodes if entry["id"] == "m"]
        assert payload, "the planned node must be summarized"
        content = payload[0]["content"]
        assert "grandchild one" in content
        assert "grandchild two" in content
        assert "direct leaf" in content
        assert payload[0]["focus"] == "the middle part"
        assert tree.nodes["m"].summary != ""
        # The undecided internal child inherits the branch summary.
        assert tree.nodes["b"].summary == tree.nodes["m"].summary

    def test_focus_reaches_the_payload_with_off_path_window_count(self) -> None:
        plans = {
            ("s0", "s1", "s2"): _plan(("s0", "s1", "s2"), focus={"s0": "noise floors"}),
            ("l00", "l01"): _plan(("l00", "l01")),
            ("l10", "l11"): _plan(("l10", "l11")),
            ("l20", "l21"): _plan(("l20", "l21")),
        }
        planned_agent = _StubAgent()
        planned_tree = _tree()
        _summarizer(planned_agent, _StubPlanner(plans)).summarize(planned_tree)

        off_agent = _StubAgent()
        off_tree = _tree()
        _summarizer(off_agent, planner=None, auto=False).summarize(off_tree)

        planned_windows = sum(len(deps.nodes) for deps in planned_agent.calls)
        off_windows = sum(len(deps.nodes) for deps in off_agent.calls)
        assert planned_windows == off_windows
        s0_entries = [entry for deps in planned_agent.calls for entry in deps.nodes if entry["id"] == "s0"]
        assert s0_entries, "the planned section must reach the prompt"
        assert all(entry["focus"] == "noise floors" for entry in s0_entries)
        # Unplanned-path payloads never carry a focus key.
        assert not any("focus" in entry for deps in off_agent.calls for entry in deps.nodes)


class TestPlannerFailure:
    def test_raising_planner_falls_back_to_the_full_tree(self, caplog: pytest.LogCaptureFixture) -> None:
        planner = _StubPlanner({}, fail=RuntimeError("decision model down"))
        agent = _StubAgent("Planned summary.")
        tree = _tree()

        with caplog.at_level(logging.WARNING, logger="kompend.chunking.summarizer"):
            _summarizer(agent, planner).summarize(tree)

        assert any("Traversal planning failed for tree r" in record.message for record in caplog.records)
        assert any("decision model down" in record.message for record in caplog.records)
        # Full-tree fallback: every eligible node was summarized.
        assert all(tree.nodes[node_id].summary == "Planned summary." for node_id in ("r", "s0", "s1", "s2"))
        assert len(agent.calls) >= 1

    def test_planner_declining_a_level_falls_back_too(self, caplog: pytest.LogCaptureFixture) -> None:
        planner = _StubPlanner({("s0", "s1", "s2"): _plan(("s0", "s1", "s2"))})  # deeper levels missing → None
        agent = _StubAgent()
        tree = _tree()

        with caplog.at_level(logging.WARNING, logger="kompend.chunking.summarizer"):
            _summarizer(agent, planner).summarize(tree)

        assert any("Traversal planning failed for tree r" in record.message for record in caplog.records)
        assert all(tree.nodes[node_id].summary for node_id in ("r", "s0", "s1", "s2"))

    def test_no_planner_with_auto_matches_the_off_path(self, caplog: pytest.LogCaptureFixture) -> None:
        agent = _StubAgent("Auto summary.")
        tree = _tree()

        with caplog.at_level(logging.WARNING, logger="kompend.chunking.summarizer"):
            _summarizer(agent, planner=None).summarize(tree)

        assert not any("Traversal planning failed" in record.message for record in caplog.records)
        assert all(tree.nodes[node_id].summary == "Auto summary." for node_id in ("r", "s0", "s1", "s2"))

    def test_cancellation_from_the_planner_propagates(self) -> None:
        planner = _StubPlanner({}, fail=KompendCancelled("run cancelled"))
        tree = _tree()

        with pytest.raises(KompendCancelled):
            _summarizer(_StubAgent(), planner).summarize(tree)

    def test_cancellation_from_the_agent_propagates(self) -> None:
        plans = {
            ("s0", "s1", "s2"): _plan(("s0", "s1", "s2")),
            ("l00", "l01"): _plan(("l00", "l01")),
            ("l10", "l11"): _plan(("l10", "l11")),
            ("l20", "l21"): _plan(("l20", "l21")),
        }
        agent = _StubAgent()
        agent.run = lambda deps: (_ for _ in ()).throw(KompendCancelled("run cancelled"))  # type: ignore[method-assign]
        tree = _tree()

        with pytest.raises(KompendCancelled):
            _summarizer(agent, _StubPlanner(plans)).summarize(tree)


class TestCacheReuse:
    def test_rerun_with_the_same_plan_makes_no_llm_calls(self) -> None:
        cache = _FakeCache()
        plans = {
            ("s0", "s1", "s2"): _plan(("s0", "s1", "s2"), skip={"s2": "boilerplate"}),
            ("l00", "l01"): _plan(("l00", "l01")),
            ("l10", "l11"): _plan(("l10", "l11")),
        }
        first = _StubAgent("First summary.")
        _summarizer(first, _StubPlanner(dict(plans)), cache=cache).summarize(_tree())

        second = _StubAgent("Second summary.")
        second_tree = _tree()
        _summarizer(second, _StubPlanner(dict(plans)), cache=cache).summarize(second_tree)

        assert second.calls, "the root must re-aggregate over cached sections"
        second_ids = [entry["id"] for deps in second.calls for entry in deps.nodes]
        assert all(entry_id == "r" or entry_id.startswith("r#") for entry_id in second_ids)
        assert second.calls[0].nodes[0]["content"] == "First summary.\n\nFirst summary."
        # Section nodes come back from the cache…
        assert second_tree.nodes["s0"].summary == "First summary."
        assert second_tree.nodes["s1"].summary == "First summary."
        # …the root re-summarizes over the cached child summaries…
        assert second_tree.nodes["r"].summary == "Second summary."
        # …and the skipped node still inherits instead of summarizing.
        assert second_tree.nodes["s2"].summary == second_tree.nodes["r"].summary


class TestPlannedMetrics:
    def test_metrics_line_counts_everything(self, caplog: pytest.LogCaptureFixture) -> None:
        plans = {
            ("s0", "s1", "s2"): _plan(("s0", "s1", "s2"), skip={"s2": "boilerplate"}),
            ("l00", "l01"): _plan(("l00", "l01")),
            ("l10", "l11"): _plan(("l10", "l11")),
        }
        agent = _StubAgent()
        tree = _tree()

        with caplog.at_level(logging.INFO, logger="kompend.chunking.summarizer"):
            _summarizer(agent, _StubPlanner(plans)).summarize(tree)

        infos = [record for record in caplog.records if record.levelno == logging.INFO]
        assert infos, "the planned path must emit exactly one metrics line"
        message = infos[0].getMessage()
        # planned=3 counts eligible work nodes (root + s0 + s1); expanded=6
        # counts every EXPAND entry incl. the four planned leaves.
        assert message.startswith("tree=r eligible=4 planned=3 expanded=6 skipped=1 sessions=")
        assert "windows=3" in message
        assert "planner_calls=3" in message

    def test_off_path_emits_no_metrics_line(self, caplog: pytest.LogCaptureFixture) -> None:
        tree = _tree()

        with caplog.at_level(logging.INFO, logger="kompend.chunking.summarizer"):
            _summarizer(_StubAgent(), planner=None, auto=False).summarize(tree)

        assert not [record for record in caplog.records if record.levelno == logging.INFO]


class TestPlannedOversizeSplit:
    def test_oversize_split_fires_in_a_planned_session(self, caplog: pytest.LogCaptureFixture) -> None:
        plans = {
            ("s0", "s1", "s2"): _plan(("s0", "s1", "s2")),
            ("l00", "l01"): _plan(("l00", "l01")),
            ("l10", "l11"): _plan(("l10", "l11")),
            ("l20", "l21"): _plan(("l20", "l21")),
        }
        agent = _StubAgent()
        agent.run = lambda deps: (_ for _ in ()).throw(LLMContextOverflowError("request (20000 tokens) exceeds the available context size (16384 tokens)"))  # type: ignore[method-assign]
        tree = _tree()
        policy = SummarizerSessionPolicy(batch_size=4, stall_retries=0, payload_chars=100_000)

        with caplog.at_level(logging.WARNING, logger="kompend.chunking.summarizer"):
            _summarizer(agent, _StubPlanner(plans), session_policy=policy).summarize(tree)

        messages = [record.message for record in caplog.records]
        assert any("splitting without a stall retry" in message for message in messages)
        assert any("exceeds the provider limit even at the split floor" in message for message in messages)
        assert any("unsummarized after" in message for message in messages)


class TestTraversalGate:
    def test_below_min_nodes_the_planner_is_never_consulted(self) -> None:
        planner = _StubPlanner({("s0", "s1", "s2"): _plan(("s0", "s1", "s2"))})
        agent = _StubAgent("Off-path summary.")
        tree = _tree()
        gated = replace(_auto_policy(), traversal_min_nodes=5)

        _summarizer(agent, planner, policy=gated).summarize(tree)

        assert planner.calls == []
        # The run silently stays on today's path instead.
        assert all(tree.nodes[node_id].summary == "Off-path summary." for node_id in ("r", "s0", "s1", "s2"))

    def test_at_min_nodes_the_planner_runs(self) -> None:
        plans = {
            ("s0", "s1", "s2"): _plan(("s0", "s1", "s2")),
            ("l00", "l01"): _plan(("l00", "l01")),
            ("l10", "l11"): _plan(("l10", "l11")),
            ("l20", "l21"): _plan(("l20", "l21")),
        }
        planner = _StubPlanner(plans)
        gated = replace(_auto_policy(), traversal_min_nodes=4)

        _summarizer(_StubAgent(), planner, policy=gated).summarize(_tree())

        # All-EXPAND plan: the root level plus every section's leaf level.
        assert len(planner.calls) == 4


class TestPlanContextPassing:
    def test_executor_hands_tree_hash_and_title_to_the_planner(self) -> None:
        plans = {
            ("s0", "s1", "s2"): _plan(("s0", "s1", "s2")),
            ("l00", "l01"): _plan(("l00", "l01")),
            ("l10", "l11"): _plan(("l10", "l11")),
            ("l20", "l21"): _plan(("l20", "l21")),
        }
        planner = _StubPlanner(plans)

        _summarizer(_StubAgent(), planner).summarize(_tree())

        assert planner.calls, "the planner must be consulted"
        contexts = [context for _, _, context in planner.calls]
        assert all(context.title == "doc" for context in contexts)
        assert len({context.tree_hash for context in contexts}) == 1
        assert all(len(context.tree_hash) == 16 for context in contexts)


class TestPlannedLedgerRow:
    def test_planned_run_records_one_traversal_row(self, tmp_path: Path) -> None:
        plans = {
            ("s0", "s1", "s2"): _plan(("s0", "s1", "s2"), skip={"s2": "boilerplate"}),
            ("l00", "l01"): _plan(("l00", "l01")),
            ("l10", "l11"): _plan(("l10", "l11")),
        }
        db = vault_db_path(tmp_path)
        set_current_run_id("ledger-run")
        try:
            _summarizer(_StubAgent(), _StubPlanner(plans), ledger_db_path=db).summarize(_tree())
        finally:
            set_current_run_id(None)

        rows = _ledger_rows(db)
        assert len(rows) == 1
        source, stage, outcome, reason_code, detail = rows[0]
        assert source == "doc.md"
        assert stage == "summarize"
        assert outcome == StageOutcome.OK.value
        assert reason_code == ReasonCode.TRAVERSAL.value
        assert detail == "planned=3 skipped=1 expanded=6 windows=3"

    def test_failing_planner_records_a_failed_row(self, tmp_path: Path) -> None:
        db = vault_db_path(tmp_path)
        set_current_run_id("ledger-run")
        try:
            _summarizer(_StubAgent(), _StubPlanner({}, fail=RuntimeError("down")), ledger_db_path=db).summarize(_tree())
        finally:
            set_current_run_id(None)

        rows = _ledger_rows(db)
        assert len(rows) == 1
        _, _, outcome, _, detail = rows[0]
        assert outcome == StageOutcome.FAILED.value
        assert detail == "planned=0 skipped=0 expanded=0 windows=0"

    def test_no_ledger_path_records_nothing(self, tmp_path: Path) -> None:
        plans = {
            ("s0", "s1", "s2"): _plan(("s0", "s1", "s2")),
            ("l00", "l01"): _plan(("l00", "l01")),
            ("l10", "l11"): _plan(("l10", "l11")),
            ("l20", "l21"): _plan(("l20", "l21")),
        }

        _summarizer(_StubAgent(), _StubPlanner(plans)).summarize(_tree())

        assert not vault_db_path(tmp_path).exists()


def _agent(summary: str = "Summary.") -> SummarizerAgent:
    return cast("SummarizerAgent", _StubAgent(summary))


def _plan(ids: tuple[str, ...], *, mode: PlanMode = PlanMode.EXPAND, focus: dict[str, str] | None = None, skip: dict[str, str] | None = None) -> LevelPlan:
    focus = focus or {}
    skip = skip or {}
    return LevelPlan(
        entries=tuple(
            PlanEntry(
                node_id=node_id,
                mode=PlanMode.SKIP if node_id in skip else PlanMode.SUMMARIZE if node_id in focus else mode,
                focus=focus.get(node_id),
                reason=skip.get(node_id),
            )
            for node_id in ids
        )
    )


def _tree() -> ChunkTree:
    """Root → 3 sections × 2 leaves: 4 eligible internal nodes."""
    root = ChunkNode(id="r", depth=0)
    nodes: dict[str, ChunkNode] = {"r": root}
    for index in range(3):
        section_id = f"s{index}"
        section = ChunkNode(id=section_id, parent_id="r", depth=1, heading_chain=(f"Section {index}",))
        for leaf in range(2):
            leaf_id = f"l{index}{leaf}"
            nodes[leaf_id] = ChunkNode(
                id=leaf_id,
                parent_id=section_id,
                depth=2,
                heading_chain=(f"Section {index}",),
                chunk=_chunk(leaf_id, f"body {leaf_id}"),
                token_count=4,
            )
            section.children_ids.append(leaf_id)
        nodes[section_id] = section
        root.children_ids.append(section_id)
    return ChunkTree(root_id="r", nodes=nodes, source_file=Path("doc.md"), strategy=ChunkStrategy.TWO_LEVEL)


def _deep_tree() -> ChunkTree:
    """Root → mid → (branch → 2 leaves, leaf): 3 eligible internal nodes.

    The SUMMARIZE-decided ``m`` carries an undecided internal child ``b``,
    so its payload must inline ``b``'s leaf content.
    """
    root = ChunkNode(id="r", depth=0)
    mid = ChunkNode(id="m", parent_id="r", depth=1, heading_chain=("Mid",))
    branch = ChunkNode(id="b", parent_id="m", depth=2, heading_chain=("Mid", "Branch"))
    leaf_b1 = ChunkNode(id="b1", parent_id="b", depth=3, heading_chain=("Mid", "Branch"), chunk=_chunk("b1", "grandchild one"), token_count=4)
    leaf_b2 = ChunkNode(id="b2", parent_id="b", depth=3, heading_chain=("Mid", "Branch"), chunk=_chunk("b2", "grandchild two"), token_count=4)
    leaf_c = ChunkNode(id="c", parent_id="m", depth=2, heading_chain=("Mid",), chunk=_chunk("c", "direct leaf"), token_count=4)
    branch.children_ids = ["b1", "b2"]
    mid.children_ids = ["b", "c"]
    root.children_ids = ["m"]
    nodes = {"r": root, "m": mid, "b": branch, "b1": leaf_b1, "b2": leaf_b2, "c": leaf_c}
    return ChunkTree(root_id="r", nodes=nodes, source_file=Path("doc.md"), strategy=ChunkStrategy.TWO_LEVEL)


def _chunk(cid: str, text: str, source: str = "doc.md") -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        element_types=[ElementType.TEXT],
        source_elements=[],
        source_file=Path(source),
        token_count=max(1, len(text) // 4),
    )


def _summarizer(
    agent: Any,
    planner: Any = None,
    *,
    auto: bool = True,
    cache: Any = None,
    session_policy: SummarizerSessionPolicy | None = None,
    ledger_db_path: Path | None = None,
    policy: SummarizerContentPolicy | None = None,
) -> NodeSummarizer:
    """Build a NodeSummarizer around a test double.

    ``agent``/``planner``/``cache`` are test doubles duck-typed by the
    executor; the real types are ``SummarizerAgent``, ``TraversalPlanner``
    and ``CacheStore``.
    """
    content = policy or (_auto_policy() if auto else SummarizerContentPolicy.from_config(BudgetConfig()))
    return NodeSummarizer(
        cast("SummarizerAgent", agent),
        cache_store=cast("CacheStore | None", cache),
        content_policy=content,
        session_policy=session_policy,
        planner=cast("TraversalPlanner | None", planner),
        ledger_db_path=ledger_db_path,
    )


def _auto_policy() -> SummarizerContentPolicy:
    # traversal_min_nodes=0: the fixtures hold 4–6 eligible nodes, far below
    # the configured gate (50) — the gate itself is covered by its own test.
    return replace(SummarizerContentPolicy.from_config(BudgetConfig()), traversal=TraversalMode.AUTO, traversal_min_nodes=0)


def _payload_ids(agent: _StubAgent) -> set[str]:
    return {entry["id"] for deps in agent.calls for entry in deps.nodes}


def _ledger_rows(db: Path) -> list[tuple[str, str, str, str, str]]:
    conn = sqlite3.connect(db)
    try:
        return list(conn.execute("SELECT source, stage, outcome, reason_code, detail FROM stages ORDER BY id"))
    finally:
        conn.close()
