"""Tests for the factory's traversal-planner adapter (knox-u84v.4).

The adapter bridges the executor's ``TraversalPlanner`` protocol to the
decision-model planner and caches every plan in the ``summarizer_plan``
layer. The decision model is stubbed at the module boundary (the factory
imports ``decide_traversal_plan`` into its own namespace); the cache is a
real ``CacheStore`` on ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kompend.agents.traversal import PlanEntry as AgentPlanEntry
from kompend.agents.traversal import PlanMode as AgentPlanMode
from kompend.agents.traversal import TraversalPlan
from kompend.chunking.outline import OutlineNode
from kompend.chunking.summarizer import PlanContext, PlanMode
from kompend.core.cache.store import (
    SUMMARIZER_PLAN_LAYER,
    CacheStore,
    summarizer_plan_deps,
    summarizer_policy_key,
    summarizer_task_hash,
)
from kompend.core.config._configs import DecisionConfig
from kompend.core.pipeline.stages import factory
from kompend.core.pipeline.stages.factory import _TraversalPlannerAdapter


def _node(node_id: str, *, heading: tuple[str, ...] = ("Doc", "Section"), chars: int = 200) -> OutlineNode:
    return OutlineNode(
        node_id=node_id,
        heading_chain=heading,
        depth=1,
        child_count=0,
        payload_chars=chars,
        subtree_chars=chars,
        token_estimate=chars // 4,
        kinds=("chunk",),
        preview="preview text",
    )


def _plan_entry(node_id: str, *, mode: AgentPlanMode = AgentPlanMode.EXPAND, focus: str | None = None) -> AgentPlanEntry:
    return AgentPlanEntry(node_id=node_id, mode=mode, focus=focus, reason=None)


def _adapter(cache: CacheStore | None, task: str = "Standing task.") -> _TraversalPlannerAdapter:
    return _TraversalPlannerAdapter(
        decision=DecisionConfig(),
        model="test/decision-model",
        budget_nodes=200,
        budget_windows=2000,
        budget_chars=10_000,
        cache_store=cache,
        include_model=True,
        task=task,
    )


def _stub_decide(entries: tuple[AgentPlanEntry, ...], calls: list[dict[str, Any]]) -> Any:
    def fake(config: object, **kwargs: Any) -> TraversalPlan | None:
        calls.append(kwargs)
        return TraversalPlan(entries=list(entries))

    return fake


def test_plan_level_converts_and_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(factory, "decide_traversal_plan", _stub_decide((_plan_entry("s0", mode=AgentPlanMode.SUMMARIZE, focus="noise floors"),), calls))
    cache = CacheStore(tmp_path)
    context = PlanContext(tree_hash="a" * 16, title="doc")

    level = _adapter(cache).plan_level(outline=[_node("s0")], depth=1, context=context)

    assert level is not None
    assert [entry.mode for entry in level.entries] == [PlanMode.SUMMARIZE]
    assert level.entries[0].focus == "noise floors"
    # The decision model was asked once, with the executor's context.
    assert len(calls) == 1
    assert calls[0]["title"] == "doc"
    assert calls[0]["outline"] == [_node("s0")]
    # The plan landed in the cache layer under the adapter's exact deps.
    assert cache.has(SUMMARIZER_PLAN_LAYER, _deps_for("a" * 16))


def _deps_for(tree_hash: str) -> dict[str, str]:
    return summarizer_plan_deps(
        tree_hash=tree_hash,
        task_hash=summarizer_task_hash("Standing task.", ""),
        model="test/decision-model",
        policy=summarizer_policy_key("auto", max_nodes=200, max_windows=2000),
        include_model=True,
    )


def test_second_call_with_the_same_tree_hits_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(factory, "decide_traversal_plan", _stub_decide((_plan_entry("s0"),), calls))
    cache = CacheStore(tmp_path)
    adapter = _adapter(cache)
    context = PlanContext(tree_hash="a" * 16, title="doc")

    first = adapter.plan_level(outline=[_node("s0")], depth=1, context=context)
    assert first is not None
    second = adapter.plan_level(outline=[_node("s0")], depth=1, context=context)
    assert second is not None

    assert [entry.node_id for entry in second.entries] == [entry.node_id for entry in first.entries]
    assert len(calls) == 1, "the second call must be served from the cache"


def test_declining_planner_caches_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def declining(config: object, **kwargs: Any) -> TraversalPlan | None:
        return None

    monkeypatch.setattr(factory, "decide_traversal_plan", declining)
    cache = CacheStore(tmp_path)

    level = _adapter(cache).plan_level(outline=[_node("s0")], depth=1, context=PlanContext(tree_hash="a" * 16, title="doc"))

    assert level is None
    assert cache.load_layer(SUMMARIZER_PLAN_LAYER) == []


def test_degraded_cache_entries_are_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(factory, "decide_traversal_plan", _stub_decide((_plan_entry("s0"),), calls))
    cache = CacheStore(tmp_path)
    context = PlanContext(tree_hash="a" * 16, title="doc")

    cache.set(SUMMARIZER_PLAN_LAYER, _deps_for("a" * 16), {"entries": [{"node_id": 7}, "garbage", {"node_id": "s0", "mode": "expand"}]})

    level = _adapter(cache).plan_level(outline=[_node("s0")], depth=1, context=context)

    # The shaped entry survives, the garbage is dropped, and the decision
    # model is never asked (the cache hit stands).
    assert level is not None
    assert [entry.node_id for entry in level.entries] == ["s0"]
    assert calls == []


def test_task_text_comes_from_the_template_resource() -> None:
    task = factory.get_template_loader().get_text(factory._TASK_TEMPLATE)
    assert task.strip(), "the standing task statement must be a non-empty resource"
    assert "wiki" in task.lower()
