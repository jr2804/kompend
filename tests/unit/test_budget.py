from __future__ import annotations

from knox.core.models import WikiNode
from knox.search.budget import ContextBudgetManager, RetrievalMode


def _make_node(
    node_id: str,
    content: str = "test content",
    depth: int = 0,
    parent_id: str | None = None,
    children_ids: list[str] | None = None,
) -> WikiNode:
    return WikiNode(
        id=node_id,
        title=node_id,
        content=content,
        node_type="concept",
        depth=depth,
        parent_id=parent_id,
        children_ids=children_ids or [],
    )


class TestContextBudgetManager:
    def test_available_tokens_default(self) -> None:
        mgr = ContextBudgetManager(128_000)
        assert 0 < mgr.available_tokens < 128_000

    def test_available_tokens_custom_reserves(self) -> None:
        mgr = ContextBudgetManager(128_000, system_prompt_reserve=500, response_reserve=500)
        expected = 128_000 - 500 - 500
        assert mgr.available_tokens == expected

    def test_available_tokens_minimum(self) -> None:
        mgr = ContextBudgetManager(100)
        assert mgr.available_tokens >= 1

    def test_plan_zoom_in_empty(self) -> None:
        mgr = ContextBudgetManager(128_000)
        plan = mgr.plan_zoom_in([], "test query")
        assert plan.mode == RetrievalMode.ZOOM_IN
        assert plan.node_ids == []

    def test_plan_zoom_in_selects_summaries(self) -> None:
        mgr = ContextBudgetManager(128_000)
        nodes = [_make_node(f"n{i}", depth=1, parent_id="root") for i in range(5)]
        plan = mgr.plan_zoom_in(nodes, "test", top_k_summaries=3, top_k_expand=2)
        assert plan.mode == RetrievalMode.ZOOM_IN
        assert len(plan.node_ids) > 0
        assert plan.estimated_tokens > 0

    def test_plan_zoom_out_empty(self) -> None:
        mgr = ContextBudgetManager(128_000)
        plan = mgr.plan_zoom_out([], "test query")
        assert plan.mode == RetrievalMode.ZOOM_OUT
        assert plan.node_ids == []

    def test_plan_zoom_out_collapses_to_sections(self) -> None:
        mgr = ContextBudgetManager(128_000)
        nodes = [
            _make_node("root", depth=0),
            _make_node("s1", depth=1, parent_id="root"),
            _make_node("s2", depth=1, parent_id="root"),
        ]
        plan = mgr.plan_zoom_out(nodes, "test", max_sections=2)
        assert plan.mode == RetrievalMode.ZOOM_OUT
        assert len(plan.node_ids) <= 2

    def test_plan_targeted_empty(self) -> None:
        mgr = ContextBudgetManager(128_000)
        plan = mgr.plan_targeted([], "test query")
        assert plan.mode == RetrievalMode.TARGETED
        assert plan.node_ids == []

    def test_plan_targeted_selects_structured(self) -> None:
        mgr = ContextBudgetManager(128_000)
        structured = _make_node("structured")
        structured.structured_sections["table"] = {"columns": ["a", "b"]}
        plain = _make_node("plain")
        plan = mgr.plan_targeted([structured, plain], "test", top_k=5)
        assert plan.mode == RetrievalMode.TARGETED
        assert "structured" in plan.node_ids
        assert "plain" not in plan.node_ids

    def test_plan_targeted_no_structured_fallback(self) -> None:
        mgr = ContextBudgetManager(128_000)
        nodes = [_make_node("n1"), _make_node("n2")]
        plan = mgr.plan_targeted(nodes, "test", top_k=5)
        assert plan.mode == RetrievalMode.TARGETED
        assert plan.node_ids == []

    def test_plan_auto_zoom_in_few_candidates(self) -> None:
        mgr = ContextBudgetManager(128_000)
        nodes = [_make_node("n1"), _make_node("n2")]
        plan = mgr.plan(nodes, "test", top_k=5)
        assert plan.mode == RetrievalMode.ZOOM_IN

    def test_plan_auto_zoom_out_many_candidates(self) -> None:
        mgr = ContextBudgetManager(128_000)
        nodes = [_make_node(f"n{i}") for i in range(20)]
        plan = mgr.plan(nodes, "test", top_k=5)
        assert plan.mode == RetrievalMode.ZOOM_OUT

    def test_plan_auto_targeted_structured_keywords(self) -> None:
        mgr = ContextBudgetManager(128_000)
        nodes = [_make_node("n1")]
        plan = mgr.plan(nodes, "show me the table", top_k=5)
        assert plan.mode == RetrievalMode.TARGETED

    def test_plan_auto_targeted_equation_keyword(self) -> None:
        mgr = ContextBudgetManager(128_000)
        nodes = [_make_node("n1")]
        plan = mgr.plan(nodes, "what is the formula for", top_k=5)
        assert plan.mode == RetrievalMode.TARGETED

    def test_plan_empty_candidates(self) -> None:
        mgr = ContextBudgetManager(128_000)
        plan = mgr.plan([], "test", top_k=5)
        assert plan.estimated_tokens == 0

    def test_estimate_tokens(self) -> None:
        ContextBudgetManager(128_000)
        nodes = [_make_node("n1", content="hello world")]
        est = ContextBudgetManager._estimate_tokens_for_nodes(nodes, ["n1"])
        assert est > 0

    def test_find_node(self) -> None:
        nodes = [_make_node("target")]
        found = ContextBudgetManager._find_node(nodes, "target")
        assert found is not None
        assert found.id == "target"

    def test_find_node_missing(self) -> None:
        nodes = [_make_node("a")]
        found = ContextBudgetManager._find_node(nodes, "missing")
        assert found is None
