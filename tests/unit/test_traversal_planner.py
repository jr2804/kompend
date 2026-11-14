"""Tests for the traversal planner (agents/traversal.py).

The decision grammar cannot carry the plan shape, so the planner asks one
typed question per child through a dynamic verdict model and builds the
executor-facing plan in code. These tests pin that contract without a model
or the decision endpoint: verdict field naming, window merging, budget
enforcement, the disabled-model fallback, and the templates. The
expressibility test below runs the same check the connectivity-decision
doctor probe runs, so the dynamic model is pinned without touching the
doctor schema.
"""

from __future__ import annotations

import importlib
from typing import Any, cast, get_args

import pytest
from pydantic import BaseModel
from pydantic_ai import ModelRetry, RunContext

from kompend.agents import TraversalAgent
from kompend.agents import TraversalDeps as LazyTraversalDeps
from kompend.agents.specs import AgentName, get_spec
from kompend.agents.traversal import (
    SPEC_TRAVERSAL,
    PlanMode,
    SkipReason,
    TraversalAnswer,
    TraversalDeps,
    TraversalPlan,
    decide_traversal_plan,
    make_traversal_agent,
    plan_from_verdict,
    traversal_field_name,
    traversal_verdict_model,
    traversal_windows,
)
from kompend.chunking.outline import OutlineNode
from kompend.core.config._configs import DecisionConfig
from kompend.core.templates import get_template_loader


def _node(node_id: str, *, heading: tuple[str, ...] = ("Doc", "Section"), subtree_chars: int = 100, preview: str = "preview text") -> OutlineNode:
    """One outline entry with fixed shape facts."""
    return OutlineNode(
        node_id=node_id,
        heading_chain=heading,
        depth=1,
        child_count=0,
        payload_chars=subtree_chars,
        subtree_chars=subtree_chars,
        token_estimate=40,
        kinds=("chunk",),
        preview=preview,
    )


def _deps(children: list[OutlineNode], *, budget_nodes: int = 2) -> TraversalDeps:
    """Deps over *children* with a fixed identity."""
    return TraversalDeps(
        title="Doc",
        purpose="Purpose.",
        task="Answer the task.",
        outline=children,
        depth=1,
        budget_nodes=budget_nodes,
        budget_windows=2,
        budget_chars=1000,
    )


def test_valid_answer_set_yields_plan_with_correct_modes_and_reasons() -> None:
    children = [_node("a"), _node("b"), _node("c"), _node("d"), _node("e"), _node("f")]
    answers = {
        "child_0_mode": "expand",
        "child_1_mode": "summarize",
        "child_2_mode": "skip_boilerplate",
        "child_3_mode": "skip_appendix",
        "child_4_mode": "skip_repeated",
        "child_5_mode": "skip_other",
    }
    plan = plan_from_verdict(answers, children=children)

    assert [entry.node_id for entry in plan.entries] == ["a", "b", "c", "d", "e", "f"]
    assert [entry.mode for entry in plan.entries] == [
        PlanMode.EXPAND,
        PlanMode.SUMMARIZE,
        PlanMode.SKIP,
        PlanMode.SKIP,
        PlanMode.SKIP,
        PlanMode.SKIP,
    ]
    assert [entry.reason for entry in plan.entries] == [None, None, "boilerplate", "appendix", "repeated material", "other"]
    assert plan.notes is None


def test_skip_variants_map_to_skip_with_fixed_labels() -> None:
    children = [_node("a"), _node("b")]
    plan = plan_from_verdict({"child_0_mode": "skip_appendix", "child_1_mode": "skip_other"}, children=children)

    assert all(entry.mode is PlanMode.SKIP for entry in plan.entries)
    assert [entry.reason for entry in plan.entries] == ["appendix", "other"]


def test_focus_is_heading_chain_or_preview() -> None:
    chained = _node("a", heading=("Doc", "Part", "Section"))
    bare = _node("b", heading=(), preview="first line here")
    empty = _node("c", heading=(), preview="")
    plan = plan_from_verdict(
        {"child_0_mode": "expand", "child_1_mode": "summarize", "child_2_mode": "skip_other"},
        children=[chained, bare, empty],
    )

    assert [entry.focus for entry in plan.entries] == ["Doc / Part / Section", "first line here", None]


def test_plan_rejects_incomplete_or_unknown_answers() -> None:
    children = [_node("a"), _node("b")]

    with pytest.raises(ValueError, match="missing"):
        plan_from_verdict({"child_0_mode": "expand"}, children=children)
    with pytest.raises(ValueError, match="extra"):
        plan_from_verdict({"child_0_mode": "expand", "child_1_mode": "summarize", "child_9_mode": "expand"}, children=children)
    with pytest.raises(ValueError, match="unknown traversal answer"):
        plan_from_verdict({"child_0_mode": "expand", "child_1_mode": "sideways"}, children=children)
    with pytest.raises(ValueError, match="at least one child"):
        plan_from_verdict({}, children=[])


def test_windows_cover_64_in_one_and_65_in_two() -> None:
    assert traversal_windows(0) == ()
    assert traversal_windows(1) == ((0, 0),)
    assert traversal_windows(64) == ((0, 63),)
    assert traversal_windows(65) == ((0, 63), (64, 64))
    assert traversal_windows(129) == ((0, 63), (64, 127), (128, 128))
    with pytest.raises(ValueError, match="at least one question"):
        traversal_windows(3, max_questions=0)


def test_window_field_names_are_global_so_windows_merge() -> None:
    children = [_node(f"n{i}") for i in range(65)]
    first = traversal_verdict_model(children[:64], first_index=0)
    second = traversal_verdict_model(children[64:], first_index=64)

    assert set(first.model_fields) == {traversal_field_name(i) for i in range(64)}
    assert set(second.model_fields) == {traversal_field_name(64)}

    merged = {traversal_field_name(i): ("expand" if i < 2 else "summarize") for i in range(65)}
    plan = plan_from_verdict(merged, children=children)
    assert len(plan.entries) == 65
    assert plan.entries[64].node_id == "n64"


def test_verdict_model_needs_at_least_one_child() -> None:
    with pytest.raises(ValueError, match="at least one child"):
        traversal_verdict_model([])


def test_verdict_model_is_expressible_like_the_doctor_probe() -> None:
    grammar: Any = importlib.import_module("pydantic_ai.models.decision")
    output: Any = importlib.import_module("pydantic_ai._output")
    model_class = importlib.import_module("pydantic_ai.models.typesafe").TypeSafeModel
    limits = grammar._Limits(
        choice_options=model_class.max_choice_options,
        score_levels=model_class.max_score_levels,
        requires_instructions=model_class.requires_instructions,
        questions=model_class.max_questions,
    )
    model = traversal_verdict_model([_node("a"), _node("b"), _node("c")])

    tool = output.OutputToolset.build([model])._tool_defs[0]
    questions = grammar._questions(grammar._fields(tool), tool, None, limits, label=None)

    assert grammar._expressible(tool, None, limits) is True
    assert len(questions) == 3


def test_answer_literal_is_derived_from_the_skip_enum() -> None:
    options = set(get_args(TraversalAnswer.__value__))

    assert "expand" in options
    assert "summarize" in options
    for reason in SkipReason:
        assert reason in options


def test_expansion_over_budget_asks_for_a_tighter_plan() -> None:
    model = traversal_verdict_model([_node("a"), _node("b")])
    over = model.model_validate({"child_0_mode": "expand", "child_1_mode": "expand"})
    fitting = model.model_validate({"child_0_mode": "expand", "child_1_mode": "summarize"})
    assert SPEC_TRAVERSAL.validate_output is not None
    hook = SPEC_TRAVERSAL.validate_output
    ctx = cast("RunContext[TraversalDeps]", _StubCtx(budget_nodes=1))

    with pytest.raises(ModelRetry, match="budget_nodes is 1"):
        hook(ctx, over)
    assert hook(ctx, fitting) is fitting


def test_disabled_model_returns_none_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        assert decide_traversal_plan(DecisionConfig(), **_plan_kwargs([_node("a")])) is None
    warnings = [record for record in caplog.records if "full tree" in record.message]
    assert len(warnings) == 1


def test_empty_outline_yields_empty_plan_without_a_model() -> None:
    config = DecisionConfig(enabled=True, model="typesafe/test-model")
    assert decide_traversal_plan(config, **_plan_kwargs([])) == TraversalPlan(entries=[])


def test_user_template_renders_empty_and_200_entry_levels() -> None:
    loader = get_template_loader()
    empty = loader.render("decision/traversal_plan.md.j2", **_template_kwargs([]))
    assert "Level 1" in empty
    assert "expand at most 2 children" in empty

    children = [_node(f"n{i:03d}") for i in range(200)]
    rendered = loader.render("decision/traversal_plan.md.j2", **_template_kwargs(children))
    for child in children:
        assert child.node_id in rendered
    assert "n199" in rendered


def test_system_template_states_the_one_question_contract_and_six_options() -> None:
    rendered = get_template_loader().render("decision/traversal_plan_system.md.j2")

    assert "one question" in rendered.lower()
    for option in ("expand", "summarize", "skip_boilerplate", "skip_appendix", "skip_repeated", "skip_other"):
        assert option in rendered
    assert "focus" not in rendered.lower() or "free text" in rendered.lower()


def test_spec_is_registered_with_templates_hook_and_budget() -> None:
    assert AgentName.SUMMARIZER_TRAVERSAL.value == "summarizer_traversal"
    assert get_spec("summarizer_traversal") is SPEC_TRAVERSAL
    assert SPEC_TRAVERSAL.temperature == 0.0
    assert SPEC_TRAVERSAL.retries == 1
    assert SPEC_TRAVERSAL.template_prefix == "decision/"
    assert SPEC_TRAVERSAL.validate_output is not None

    assert TraversalAgent.spec is SPEC_TRAVERSAL
    assert LazyTraversalDeps is TraversalDeps


def test_factory_returns_none_when_disabled_and_agent_when_enabled() -> None:
    assert make_traversal_agent(DecisionConfig()) is None
    assert make_traversal_agent(None) is None

    config = DecisionConfig(enabled=True, model="typesafe/test-model")
    agent = make_traversal_agent(config)
    assert agent is not None
    assert agent.model_string == "typesafe/test-model"
    assert agent.timeout == config.timeout
    assert agent.retries == 1

    overridden = make_traversal_agent(DecisionConfig(), model="typesafe/override", timeout=3.0)
    assert overridden is not None
    assert overridden.model_string == "typesafe/override"
    assert overridden.timeout == 3.0


def test_decide_loop_merges_windows_and_falls_back_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    children = [_node(f"n{i}") for i in range(65)]
    stub = _StubAgent(
        {
            0: {traversal_field_name(i): "expand" for i in range(64)},
            64: {traversal_field_name(64): "summarize"},
        }
    )
    monkeypatch.setattr("kompend.agents.traversal.make_traversal_agent", lambda *args, **kwargs: stub)

    plan = decide_traversal_plan(DecisionConfig(enabled=True, model="typesafe/test-model"), **_plan_kwargs(children, budget_nodes=64))
    assert plan is not None
    assert len(plan.entries) == 65
    assert plan.entries[0].mode is PlanMode.EXPAND
    assert plan.entries[64].mode is PlanMode.SUMMARIZE
    # The remaining budget shrinks window to window: the second window saw 0 left.
    assert stub.seen_budgets == [64, 0]

    failing = _StubAgent({}, fail=True)
    monkeypatch.setattr("kompend.agents.traversal.make_traversal_agent", lambda *args, **kwargs: failing)
    assert decide_traversal_plan(DecisionConfig(enabled=True, model="typesafe/test-model"), **_plan_kwargs(children[:2])) is None


def _plan_kwargs(children: list[OutlineNode], *, budget_nodes: int = 2) -> dict[str, Any]:
    """Shared keyword args for :func:`decide_traversal_plan`."""
    return {
        "title": "Doc",
        "purpose": "Purpose.",
        "task": "Answer the task.",
        "outline": children,
        "depth": 1,
        "budget_nodes": budget_nodes,
        "budget_windows": 2,
        "budget_chars": 1000,
    }


def _template_kwargs(children: list[OutlineNode]) -> dict[str, Any]:
    """Shared render variables for the user template."""
    return {
        "title": "Doc",
        "purpose": "Purpose.",
        "task": "Answer the task.",
        "outline": children,
        "depth": 1,
        "budget_nodes": 2,
        "budget_windows": 2,
        "budget_chars": 1000,
    }


class _StubCtx:
    """Minimal ``RunContext`` stand-in carrying window deps."""

    def __init__(self, *, budget_nodes: int) -> None:
        self.deps = TraversalDeps(
            title="Doc",
            purpose="Purpose.",
            task="Answer the task.",
            outline=[],
            depth=1,
            budget_nodes=budget_nodes,
            budget_windows=2,
            budget_chars=1000,
        )


class _StubAgent:
    """Fake traversal agent answering each window from a fixed mapping."""

    def __init__(self, windows: dict[int, dict[str, str]], *, fail: bool = False) -> None:
        self._windows = windows
        self._fail = fail
        self.seen_budgets: list[int] = []
        self.result_type: Any = None
        self.timeout = 10.0

    def run(self, deps: TraversalDeps) -> BaseModel:
        if self._fail:
            msg = "endpoint down"
            raise RuntimeError(msg)
        self.seen_budgets.append(deps.budget_nodes)
        first = int(deps.outline[0].node_id[1:])
        model = traversal_verdict_model(deps.outline, first_index=first)
        return model.model_validate(self._windows[first])
