"""TraversalAgent — one typed judgement per chunk-tree level.

The summarizer walks the chunk tree bottom-up, but not every branch deserves
the same work: a dense section is expanded in full, a homogeneous branch needs
one summary, boilerplate needs none. This agent plans one level at a time —
which children to expand, summarize, or skip — so the executor spends model
calls where the task needs them.

The decision grammar cannot carry the plan shape (a list of models is refused
outright, and free text is not expressible — see the ``check_decision_schema``
doctor check), so the agent is never asked for the plan directly. Each window
of at most :data:`MAX_TRAVERSAL_QUESTIONS` children is asked one question per
child through a dynamic verdict model minted by
:func:`traversal_verdict_model` (the split-judge pattern from
``kompend.preprocessing.splitting``), and :func:`plan_from_verdict` builds
the executor-facing :class:`TraversalPlan` in code. The budget stays a
contract: a window that expands past its remaining ``budget_nodes`` is sent
back with ``ModelRetry``, and the remaining budget shrinks window to window
so the merged plan can never exceed the level's budget.

A disabled or unreachable decision model yields ``None`` — never a partial
plan — so the executor falls back to the full tree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Final, Literal

from pydantic import BaseModel, Field, create_model
from pydantic_ai import ModelRetry, RunContext

from kompend.agents.base import AgentSpec, KompendAgentBase
from kompend.chunking.outline import OutlineNode
from kompend.core.config._configs import DecisionConfig
from kompend.llm.decision import decision_model_string
from kompend.llm.models import build_decision_model
from kompend.logging import get_logger

logger = get_logger(__name__)

#: Most questions the decision endpoint answers in one request. The cap is the
#: endpoint's ("questions must contain 1–64 fields"), mirrored here rather than
#: imported: ``kompend.preprocessing.splitting`` already depends on the agent
#: layer, so importing its ``MAX_QUESTIONS_PER_REQUEST`` would close a
#: preprocessing <-> agents cycle for one integer. A wider level is judged in
#: consecutive windows (see :func:`traversal_windows`).
MAX_TRAVERSAL_QUESTIONS: Final[int] = 64

#: Fallback per-call timeout when no config provides one (mirrors the
#: ``DecisionConfig.timeout`` default).
_DEFAULT_TIMEOUT: Final[float] = 10.0


class PlanMode(StrEnum):
    """What the summarizer should do with a child node."""

    EXPAND = "expand"
    SUMMARIZE = "summarize"
    SKIP = "skip"


class PlanEntry(BaseModel):
    """The executor-facing plan for one child node.

    ``focus`` is structural — the child's heading chain, or its preview when
    the chain is empty — because the decision grammar has no free text and the
    summarizer already receives the task statement. ``reason`` is a fixed
    label from the skip variant, never model prose.
    """

    node_id: str
    mode: PlanMode
    focus: str | None = None
    reason: str | None = None


class TraversalPlan(BaseModel):
    """The executor-facing plan for one level: every child exactly once."""

    entries: list[PlanEntry]
    notes: str | None = None


class SkipReason(StrEnum):
    """Skip vocabulary: one variant per skippable branch kind.

    The member value is the verdict option the model answers; the registry
    below carries the fixed reason label the plan reports. The verdict
    ``Literal`` is built from these members, so the grammar and the mapping
    cannot drift apart.
    """

    BOILERPLATE = "skip_boilerplate"
    APPENDIX = "skip_appendix"
    REPEATED = "skip_repeated"
    OTHER = "skip_other"


#: One question per child: expand, summarize, or skip with the reason baked
#: into the variant, so the verdict needs no second question and no free text.
type TraversalAnswer = Literal[
    "expand",
    "summarize",
    SkipReason.BOILERPLATE,
    SkipReason.APPENDIX,
    SkipReason.REPEATED,
    SkipReason.OTHER,
]

#: Fixed reason label per skip variant — code text, not model text.
_SKIP_REASON_LABELS: Final[dict[SkipReason, str]] = {
    SkipReason.BOILERPLATE: "boilerplate",
    SkipReason.APPENDIX: "appendix",
    SkipReason.REPEATED: "repeated material",
    SkipReason.OTHER: "other",
}


@dataclass
class TraversalDeps:
    """Deps for the traversal decision over one level.

    Attributes:
        title: Document title (rendered as the plan's context).
        purpose: Frontmatter/purpose text of the document.
        task: What the wiki must answer — the standard the plan is judged by.
        outline: The level's children, in document order.
        depth: Depth of the level in the tree.
        budget_nodes: Most children the plan may expand; the hook enforces it.
        budget_windows: Window budget, informational context for the model.
        budget_chars: Character budget, informational context for the model.
        source_id: Log/provenance label.
        timeout: Per-call timeout in seconds.
    """

    title: str
    purpose: str
    task: str
    outline: list[OutlineNode] = field(default_factory=list)
    depth: int = 0
    budget_nodes: int = 0
    budget_windows: int = 0
    budget_chars: int = 0
    source_id: str = "traversal"
    timeout: float = _DEFAULT_TIMEOUT


def traversal_field_name(index: int) -> str:
    """Field name of the *index*-th child (0-based, global to the level).

    Positional on purpose, like the split judge's ``boundary_{index:03d}``:
    the child each answer belongs to is decided by :func:`plan_from_verdict`
    from the ordered outline, so an answer cannot be attached to the wrong
    child through an echoed node id. Global so every window's answers land in
    one mapping.
    """
    return f"child_{index}_mode"


def traversal_windows(child_count: int, *, max_questions: int = MAX_TRAVERSAL_QUESTIONS) -> tuple[tuple[int, int], ...]:
    """Split a level's children into request windows.

    Returns inclusive ``(first_index, last_index)`` pairs over the children,
    in order, each covering at most *max_questions* of them. A level that fits
    one request yields one window — the whole level — so an ordinary planning
    call is unchanged.
    """
    if child_count < 0:
        msg = f"a traversal verdict needs a non-negative child count; got {child_count}"
        raise ValueError(msg)
    if max_questions < 1:
        msg = f"a traversal window needs at least one question; got {max_questions}"
        raise ValueError(msg)
    if child_count == 0:
        return ()
    return tuple((first, min(first + max_questions - 1, child_count - 1)) for first in range(0, child_count, max_questions))


def traversal_verdict_model(children: Sequence[OutlineNode], *, first_index: int = 0) -> type[BaseModel]:
    """Mint the verdict model for one window of children.

    One :data:`TraversalAnswer` field per child, named with the global
    :func:`traversal_field_name`, so the answers of every window merge into
    one mapping. The field description is the question — it names the child,
    its heading, and its size, because a positional reference alone risks the
    model answering for the wrong row of the outline table.

    Raises:
        ValueError: when *children* is empty (nothing to decide).
    """
    if not children:
        msg = "a traversal verdict needs at least one child; got none"
        raise ValueError(msg)
    fields: dict[str, Any] = {}
    for offset, child in enumerate(children):
        heading = " / ".join(child.heading_chain) if child.heading_chain else "untitled"
        fields[traversal_field_name(first_index + offset)] = (
            TraversalAnswer,
            Field(
                description=(
                    f"Decide how the summarizer should treat child {child.node_id} "
                    f"({heading}, {child.subtree_chars} chars): expand = distinct dense content, "
                    "summarized in full; summarize = homogeneous branch, one summary suffices; "
                    "skip_boilerplate / skip_appendix / skip_repeated / skip_other = boilerplate / "
                    "appendix / repeated material / other content that is never summarized."
                )
            ),
        )
    return create_model("TraversalVerdict", __module__=__name__, **fields)


def plan_from_verdict(answers: Mapping[str, str], *, children: Sequence[OutlineNode]) -> TraversalPlan:
    """Turn answered verdict fields into the executor-facing plan.

    Every child appears exactly once: a missing or extra key is a caller bug,
    not a model judgement, so it raises instead of degrading. An unknown
    answer value raises the same way — the verdict model's ``Literal``
    already bars it at parse time, so reaching here means the mapping was
    built by hand.

    Raises:
        ValueError: when the answers do not cover exactly *children*, or an
            answer is not a known option.
    """
    if not children:
        msg = "a traversal plan needs at least one child; got none"
        raise ValueError(msg)
    expected = {traversal_field_name(index) for index in range(len(children))}
    given = set(answers)
    if given != expected:
        missing = sorted(expected - given)
        extra = sorted(given - expected)
        msg = f"traversal answers must cover every child exactly once: missing {missing}, extra {extra}"
        raise ValueError(msg)
    entries: list[PlanEntry] = []
    for index, child in enumerate(children):
        answer = answers[traversal_field_name(index)]
        if answer == PlanMode.EXPAND.value:
            mode, reason = PlanMode.EXPAND, None
        elif answer == PlanMode.SUMMARIZE.value:
            mode, reason = PlanMode.SUMMARIZE, None
        else:
            try:
                skip = SkipReason(answer)
            except ValueError:
                msg = f"unknown traversal answer {answer!r} for child {child.node_id}; want expand, summarize, or a skip_* option"
                raise ValueError(msg) from None
            mode, reason = PlanMode.SKIP, _SKIP_REASON_LABELS[skip]
        heading = " / ".join(child.heading_chain)
        entries.append(PlanEntry(node_id=child.node_id, mode=mode, focus=heading or child.preview or None, reason=reason))
    return TraversalPlan(entries=entries)


def _traversal_validate_output(ctx: RunContext[TraversalDeps], output: BaseModel) -> BaseModel:
    """Enforce the expansion budget: too many EXPANDs ask for a tighter plan.

    The budget is a contract, not advice. Summaries are cheap by design, so
    only expansions count — a window that spends past its remaining
    ``budget_nodes`` is sent back for a tighter plan instead of silently
    overrunning the level.
    """
    expands = sum(1 for value in output.model_dump().values() if value == PlanMode.EXPAND.value)
    budget = ctx.deps.budget_nodes
    if expands > budget:
        raise ModelRetry(
            f"Plan expands {expands} children but budget_nodes is {budget}: change at least {expands - budget} of them to summarize or a skip_* option."
        )
    return output


SPEC_TRAVERSAL: Final[AgentSpec] = AgentSpec(
    name="summarizer_traversal",
    result_type=TraversalPlan,
    deps_type=TraversalDeps,
    template_name="traversal_plan.md.j2",
    system_template_name="traversal_plan_system.md.j2",
    template_prefix="decision/",
    temperature=0.0,
    retries=1,
    model_builder=build_decision_model,
    validate_output=_traversal_validate_output,
)


class TraversalAgent(KompendAgentBase[TraversalPlan, TraversalDeps]):
    """Decision-model agent that plans one chunk-tree level.

    The registered spec documents the executor-facing contract
    (:class:`TraversalPlan`); each window run sets the instance
    ``result_type`` to its minted verdict model before ``run()``, because the
    grammar cannot carry the plan shape. The validation hook and the prompt
    wiring come from the spec either way.
    """

    spec: ClassVar[AgentSpec | None] = SPEC_TRAVERSAL
    temperature: float | None = 0.0


def make_traversal_agent(
    config: DecisionConfig | None,
    *,
    model: str | None = None,
    timeout: float | None = None,
) -> TraversalAgent | None:
    """Build a :class:`TraversalAgent` from the decision config.

    Args:
        config: The ``[decision]`` block — model and timeout source.
        model: Optional per-feature override for the model string.
        timeout: Optional per-feature override for the per-call timeout.

    Returns:
        The agent, or ``None`` when no decision model is configured (disabled
        block or no model) — the caller then falls back to the full tree.
    """
    model_string = model if model is not None else decision_model_string(config)
    if model_string is None:
        return None
    if timeout is None:
        timeout = config.timeout if config is not None else _DEFAULT_TIMEOUT
    return TraversalAgent(model_string=model_string, timeout=timeout, retries=1)


def decide_traversal_plan(
    config: DecisionConfig | None,
    *,
    title: str,
    purpose: str,
    task: str,
    outline: list[OutlineNode],
    depth: int,
    budget_nodes: int,
    budget_windows: int,
    budget_chars: int,
    source_id: str = "traversal",
    model: str | None = None,
    timeout: float | None = None,
) -> TraversalPlan | None:
    """Plan one level: which children to expand, summarize, or skip.

    Windows of at most :data:`MAX_TRAVERSAL_QUESTIONS` children are judged in
    turn; the remaining node budget shrinks window to window, so the merged
    plan can never exceed the level's budget even though each window is
    validated on its own.

    Returns the plan, or ``None`` — never a partial plan — when no decision
    model is configured or a window fails; the executor then falls back to
    the full tree.
    """
    agent = make_traversal_agent(config, model=model, timeout=timeout)
    if agent is None:
        logger.warning("Decision model disabled — falling back to the full tree")
        return None
    if not outline:
        return TraversalPlan(entries=[])
    windows = traversal_windows(len(outline))
    if len(windows) > 1:
        logger.info(
            "Planning %d children in %d window(s): the decision service takes at most %d questions per request",
            len(outline),
            len(windows),
            MAX_TRAVERSAL_QUESTIONS,
        )
    answers: dict[str, str] = {}
    remaining = budget_nodes
    for first, last in windows:
        window_children = outline[first : last + 1]
        agent.result_type = traversal_verdict_model(window_children, first_index=first)
        deps = TraversalDeps(
            title=title,
            purpose=purpose,
            task=task,
            outline=window_children,
            depth=depth,
            budget_nodes=remaining,
            budget_windows=budget_windows,
            budget_chars=budget_chars,
            source_id=source_id,
            timeout=agent.timeout,
        )
        try:
            verdict = agent.run(deps)
        except Exception as exc:  # ponytail: a sidecar failure must never break the pipeline
            logger.warning("Decision model unavailable (%s) — falling back to the full tree", exc)
            return None
        window_answers = {name: value for name, value in verdict.model_dump().items() if isinstance(value, str)}
        answers.update(window_answers)
        remaining -= sum(1 for value in window_answers.values() if value == PlanMode.EXPAND.value)
    try:
        return plan_from_verdict(answers, children=outline)
    except ValueError as exc:  # ponytail: a merged plan that covers wrong is no plan
        logger.warning("Traversal verdict did not cover the level (%s) — falling back to the full tree", exc)
        return None
