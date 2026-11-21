"""Variant A of the traversal spike (bead knox-u84v.6): shipped planner + worker.

Runs the shipped chunk-tree path over the synthetic corpus: a
``NodeSummarizer`` with ``TraversalMode.AUTO`` consults a traversal planner
for every eligible level top-down, then summarizes the surviving worklist
with the shipped ``SummarizerAgent`` session loop. Every model request goes
through the shared counted model so the runner can compare totals against
variant B.

Injection route: both agents are built by their shipped factories and then
given the counted model through the per-instance ``model_builder`` hook
(``KompendAgentBase.run`` prefers ``self.model_builder`` when set, and the
base dataclass has no slots, so an instance attribute cleanly shadows the
spec-installed builder without touching the class). The planner additionally
needs its window loop replicated here (``_decide_traversal`` mirrors
``decide_traversal_plan`` line for line) because that function constructs its
own SystemOne-backed agent internally, and this spike's endpoint only speaks
the Anthropic chat protocol.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

# The spike scripts run as top-level modules out of this directory
# (``run_spike.py`` imports ``variant_a`` by bare name), so make the
# directory importable for direct runs too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    BUDGET_NODES,
    BUDGET_WINDOWS,
    MODEL,
    Counter,
    Stopwatch,
    SyntheticDoc,
    VariantResult,
    build_counted_model,
    configure_provider,
    task_statement,
)
from pydantic_ai.models import Model
from tree import build_synthetic_doc

from kompend.agents.summarizer import make_summarizer_agent
from kompend.agents.traversal import (
    PlanEntry as TraversalPlanEntry,
)
from kompend.agents.traversal import (
    PlanMode as TraversalPlanMode,
)
from kompend.agents.traversal import (
    TraversalAgent,
    TraversalDeps,
    TraversalPlan,
    make_traversal_agent,
    traversal_field_name,
    traversal_verdict_model,
    traversal_windows,
)
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
)
from kompend.core.config.models.compile import BudgetConfig
from kompend.core.text.density import CONSERVATIVE_CHARS_PER_TOKEN
from kompend.llm.context_window import ContextSource, ContextWindow
from kompend.llm.request_budget import RequestBudget

logger = logging.getLogger(__name__)

#: Model window the spike sizes payloads against. Mirrors the
#: ``context_window: 200_000`` the shared provider block registers in
#: ``common.configure_provider``; the payload math below reads only
#: ``max_input``. ``max_output`` is undiscovered here and unused by
#: ``RequestBudget`` — the placeholder only completes the window.
#: Per-call timeout for the decision (traversal) agent. The shipped default
#: is ``DecisionConfig.timeout`` = 10 s, which is sized for a local model
#: answering a handful of fields; against a remote endpoint a 12-field
#: verdict regularly needs longer, and a timeout makes the planner decline
#: the level, which makes the executor summarize the whole tree. The spike
#: therefore gives the decision agent a remote-sized timeout and records it.
DECISION_TIMEOUT_S = 120.0

_SPIKE_WINDOW = ContextWindow(
    max_input=200_000,
    max_output=8192,
    source=ContextSource.USER_PROVIDER,
)


def _shipped_default(field_name: str) -> int | bool:
    """A ``BudgetConfig`` field default, so the spike tracks the shipped sizing."""
    default = BudgetConfig.model_fields[field_name].default
    if not isinstance(default, (int, bool)):
        raise TypeError(f"unexpected default for {field_name}: {default!r}")
    return default


def _decide_traversal(
    agent: TraversalAgent,
    *,
    title: str,
    purpose: str,
    task: str,
    outline: list[OutlineNode],
    depth: int,
    budget_nodes: int,
    budget_windows: int,
    budget_chars: int,
) -> TraversalPlan | None:
    """The shipped ``decide_traversal_plan`` window loop, minus agent construction.

    The agent arrives prebuilt with the counted model (see :func:`run`);
    everything else — windowing, per-window verdict, remaining-budget
    accounting, verdict merge — follows the shipped function exactly.
    """
    if not outline:
        return TraversalPlan(entries=[])
    answers: dict[str, str] = {}
    remaining = budget_nodes
    for first, last in traversal_windows(len(outline)):
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
            source_id="traversal",
            timeout=agent.timeout,
        )
        try:
            verdict = agent.run(deps)
        except Exception as exc:  # a sidecar failure must never break the run
            logger.warning("Decision model unavailable (%s) — falling back to the full tree", exc)
            return None
        window_answers = {name: value for name, value in verdict.model_dump().items() if isinstance(value, str)}
        answers.update(window_answers)
        remaining -= sum(1 for value in window_answers.values() if value == TraversalPlanMode.EXPAND.value)
    try:
        return _merge_verdict(answers, outline)
    except ValueError as exc:  # a merged plan that covers wrong is no plan
        logger.warning("Traversal verdict did not cover the level (%s) — falling back to the full tree", exc)
        return None


#: The skip labels the shipped planner reports, mirrored for the merge.
_SKIP_LABELS: dict[str, str] = {  # nocheck: str-registry — mirrors the shipped planner's skip vocabulary (kompend.agents.traversal._SKIP_REASON_LABELS)
    "skip_boilerplate": "boilerplate",
    "skip_appendix": "appendix",
    "skip_repeated": "repeated material",
    "skip_other": "other",
}


def _merge_verdict(answers: dict[str, str], children: list[OutlineNode]) -> TraversalPlan:
    """Merge a windowed verdict the way the *executor* merges a plan.

    The shipped :func:`kompend.agents.traversal.plan_from_verdict` refuses a
    verdict that does not cover the level exactly, and
    :func:`kompend.agents.traversal.decide_traversal_plan` then declines the
    level — which the executor answers by summarizing the whole tree. The
    model this spike runs against over-answers the minted verdict model (it
    echoed the previous level's field count), so with the strict merge every
    level declined and this arm measured the fallback instead of the
    traversal shape.

    The executor's own rule is the tolerant one: a child the plan does not
    mention is EXPAND, because "a partial plan can never silently drop a
    subtree" (``NodeSummarizer._plan_tree``). The spike merges the same
    way — an unmentioned child is EXPAND, an answer outside the level is
    ignored — so the arm measures the shape and not one model's schema
    discipline. The shipped behaviour is recorded as a finding: a strict
    merge plus a fail-open executor turns one sloppy verdict into the whole
    tree, which is the cost the epic set out to bound.
    """
    entries: list[TraversalPlanEntry] = []
    for index, child in enumerate(children):
        answer = answers.get(traversal_field_name(index))
        if answer in (TraversalPlanMode.EXPAND.value, TraversalPlanMode.SUMMARIZE.value):
            mode = TraversalPlanMode(answer)
            reason = None
        elif answer is None:
            # The plan does not mention this child: EXPAND, the executor's
            # safe default.
            mode = TraversalPlanMode.EXPAND
            reason = None
        else:
            mode = TraversalPlanMode.SKIP
            reason = _SKIP_LABELS.get(answer, "other")
        heading = " / ".join(child.heading_chain)
        entries.append(
            TraversalPlanEntry(
                node_id=child.node_id,
                mode=mode,
                focus=heading or child.preview or None,
                reason=reason,
            )
        )
    return TraversalPlan(entries=entries)


class _SpikePlanner:
    """The ``TraversalPlanner`` protocol over the shipped decision loop.

    No plan cache is attached on purpose: the spike measures cold runs, so
    every eligible level calls the planner agent and every call is counted
    under the ``"decision"`` phase.

    A declined level falls back to EXPAND-everything — the executor's own
    treatment of unplanned children, and what the shipped docstring promises
    ("falling back to the full tree") — instead of raising
    ``_PlannerUnavailable``. The fallback is recorded in notes so the
    comparison stays honest.

    The shipped node budget is per level: ``decide_traversal_plan`` receives
    the same ``BUDGET_NODES`` at every level, so it cannot bound the tree.
    The spike gives both variants the same *total* budget instead: this
    planner counts planned work nodes (EXPAND + SUMMARIZE — both cost a
    summary) across the whole run, and converts a level's excess entries to
    recorded ``"budget"`` skips, trailing entries first and SUMMARIZE
    before EXPAND.
    """

    def __init__(self, agent_factory: Callable[[], TraversalAgent], counter: Counter, *, task: str, budget_chars: int) -> None:
        self._agent_factory = agent_factory
        self._counter = counter
        self._task = task
        self._budget_chars = budget_chars
        self.skips: dict[str, str] = {}
        self.plan_calls = 0
        self.fallbacks = 0
        self.planned_children = 0
        self.work_planned = 0
        self.budget_unplanned = 0

    def plan_level(self, *, outline: list[OutlineNode], depth: int, context: PlanContext) -> LevelPlan | None:
        """Plan one level; never returns None (EXPAND-all on decline)."""
        self.plan_calls += 1
        self.planned_children += len(outline)
        previous, self._counter.phase = self._counter.phase, "decision"
        try:
            # A fresh agent per level: the minted verdict model is fixed when
            # the pydantic-ai agent is built, so an agent shared across levels
            # keeps the first level's field names and answers the next level
            # against the wrong schema. The shipped adapter builds its agent
            # per level for the same reason (``decide_traversal_plan`` mints
            # the agent inside the call).
            plan = _decide_traversal(
                self._agent_factory(),
                title=context.title,
                purpose=f"Compiled wiki for {context.title}",
                task=self._task,
                outline=list(outline),
                depth=depth,
                budget_nodes=BUDGET_NODES,
                budget_windows=BUDGET_WINDOWS,
                budget_chars=self._budget_chars,
            )
        finally:
            self._counter.phase = previous
        entries: list[PlanEntry | None]
        if plan is None:
            self.fallbacks += 1
            entries = [PlanEntry(node_id=child.node_id, mode=PlanMode.EXPAND) for child in outline]
        else:
            entries = [
                PlanEntry(
                    node_id=entry.node_id,
                    mode=PlanMode(entry.mode.value),
                    focus=entry.focus,
                    reason=entry.reason,
                )
                for entry in plan.entries
            ]
        self._enforce_total_budget(entries)
        planned = [entry for entry in entries if entry is not None]
        for entry in planned:
            if entry.mode is PlanMode.SKIP:
                self.skips[entry.node_id] = entry.reason or PlanMode.SKIP.value
        return LevelPlan(entries=tuple(planned))

    def _enforce_total_budget(self, entries: list[PlanEntry | None]) -> None:
        """Cap planned work at ``BUDGET_NODES`` across the whole tree.

        EXPAND and SUMMARIZE entries both cost a summary, so both count as
        work. When the cap bites, the excess entries are **dropped from the
        plan**, not turned into skips: a child the plan does not mention is
        undecided, and the executor inlines an undecided child's subtree
        into its parent's payload ("a child with no decision has its
        undecided subtree inlined leaf by leaf, so summarizing the branch
        never silently loses the skipped-under-it content").

        That difference is the whole measurement. A skip drops the branch's
        content; an unplanned child keeps it, inside the parent's summary.
        The first version of this cap converted the excess into skips, and
        the content of the cut branches was lost — the arm answered none of
        the planted questions, and the number measured the cap's allocation
        rule rather than the traversal shape.
        """
        work = [i for i, entry in enumerate(entries) if entry.mode is not PlanMode.SKIP]
        allowed = BUDGET_NODES - self.work_planned
        if len(work) <= allowed:
            self.work_planned += len(work)
            return
        excess = len(work) - allowed
        victims = sorted(work, key=lambda i: (entries[i].mode is PlanMode.EXPAND, -i))
        for i in victims[:excess]:
            entries[i] = None
        self.budget_unplanned += excess
        self.work_planned += allowed


def _counted_model(counter: Counter) -> Model:
    """The shared counted model, see-through for protocol detection.

    ``requires_output_ceiling`` looks through a wrapper's ``delegate`` (the
    same hook its own recursion exists for); without it the wrapper hides
    the inner Anthropic protocol model and the SDK silently caps every
    reply at its 4096-token default. Pointing ``delegate`` at the wrapped
    model keeps the shipped ceiling behavior.
    """
    counted = build_counted_model(counter)
    inner = getattr(counted, "_inner", None)
    if inner is not None:
        counted.delegate = inner  # type: ignore[attr-defined]
    return counted


def run(doc: SyntheticDoc, counter: Counter) -> VariantResult:
    """Run the shipped planner + worker path over ``doc.tree`` and return what it produced."""
    configure_provider()
    counted = _counted_model(counter)
    task = task_statement()
    payload_chars = RequestBudget(window=_SPIKE_WINDOW).content_chars(CONSERVATIVE_CHARS_PER_TOKEN)

    def _use_counted(_model_string: str) -> Model:
        return counted

    def _make_decision_agent() -> TraversalAgent:
        agent = make_traversal_agent(None, model=MODEL, timeout=DECISION_TIMEOUT_S)
        if agent is None:
            msg = "decision agent disabled even with an explicit model"
            raise RuntimeError(msg)
        agent.model_builder = _use_counted
        return agent

    planner = _SpikePlanner(_make_decision_agent, counter, task=task, budget_chars=payload_chars)

    worker = make_summarizer_agent(
        MODEL,
        context_window=_SPIKE_WINDOW.max_input,
        requests_per_node=_shipped_default("summary_requests_per_node"),
    )
    worker.model_builder = _use_counted
    # Compaction inside a session would build its own sidecar models outside
    # the counter; the spike measures the session loop itself, so it stays off
    # (with no llm config it would already defer to disabled — this is explicit).
    worker.context_manager = False
    base_run = worker.run

    def _run_and_count_windows(deps: SummarizerDeps, *args, **kwargs) -> str:
        # One payload hand-off per session call: each entry the executor
        # hands the session is one request's worth of node content, never
        # cropped (the executor windows oversized nodes instead). A stalled
        # session re-hands its payload on retry, and that re-hand is counted
        # again — the same accounting as the executor's own stats.windows.
        counter.record_window(len(deps.nodes))
        return base_run(deps, *args, **kwargs)

    worker.run = _run_and_count_windows

    content_policy = SummarizerContentPolicy(
        window_chars=payload_chars,
        skip_single_child=_shipped_default("summary_skip_single_child"),
        traversal=TraversalMode.AUTO,
        traversal_min_nodes=_shipped_default("summary_traversal_min_nodes"),
    )
    session_policy = SummarizerSessionPolicy(
        batch_size=_shipped_default("summary_session_batch_size"),
        stall_retries=_shipped_default("summary_session_stall_retries"),
        payload_chars=payload_chars,
    )
    summarizer = NodeSummarizer(
        worker,
        None,
        content_policy=content_policy,
        session_policy=session_policy,
        planner=planner,
    )
    previous, counter.phase = counter.phase, "summarize"
    try:
        summarizer.summarize(doc.tree)
    finally:
        counter.phase = previous

    summaries = {node_id: node.summary for node_id, node in doc.tree.nodes.items() if getattr(node, "summary", "")}
    notes = [
        f"payload_chars={payload_chars} window_chars={payload_chars} "
        f"batch={session_policy.batch_size} stall_retries={session_policy.stall_retries} "
        f"requests_per_node={worker.requests_per_node} "
        f"traversal_min_nodes={content_policy.traversal_min_nodes}",
        f"planner calls={planner.plan_calls} planned_children={planner.planned_children} "
        f"decline_fallbacks={planner.fallbacks} work_planned={planner.work_planned} "
        f"budget_unplanned={planner.budget_unplanned}",
        "shipped budget is per level; spike enforces BUDGET_NODES=60 total: "
        "EXPAND+SUMMARIZE count as work, excess trailing entries (SUMMARIZE first) "
        "become recorded skips with reason 'budget'.",
        "windows count each payload entry handed to a session (retried payloads re-counted); "
        "no cache store; compaction inside sessions disabled; decline falls back to EXPAND-all.",
        f"decision_timeout_s={DECISION_TIMEOUT_S} (the shipped DecisionConfig.timeout is 10 s, "
        "sized for a local model; against a remote endpoint a verdict regularly needs longer, "
        "and a timeout makes the planner decline the level)",
        "the verdict is merged the way the executor merges a plan (an unmentioned child is EXPAND, "
        "an answer outside the level is ignored); the shipped strict merge is not imported (knox-3zjw)",
        "the counted model is exposed through .delegate so kompend's output-ceiling detection still sees the Anthropic protocol behind the wrapper",
        "the planner is uncached: every run is cold, every level is a fresh request",
        "the node budget is enforced across the tree, not per level (knox-s2l1): the excess is "
        "dropped from the plan, so the branch's content is inlined into its parent's summary "
        "instead of being dropped",
    ]
    return VariantResult(
        variant="a",
        summaries=summaries,
        skips=dict(planner.skips),
        counter=counter,
        notes=notes,
    )


def main() -> None:
    """Standalone run against the real synthetic tree; prints the measurements."""
    configure_provider()
    doc = build_synthetic_doc()
    counter = Counter()
    with Stopwatch(counter):
        result = run(doc, counter)
    by_kind = dict(counter.by_kind)
    print(f"variant:            {result.variant}")
    print(f"nodes in tree:      {len(doc.tree.nodes)}")
    print(f"nodes summarized:   {len(result.summaries)}")
    print(f"nodes skipped:      {len(result.skips)}")
    print(f"total requests:     {counter.requests}")
    print(f"requests by kind:   {by_kind}")
    print(f"content windows:    {counter.windows}")
    print(f"prompt tokens:      {counter.prompt_tokens}")
    print(f"completion tokens:  {counter.completion_tokens}")
    print(f"peak prompt tokens: {counter.peak_prompt}")
    print(f"wall time (s):      {counter.wall_time_s:.1f}")
    for note in result.notes:
        print(f"note: {note}")


if __name__ == "__main__":
    main()
