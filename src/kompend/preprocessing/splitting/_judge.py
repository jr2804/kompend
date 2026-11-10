"""Decision-model boundary judge for page-range splitting.

The judge answers one typed question per candidate cut point — "does a new
document begin here?" — and the segments are built from the answers, so the
model can never invent a page range it was not offered. One boolean per
candidate is also the widest verdict the decision grammar carries for a
per-page question: a list of models is refused before any request is sent, and
a list of string options collapses several verdicts into fewer answers.

The whole packet is one request, because a decision model takes one state:
batching the pages would re-send the state per batch and ask the same
questions about the same text more than once.

The judge is a seam, not a stage: it is handed the full page texts and the
category rules, and it either returns a plan or raises. Whether that plan is
sound is decided by :func:`kompend.preprocessing.splitting.validate_plan`, and
whether a plan is worth applying at all is decided by
:class:`~kompend.preprocessing.splitting.SourceSplitter`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, create_model

from kompend.agents.base import AgentSpec, build_agent
from kompend.llm.models import build_decision_model
from kompend.preprocessing.splitting._plan import (
    PageText,
    SplitPlan,
    SplitRule,
    SplitSegment,
    rules_digest,
)

#: ``source_id`` of every judgement: one packet is one judgement, and the id
#: names the agent in traces rather than the file (the file is in the state).
_JUDGE_SOURCE_ID: str = "split-judge"


@dataclass(slots=True)
class SplitJudgeDeps:
    """What the judge is asked about: every page of the packet, and the rules.

    ``pages`` is rendered by :func:`render_pages` and ``rules`` by
    :func:`render_rules` — both carry the full text, because a judge that sees
    a cropped page cannot answer for the page it did not get.
    """

    source_id: str
    pages: str
    rules: str


class SplitJudgeUnavailable(RuntimeError):
    """The judge has no decision model to ask, so it cannot decide.

    Raised, never worked around: the splitter turns it into a recorded decline
    and the source is kept whole. An invented boundary would fragment a real
    document, which is worse than not splitting.
    """


class NoDecisionModel:
    """Judge that declines because no decision model is configured.

    Deliberately not a heuristic: the splitter stays wired, so the run records
    "no split, and here is why" instead of silently behaving as if splitting
    were off.
    """

    def plan_pages(self, page_texts: Sequence[PageText], rules: Sequence[SplitRule]) -> SplitPlan:  # noqa: PLR6301 — a judge is an object with state (a model, a timeout)
        """Refuse to plan a split."""
        del page_texts, rules
        raise SplitJudgeUnavailable("no decision model configured: set splitting.judge_model, or enable [decision] with a model")


class DecisionModelSplitJudge:
    """Ask a decision model whether each page begins a new document.

    The candidate cut points come from the page structure (the pages the
    parser produced), never from the model: the model answers one boolean per
    candidate, and the plan is assembled here.
    """

    def __init__(self, model_string: str, *, timeout: float) -> None:
        self._model_string = model_string
        self._timeout = timeout

    @property
    def model_string(self) -> str:
        """The ``provider/model`` string this judge asks."""
        return self._model_string

    def plan_pages(self, page_texts: Sequence[PageText], rules: Sequence[SplitRule]) -> SplitPlan:
        """Return the plan the model implies; raises when it cannot answer."""
        page_numbers = [page.page_number for page in page_texts]
        digest = rules_digest(rules)
        if len(page_texts) < 2:
            return SplitPlan(
                segments=(SplitSegment(start_page=page_numbers[0], end_page=page_numbers[0]),),
                decided_by=self._model_string,
                rules_sha256=digest,
            )
        spec = AgentSpec(
            name=spec_name(len(page_texts)),
            result_type=boundary_verdict_model(len(page_texts)),
            deps_type=SplitJudgeDeps,
            template_name="split_boundary.md.j2",
            system_template_name="split_boundary_system.md.j2",
            template_prefix="decision/",
            temperature=0.0,
            model_builder=build_decision_model,
        )
        agent = build_agent(spec, model_string=self._model_string, timeout=self._timeout, retries=0)
        deps = SplitJudgeDeps(
            source_id=_JUDGE_SOURCE_ID,
            pages=render_pages(page_texts),
            rules=render_rules(rules),
        )
        verdict: Any = agent.run(deps)
        return plan_from_verdict(verdict, page_numbers=page_numbers, decided_by=self._model_string, rules_sha256=digest)


def spec_name(page_count: int) -> str:
    """Agent-spec name of a verdict over *page_count* pages.

    Keyed by the question count, because the questions are positional: the
    spec's materialised class is cached per name, so a name that carried a
    page range would mint a class per packet while two packets of the same
    length ask an identical set of questions.
    """
    return f"split_boundary_judge_{page_count}"


def boundary_model(page_numbers: Sequence[int]) -> type[BaseModel]:
    """The verdict model for a packet whose pages are *page_numbers*.

    The first page is not a candidate — it starts the packet — so ``N`` pages
    ask ``N - 1`` questions; the page numbers themselves never appear in the
    verdict, because the answers are matched positionally.
    """
    return boundary_verdict_model(len(page_numbers))


def boundary_verdict_model(page_count: int) -> type[BaseModel]:
    """The verdict model a judge over *page_count* pages is asked for.

    Public and stable so a diagnostic tool can check the shape against the
    decision grammar without building a judge or contacting a model: the verdict
    carries one boolean per candidate cut point, and the candidate cut points
    are the ``page_count - 1`` transitions between consecutive pages.

    Raises:
        ValueError: when *page_count* leaves nothing to decide (fewer than two
            pages has no boundary, so a judge is never asked about it).
    """
    if page_count < 2:
        msg = f"a split verdict needs at least two pages; got {page_count}"
        raise ValueError(msg)
    fields: dict[str, Any] = {
        boundary_field_name(index): (
            bool,
            Field(description=f"True when the page after transition {index} begins a new document rather than continuing the previous one."),
        )
        for index in range(1, page_count)
    }
    return create_model(f"SplitBoundaryVerdict{page_count}", __module__=__name__, **fields)


def plan_from_verdict(
    verdict: BaseModel,
    *,
    page_numbers: Sequence[int],
    decided_by: str,
    rules_sha256: str,
) -> SplitPlan:
    """Turn the answered booleans into segments covering every page exactly once.

    Transition ``i`` sits between ``page_numbers[i - 1]`` and
    ``page_numbers[i]``: a "yes" closes the running segment at the page before
    it. A transition the model left unanswered keeps its document whole rather
    than cutting it.
    """
    answered = verdict.model_dump()
    segments: list[SplitSegment] = []
    start = page_numbers[0]
    for index in range(1, len(page_numbers)):
        if answered.get(boundary_field_name(index), False):
            segments.append(SplitSegment(start_page=start, end_page=page_numbers[index - 1]))
            start = page_numbers[index]
    segments.append(SplitSegment(start_page=start, end_page=page_numbers[-1]))
    return SplitPlan(segments=tuple(segments), decided_by=decided_by, rules_sha256=rules_sha256)


def boundary_field_name(index: int) -> str:
    """Field name of the *index*-th page transition (1-based).

    Positional on purpose. The page each answer belongs to is decided by
    :func:`plan_from_verdict` from the ordered page list, so an answer cannot
    be attached to the wrong boundary through an echoed page number.
    """
    return f"boundary_{index:03d}"


def render_pages(page_texts: Sequence[PageText]) -> str:
    """Render every page, full text, under a page marker the questions refer to."""
    return "\n\n".join(f"<<< page {page.page_number} >>>\n{page.text}" for page in page_texts)


def render_rules(rules: Sequence[SplitRule]) -> str:
    """Render the category rules the judge may see documents as."""
    if not rules:
        return "(no categories configured)"
    return "\n".join(f"- {rule.id}: {rule.description}" for rule in rules)


__all__ = [
    "DecisionModelSplitJudge",
    "NoDecisionModel",
    "SplitJudgeDeps",
    "SplitJudgeUnavailable",
    "boundary_field_name",
    "boundary_model",
    "boundary_verdict_model",
    "plan_from_verdict",
    "render_pages",
    "render_rules",
    "spec_name",
]
