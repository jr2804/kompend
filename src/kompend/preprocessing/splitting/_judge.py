"""Decision-model boundary judge for page-range splitting.

The judge answers one typed question per candidate cut point — "does a new
document begin here?" — and the segments are built from the answers, so the
model can never invent a page range it was not offered. One boolean per
candidate is also the widest verdict the decision grammar carries for a
per-page question: a list of models is refused before any request is sent, and
a list of string options collapses several verdicts into fewer answers.

One request carries every question it can: batching the pages would re-send the
state per batch and ask the same questions about the same text more than once.
The decision service does, however, refuse more than
:data:`MAX_QUESTIONS_PER_REQUEST` questions in one request
("questions must contain 1–64 fields"), so a packet with more transitions is
judged in consecutive **windows**: each window asks about a contiguous run of
transitions and sees exactly the pages those transitions sit between, in full.
A window is told that the packet continues outside it, so its first page is not
mistaken for the start of a document. Nothing is cropped to make a window fit —
the ceiling is on questions, not on page text (`splitting.max_page_chars`
handles an oversized page by declining).

The judge is a seam, not a stage: it is handed the full page texts and the
category rules, and it either returns a plan or raises. Whether that plan is
sound is decided by :func:`kompend.preprocessing.splitting.validate_plan`, and
whether a plan is worth applying at all is decided by
:class:`~kompend.preprocessing.splitting.SourceSplitter`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, Field, create_model

from kompend.agents.base import AgentSpec, build_agent
from kompend.llm.models import build_decision_model
from kompend.logging import get_logger
from kompend.preprocessing.splitting._plan import (
    DEFAULT_MIN_SEGMENT_PAGES,
    PageText,
    SplitPlan,
    SplitRule,
    SplitSegment,
    rules_digest,
)

#: ``source_id`` of every judgement: one packet is one judgement, and the id
#: names the agent in traces rather than the file (the file is in the state).
_JUDGE_SOURCE_ID: str = "split-judge"

#: Most questions the decision service accepts in one request.
#:
#: The service answers with ``questions must contain 1–64 fields``, so a packet
#: with more page transitions than this is judged in windows. The number is a
#: limit of the service, not a preference: a request above it fails, and the
#: refusal used to be silent (the source stayed whole with ``judge_failed``).
MAX_QUESTIONS_PER_REQUEST: Final[int] = 64

logger = get_logger(__name__)


@dataclass(slots=True)
class SplitJudgeDeps:
    """What the judge is asked about: the pages in view, and the rules.

    ``pages`` is rendered by :func:`render_pages` and ``rules`` by
    :func:`render_rules` — both carry the full text, because a judge that sees
    a cropped page cannot answer for the page it did not get. ``whole`` tells
    the prompt whether the pages in view are the whole packet or one window of
    a longer one, because the first page of a window continues a document that
    began before it.
    """

    source_id: str
    pages: str
    rules: str
    whole: bool = True


class SplitWindowFailed(RuntimeError):
    """One judging window failed, and the message names which.

    Wrapped rather than re-raised bare so the ledger row says where the
    judgement stopped (``window 2/2 (pages 66..113)``) instead of leaving the
    operator with ``judge_failed`` and no shape to the failure.
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
    candidate, and the plan is assembled here — under the segment floor, so a
    cut the model asks for that would leave a fragment is refused rather than
    applied.
    """

    def __init__(
        self,
        model_string: str,
        *,
        timeout: float,
        min_segment_pages: int = DEFAULT_MIN_SEGMENT_PAGES,
        max_questions: int = MAX_QUESTIONS_PER_REQUEST,
    ) -> None:
        self._model_string = model_string
        self._timeout = timeout
        self._min_segment_pages = min_segment_pages
        self._max_questions = max_questions

    @property
    def model_string(self) -> str:
        """The ``provider/model`` string this judge asks."""
        return self._model_string

    @property
    def min_segment_pages(self) -> int:
        """Fewest pages a segment may have before a cut is refused."""
        return self._min_segment_pages

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
        windows = question_windows(len(page_texts), max_questions=self._max_questions)
        whole = len(windows) == 1
        if not whole:
            logger.info(
                "Judging %d page transitions in %d window(s): the decision service takes at most %d questions per request",
                len(page_texts) - 1,
                len(windows),
                self._max_questions,
            )
        answers: dict[str, bool] = {}
        for window_number, (first, last) in enumerate(windows, start=1):
            answers.update(
                self._ask_window(
                    page_texts,
                    first_transition=first,
                    last_transition=last,
                    rules=rules,
                    whole=whole,
                    window=(window_number, len(windows)),
                ),
            )
        return plan_from_answers(
            answers,
            page_numbers=page_numbers,
            min_segment_pages=self._min_segment_pages,
            decided_by=self._model_string,
            rules_sha256=digest,
        )

    def _ask_window(
        self,
        page_texts: Sequence[PageText],
        *,
        first_transition: int,
        last_transition: int,
        rules: Sequence[SplitRule],
        whole: bool,
        window: tuple[int, int],
    ) -> dict[str, bool]:
        """Ask about transitions ``first_transition..last_transition``, in one request.

        The pages sent are exactly the pages those transitions sit between: a
        window is an excerpt of the packet, never a cropped page. Any failure is
        re-raised as :class:`SplitWindowFailed` naming the window and its page
        range, so the ledger row shows a structural failure instead of a bare
        model error.
        """
        excerpt = list(page_texts[first_transition - 1 : last_transition + 1])
        spec = AgentSpec(
            name=spec_name(len(page_texts), first_transition=first_transition),
            result_type=window_verdict_model(first_transition, last_transition),
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
            pages=render_pages(excerpt),
            rules=render_rules(rules),
            whole=whole,
        )
        try:
            verdict: Any = agent.run(deps)
        except Exception as exc:
            number, total = window
            pages = f"{excerpt[0].page_number}..{excerpt[-1].page_number}"
            msg = f"window {number}/{total} (pages {pages}): {type(exc).__name__}: {exc}"
            raise SplitWindowFailed(msg) from exc
        return {name: value for name, value in verdict.model_dump().items() if isinstance(value, bool)}


def spec_name(page_count: int, *, first_transition: int = 1) -> str:
    """Agent-spec name of a verdict over *page_count* pages from *first_transition*.

    Keyed by the question count and the first question, because the questions
    are positional: the spec's materialised class is cached per name, so a name
    that carried a page range would mint a class per packet while two packets
    of the same length ask an identical set of questions. The whole-packet name
    stays exactly what it was, so a single-request judgement keeps its cache
    key.
    """
    if first_transition == 1:
        return f"split_boundary_judge_{page_count}"
    return f"split_boundary_judge_{page_count}_from_{first_transition}"


def question_windows(page_count: int, *, max_questions: int = MAX_QUESTIONS_PER_REQUEST) -> tuple[tuple[int, int], ...]:
    """Split the transitions of a *page_count*-page packet into request windows.

    Returns inclusive ``(first_transition, last_transition)`` pairs over the
    ``page_count - 1`` transitions, in order, each covering at most
    *max_questions* of them. A packet that fits one request yields one window —
    the whole packet — so an ordinary judgement is unchanged.
    """
    if page_count < 2:
        msg = f"a split verdict needs at least two pages; got {page_count}"
        raise ValueError(msg)
    if max_questions < 1:
        msg = f"a judging window needs at least one question; got {max_questions}"
        raise ValueError(msg)
    transitions = page_count - 1
    return tuple((first, min(first + max_questions - 1, transitions)) for first in range(1, transitions + 1, max_questions))


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
    return window_verdict_model(1, page_count - 1)


def window_verdict_model(first_transition: int, last_transition: int) -> type[BaseModel]:
    """The verdict model for transitions ``first_transition..last_transition``.

    The field names carry the *global* transition index, so the answers of every
    window land in one mapping and no window can be read as another's. A
    whole-packet judgement is the single window ``1..page_count - 1``.
    """
    if first_transition < 1 or last_transition < first_transition:
        msg = f"a judging window needs 1 <= first <= last transition; got {first_transition}..{last_transition}"
        raise ValueError(msg)
    fields: dict[str, Any] = {
        boundary_field_name(index): (
            bool,
            Field(description=f"True when the page after transition {index} begins a new document rather than continuing the previous one."),
        )
        for index in range(first_transition, last_transition + 1)
    }
    return create_model(f"SplitBoundaryVerdict{first_transition}_{last_transition}", __module__=__name__, **fields)


def plan_from_verdict(
    verdict: BaseModel,
    *,
    page_numbers: Sequence[int],
    min_segment_pages: int = 1,
    decided_by: str,
    rules_sha256: str,
) -> SplitPlan:
    """Turn an answered verdict into a plan (see :func:`plan_from_answers`)."""
    return plan_from_answers(
        verdict.model_dump(),
        page_numbers=page_numbers,
        min_segment_pages=min_segment_pages,
        decided_by=decided_by,
        rules_sha256=rules_sha256,
    )


def plan_from_answers(
    answers: Mapping[str, bool],
    *,
    page_numbers: Sequence[int],
    min_segment_pages: int = 1,
    decided_by: str,
    rules_sha256: str,
) -> SplitPlan:
    """Turn answered booleans into segments covering every page exactly once.

    Transition ``i`` sits between ``page_numbers[i - 1]`` and
    ``page_numbers[i]``: a "yes" closes the running segment at the page before
    it. A transition the model left unanswered keeps its document whole rather
    than cutting it, and so does a "yes" that would leave either side below
    *min_segment_pages* — the cut is refused, not the plan, because the floor
    bounds a cut and the other boundaries are still the model's judgement.
    """
    segments: list[SplitSegment] = []
    start = page_numbers[0]
    for index in range(1, len(page_numbers)):
        if answers.get(boundary_field_name(index), False) and _cut_respects_floor(
            start=start,
            cut_after=page_numbers[index - 1],
            last_page=page_numbers[-1],
            min_segment_pages=min_segment_pages,
        ):
            segments.append(SplitSegment(start_page=start, end_page=page_numbers[index - 1]))
            start = page_numbers[index]
    segments.append(SplitSegment(start_page=start, end_page=page_numbers[-1]))
    return SplitPlan(segments=tuple(segments), decided_by=decided_by, rules_sha256=rules_sha256)


def _cut_respects_floor(*, start: int, cut_after: int, last_page: int, min_segment_pages: int) -> bool:
    """Whether cutting after *cut_after* leaves both sides at or above the floor.

    The closing side is measured from the segment's own start, the remaining
    side from the cut to the packet's last page — so a plan assembled here has
    no fragment at all, not even the tail.
    """
    return cut_after - start + 1 >= min_segment_pages and last_page - cut_after >= min_segment_pages


def boundary_field_name(index: int) -> str:
    """Field name of the *index*-th page transition (1-based).

    Positional on purpose. The page each answer belongs to is decided by
    :func:`plan_from_answers` from the ordered page list, so an answer cannot
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
    "MAX_QUESTIONS_PER_REQUEST",
    "DecisionModelSplitJudge",
    "NoDecisionModel",
    "SplitJudgeDeps",
    "SplitJudgeUnavailable",
    "SplitWindowFailed",
    "boundary_field_name",
    "boundary_model",
    "boundary_verdict_model",
    "plan_from_answers",
    "plan_from_verdict",
    "question_windows",
    "render_pages",
    "render_rules",
    "spec_name",
    "window_verdict_model",
]
