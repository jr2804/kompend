"""Split planning: the page digests a judge sees, and the partition invariant.

This module is stdlib-only on purpose: it is the seam a decision model plugs
into (see :class:`SplitJudge`), and the pipeline stage imports it on the hot
path. The page/element half lives in ``_elements``.

The one invariant that matters: a plan is a **partition** of the document's
canonical pages — every page in ``1..page_count`` appears in exactly one
segment, in order, with no gap and no overlap. A plan that breaks it raises
:class:`SplitPlanError`; it is never repaired, cropped or reinterpreted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

__all__ = [
    "NeverSplit",
    "PageText",
    "SplitJudge",
    "SplitPlan",
    "SplitPlanError",
    "SplitReason",
    "SplitRule",
    "SplitSegment",
    "rules_digest",
    "validate_plan",
]


class SplitReason(StrEnum):
    """Why a split did (or did not) happen — one value per ledger row."""

    #: The plan was applied: the source became N virtual sources.
    APPLIED = "applied"
    #: The plan is a single segment, i.e. the judge found no boundary.
    NO_BOUNDARY = "no_boundary"
    #: ``page_count`` is below the configured floor; not worth a judgement.
    TOO_FEW_PAGES = "too_few_pages"
    #: At least one element carries no page number, so nothing could be
    #: attributed to a range.
    UNPAGED = "unpaged"
    #: A single page's text exceeds the ceiling one judge call may carry.
    PAGE_OVERSIZED = "page_oversized"
    #: The judge returned a plan that violates the partition invariant.
    PLAN_INVALID = "plan_invalid"
    #: The judge raised (unavailable, timed out, unparseable output).
    JUDGE_FAILED = "judge_failed"


@dataclass(frozen=True, slots=True)
class SplitRule:
    """One document category the judge may assign, with its purpose.

    The description is what the judge reasons over, so it is authored in a
    rules file (resource), never in Python.
    """

    id: str
    description: str


@dataclass(frozen=True, slots=True)
class PageText:
    """The text of one canonical page, in full.

    ``text`` is never shortened: a page that does not fit a request window is
    a ``PAGE_OVERSIZED`` decline, not a crop.
    """

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class SplitSegment:
    """One document the packet contains: an inclusive page range."""

    start_page: int
    end_page: int
    category: str | None = None
    review_flags: tuple[str, ...] = ()

    def covers(self, page: int) -> bool:
        """Whether *page* falls inside this segment."""
        return self.start_page <= page <= self.end_page

    def label(self) -> str:
        """Compact page-range label used in logs, ledger detail and stems."""
        return f"p{self.start_page:03d}-{self.end_page:03d}"


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """A judge's answer: how the packet's pages divide into documents."""

    segments: tuple[SplitSegment, ...]
    #: Identity of the decision tier that produced this plan (``""`` for a
    #: deterministic judge such as :class:`NeverSplit`).
    decided_by: str = ""
    #: Digest of the rules the plan was decided against, for cache identity.
    rules_sha256: str = ""


class SplitPlanError(ValueError):
    """A split plan that is not a partition of the document's pages."""


class SplitJudge(Protocol):
    """The decision-model seam.

    Implementations receive the document's pages in order and the category
    rules, and return a partition of ``page_texts[0].page_number`` ..
    ``page_texts[-1].page_number``. They must not shorten ``text``; a request
    that cannot carry a page is a raised error (the caller then keeps the
    source whole), never a truncated page.

    Today only :class:`NeverSplit` implements this. The Jev/decision-model
    adapter is wired later; nothing in this package imports an agent.
    """

    def plan_pages(self, page_texts: Sequence[PageText], rules: Sequence[SplitRule]) -> SplitPlan: ...


class NeverSplit:
    """Deterministic judge: one segment spanning every page.

    The default, so the whole plumbing (plan validation, ledger rows,
    child construction) is exercised without a decision model, and so an
    explicit ``splitting`` injection with no judge behaves like today.
    """

    def plan_pages(self, page_texts: Sequence[PageText], rules: Sequence[SplitRule]) -> SplitPlan:  # noqa: ARG002, PLR6301 — protocol parity
        """Return the trivial single-segment plan."""
        if not page_texts:
            raise SplitPlanError("no pages to plan")
        return SplitPlan(
            segments=(SplitSegment(start_page=page_texts[0].page_number, end_page=page_texts[-1].page_number),),
        )


def rules_digest(rules: Sequence[SplitRule]) -> str:
    """Stable digest of *rules* — part of a child's cache identity.

    A changed rules file must invalidate previously split children, exactly
    like a changed ``[ingest]`` restriction invalidates a parse-cache entry.
    """
    payload = "\n".join(f"{rule.id}\x1f{rule.description}" for rule in rules)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_plan(plan: SplitPlan, *, first_page: int, last_page: int) -> None:
    """Raise :class:`SplitPlanError` unless *plan* partitions ``first_page..last_page``.

    Checks, in order: no empty plan, every segment inside the range and
    non-inverted, each segment starting on the page after the previous one
    ended (which rejects both gaps and overlaps), and the last segment
    reaching *last_page*. Nothing is repaired — a violating plan means the
    judge is not trustworthy for this document, and the caller keeps the
    source whole.
    """
    if not plan.segments:
        raise SplitPlanError(f"split plan has no segments (expected {first_page}..{last_page})")
    expected = first_page
    for segment in plan.segments:
        if segment.end_page < segment.start_page:
            raise SplitPlanError(f"segment {segment.label()} ends before it starts")
        if segment.start_page < first_page or segment.end_page > last_page:
            raise SplitPlanError(f"segment {segment.label()} is outside {first_page}..{last_page}")
        if segment.start_page != expected:
            clash = "overlaps" if segment.start_page < expected else "skips"
            raise SplitPlanError(f"segment {segment.label()} {clash} page {expected}")
        expected = segment.end_page + 1
    if expected != last_page + 1:
        raise SplitPlanError(f"split plan stops at page {expected - 1}, expected {last_page}")
