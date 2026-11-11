"""Element↔page assembly and boundary application for packet splitting.

Two directions live here:

* read side — group a :class:`ParsedDocument`'s elements by canonical page so a
  judge can see the packet page by page (:func:`page_texts`);
* write side — cut the *same* elements into child documents along page ranges
  (:func:`partition_document`), so every element lands in exactly one child and
  none is re-parsed.

:class:`SourceSplitter` is the object the parse stage injects. It owns the
decline rules (too few pages, unpaged source, oversized page, judge failure,
invalid plan) so fail-closed behaviour is one code path, not five call sites.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final

from kompend.core.models import DocumentElement, MetadataField, ParsedDocument
from kompend.core.models.elements import element_text

from ._plan import (
    DEFAULT_MIN_SEGMENT_PAGES,
    NeverSplit,
    PageText,
    SegmentTooShortError,
    SplitJudge,
    SplitPlanError,
    SplitReason,
    SplitRule,
    SplitSegment,
    rules_digest,
    validate_plan,
)

__all__ = [
    "LEDGER_DETAIL_CHARS",
    "SPLIT_FIELDS",
    "SourceSplitter",
    "SplitDecision",
    "child_source_path",
    "page_range",
    "page_texts",
    "partition_document",
]


#: Budget for :meth:`SplitDecision.ledger_detail`.
#:
#: :class:`~kompend.core.reporting.ledger.StageLedger` keeps 240 characters of
#: ``detail`` and cuts the rest on a word boundary. Staying under that here
#: means the boundary account is compressed by this module (which knows what
#: matters) instead of being cut mid-range by the ledger (which does not).
LEDGER_DETAIL_CHARS = 200


#: The split's provenance keys, in the order a child document stamps them.
#:
#: The vocabulary lives in :class:`~kompend.core.models.enums.MetadataField`
#: because these keys cross file boundaries — splitting writes them and figure
#: attribution reads them. This tuple is the package's ordering, so "every
#: split key" is asked once and no call site retypes a key.
SPLIT_FIELDS: Final[tuple[MetadataField, ...]] = (
    MetadataField.SPLIT_PARENT_STEM,
    MetadataField.SPLIT_PARENT_SOURCE,
    MetadataField.SPLIT_PAGES,
    MetadataField.SPLIT_CATEGORY,
    MetadataField.SPLIT_DECIDED_BY,
    MetadataField.SPLIT_RULES_SHA256,
)


@dataclass(frozen=True, slots=True)
class SplitDecision:
    """The outcome of one split attempt: what to do and why, for the ledger."""

    segments: tuple[SplitSegment, ...] = ()
    reason: SplitReason = SplitReason.NO_BOUNDARY
    detail: str = ""
    decided_by: str = ""
    rules_sha256: str = ""
    first_page: int = 0
    last_page: int = 0

    @property
    def applies(self) -> bool:
        """Whether this decision cuts the source into more than one document."""
        return self.reason is SplitReason.APPLIED and len(self.segments) > 1

    def summary(self) -> str:
        """One-line human account of the cut, for the log.

        Lists every segment, so it can exceed :data:`LEDGER_DETAIL_CHARS` —
        the ledger takes :meth:`ledger_detail` instead, which stays inside
        the column's budget by construction.
        """
        if not self.applies:
            return self.detail or self.reason.value
        cuts = ", ".join(_segment_label(segment) for segment in self.segments)
        return f"1->{len(self.segments)}: {cuts}"

    def ledger_detail(self) -> str:
        """The one form both the ledger and the log take their account from.

        ``stages.detail`` keeps 240 characters and cuts anything longer, which
        for a many-document packet would drop boundaries mid-list. So this
        method owns the compression instead: a decline reads
        ``<reason>: <why>`` cut on a word boundary, and a cut that does not fit
        names the first and last segment and counts the rest — a reader can
        still tell what was cut and where it ended, and :meth:`summary` in the
        run log always has the full list.
        """
        if not self.applies:
            text = f"{self.reason.value}: {self.detail}" if self.detail else self.reason.value
            return _cut_on_word(text, LEDGER_DETAIL_CHARS)
        full = f"1->{len(self.segments)}: {', '.join(_segment_label(segment) for segment in self.segments)}"
        if len(full) <= LEDGER_DETAIL_CHARS:
            return full
        first, last = self.segments[0], self.segments[-1]
        between = len(self.segments) - 2
        return f"1->{len(self.segments)}: {_segment_label(first)} … {_segment_label(last)} ({between} between)"


@dataclass(frozen=True, slots=True)
class SourceSplitter:
    """Decides and applies page-range splits for one source.

    Injected into the parse stage as ``splitting=SourceSplitter(...)``; the
    default injection is ``None`` (the stage never builds one), so splitting
    stays off until the config layer constructs this object. The field
    defaults mirror the planned ``[preprocessing.split]`` block.

    Attributes:
        judge: The decision-model seam. :class:`NeverSplit` is the default so
            the plumbing is exercised without a model.
        rules: Category rules handed to the judge (from the rules resource).
        min_pages: Floor below which a judgement is not worth its cost.
        min_segment_pages: Floor below which a segment is a fragment rather
            than a document. A cut that would leave either side shorter is
            refused by the plan builder; this floor is the backstop that
            catches it in any plan a judge returns.
        max_page_chars: Ceiling for one page's text. A page above it declines
            the whole split — pages are never cropped to fit.
    """

    judge: SplitJudge = field(default_factory=NeverSplit)
    rules: tuple[SplitRule, ...] = ()
    min_pages: int = 6
    min_segment_pages: int = DEFAULT_MIN_SEGMENT_PAGES
    max_page_chars: int = 12_000

    def decide(self, document: ParsedDocument) -> SplitDecision:
        """Plan a split for *document*, or explain why it declines.

        Fail-closed by construction: every decline path returns a decision with
        ``applies=False`` and a reason, which the caller records. Nothing here
        repairs a plan, invents boundaries or shortens a page.
        """
        span = page_range(document)
        if span is None:
            return SplitDecision(reason=SplitReason.UNPAGED, detail="elements carry no page numbers")
        first_page, last_page = span
        if last_page - first_page + 1 < self.min_pages:
            return SplitDecision(
                reason=SplitReason.TOO_FEW_PAGES,
                detail=f"{last_page - first_page + 1} page(s) < min_pages {self.min_pages}",
                first_page=first_page,
                last_page=last_page,
            )

        texts = page_texts(document, first_page=first_page, last_page=last_page)
        oversized = [(page.page_number, len(page.text)) for page in texts if len(page.text) > self.max_page_chars]
        if oversized:
            page_number, chars = oversized[0]
            return SplitDecision(
                reason=SplitReason.PAGE_OVERSIZED,
                detail=f"page {page_number}: {chars} chars > max_page_chars {self.max_page_chars}",
                first_page=first_page,
                last_page=last_page,
            )

        digest = rules_digest(self.rules)
        decided_by = type(self.judge).__name__
        try:
            plan = self.judge.plan_pages(texts, self.rules)
        except Exception as exc:
            # The judge is a third-party seam: whatever the model layer raises
            # must become a recorded decline, never a failed run.
            return SplitDecision(
                reason=SplitReason.JUDGE_FAILED,
                detail=f"{decided_by}: {type(exc).__name__}: {exc}",
                decided_by=decided_by,
                rules_sha256=digest,
                first_page=first_page,
                last_page=last_page,
            )
        try:
            validate_plan(plan, first_page=first_page, last_page=last_page, min_segment_pages=self.min_segment_pages)
        except SegmentTooShortError as exc:
            return SplitDecision(
                reason=SplitReason.SEGMENT_TOO_SHORT,
                detail=f"{decided_by}: {exc}",
                decided_by=decided_by,
                rules_sha256=digest,
                first_page=first_page,
                last_page=last_page,
            )
        except SplitPlanError as exc:
            return SplitDecision(
                reason=SplitReason.PLAN_INVALID,
                detail=f"{decided_by}: {exc}",
                decided_by=decided_by,
                rules_sha256=digest,
                first_page=first_page,
                last_page=last_page,
            )
        if len(plan.segments) <= 1:
            return SplitDecision(
                reason=SplitReason.NO_BOUNDARY,
                detail=f"{decided_by}: single segment {first_page}..{last_page}",
                decided_by=decided_by,
                rules_sha256=digest,
                first_page=first_page,
                last_page=last_page,
            )
        return SplitDecision(
            segments=plan.segments,
            reason=SplitReason.APPLIED,
            detail="",
            decided_by=plan.decided_by or decided_by,
            rules_sha256=plan.rules_sha256 or digest,
            first_page=first_page,
            last_page=last_page,
        )


def page_range(document: ParsedDocument) -> tuple[int, int] | None:
    """Inclusive ``(first, last)`` page range of *document*, or *None*.

    ``None`` means at least one element is unpaged, so no element can be
    attributed to a range: the source is not splittable. The range is taken
    from the elements themselves (not ``page_count``) because that is exactly
    what :func:`partition_document` can cut.
    """
    pages: list[int] = []
    for element in document.elements:
        if element.page is None:
            return None
        pages.append(element.page)
    if not pages:
        return None
    return min(pages), max(pages)


def page_texts(document: ParsedDocument, *, first_page: int, last_page: int) -> list[PageText]:
    """Full text of every page in ``first_page..last_page``, empty pages included.

    Empty pages are kept so the returned list is contiguous with the range the
    plan must partition — a page with no elements still occupies a slot.
    """
    by_page: dict[int, list[str]] = {page: [] for page in range(first_page, last_page + 1)}
    for element in document.elements:
        if element.page is None or element.page not in by_page:
            continue
        text = element_text(element.content)
        if text:
            by_page[element.page].append(text)
    return [PageText(page_number=page, text="\n\n".join(by_page[page])) for page in range(first_page, last_page + 1)]


def partition_document(
    document: ParsedDocument,
    segments: Sequence[SplitSegment],
    *,
    parent_source: Path,
    decided_by: str,
    rules_sha256: str,
) -> list[ParsedDocument]:
    """Cut *document* into one child per segment, by page.

    Calls :func:`validate_plan` itself (through the caller's decision) so a
    partition can never lose or duplicate an element: every element whose page
    falls outside all segments raises instead of vanishing.
    """
    children: list[ParsedDocument] = []
    for segment in segments:
        elements = [element for element in document.elements if element.page is not None and segment.covers(element.page)]
        child_source = child_source_path(parent_source, segment)
        metadata = copy.deepcopy(document.metadata or {})
        metadata[MetadataField.SPLIT_PARENT_STEM.value] = parent_source.stem
        metadata[MetadataField.SPLIT_PARENT_SOURCE.value] = parent_source.as_posix()
        metadata[MetadataField.SPLIT_PAGES.value] = segment.label()
        metadata[MetadataField.SPLIT_CATEGORY.value] = segment.category or ""
        metadata[MetadataField.SPLIT_DECIDED_BY.value] = decided_by
        metadata[MetadataField.SPLIT_RULES_SHA256.value] = rules_sha256
        children.append(
            ParsedDocument(
                source_file=child_source,
                elements=_rehome(elements, child_source),
                title=_child_title(document.title, segment),
                author=document.author,
                language=document.language,
                page_count=segment.end_page - segment.start_page + 1,
                metadata=metadata,
            )
        )
    attributed = sum(len(child.elements) for child in children)
    if attributed != len(document.elements):
        raise SplitPlanError(f"partition lost {len(document.elements) - attributed} element(s)")
    return children


def child_source_path(parent: Path, segment: SplitSegment) -> Path:
    """Virtual source path for one child: ``sources/packet#p005-009.pdf``.

    The page range sits *before* the suffix so ``Path.stem`` (and
    ``safe_stem``) keep it: the child's stem is ``packet#p005-009``, which makes
    its identity — and therefore its state row, parse-cache key and chunk-cache
    key — distinct from the parent's and from every sibling's. No such file
    exists on disk; the path is an identity, and the parent's bytes back it.
    """
    suffix = parent.suffix
    name = f"{parent.stem}#{segment.label()}{suffix}"
    return parent.with_name(name)


def _rehome(elements: list[DocumentElement], child_source: Path) -> list[DocumentElement]:
    """Copies of *elements* pointing at the child source, content untouched.

    The copies matter: children are cut from one parsed document, so rehoming
    the originals would repoint the parent's and its siblings' elements at
    whichever child was built last. Copying keeps ``partition_document`` a pure
    read of the parent — a caller may still use the whole document afterwards.
    Element metadata is copied too, for the same reason: a child must not be
    able to mutate its parent's element dicts.
    """
    return [replace(element, source_file=child_source, metadata=copy.deepcopy(element.metadata)) for element in elements]


def _child_title(parent_title: str, segment: SplitSegment) -> str:
    """Child title: the parent's title scoped by its page range."""
    return f"{parent_title} [{segment.label()}]" if parent_title else segment.label()


def _segment_label(segment: SplitSegment) -> str:
    """``p001-004`` plus the judge's category, when it assigned one."""
    return f"{segment.label()} {segment.category}" if segment.category else segment.label()


def _cut_on_word(text: str, limit: int) -> str:
    """*text* cut at *limit* on a word boundary, marked as cut.

    Mirrors :func:`kompend.core.reporting.ledger._excerpt` so a detail that
    this module had to shorten reads the same as one the ledger shortened.
    """
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return f"{text[: cut if cut > 0 else limit].rstrip()}…"
