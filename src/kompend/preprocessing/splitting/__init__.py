"""Page-range splitting of a parsed document into independent child sources.

A packet (a standard plus amendments, a scanned volume of several papers) is
parsed as one file, so its pages are one source downstream: one chunk tree, one
generation pass, one concept neighbourhood. This package cuts such a source
into per-document virtual sources along page ranges decided by a decision
model, so each becomes a source in its own right.

The seam a model plugs into is :class:`SplitJudge` (``_plan``): page digests in,
:class:`SplitPlan` out. The default judge is :class:`NeverSplit`, so the whole
plumbing runs without a model and splitting is off until config constructs a
:class:`SourceSplitter` with a real judge.

Fail-closed: a judge error, an oversized page, an unpaged source or a plan that
is not a partition leaves the source whole and records why. Nothing here
invents boundaries from heuristics, and no page is ever cropped to fit a
request window.
"""

from __future__ import annotations

from ._elements import (
    LEDGER_DETAIL_CHARS,
    SourceSplitter,
    SplitDecision,
    SplitMetadata,
    child_source_path,
    page_range,
    page_texts,
    partition_document,
)
from ._plan import (
    NeverSplit,
    PageText,
    SplitJudge,
    SplitPlan,
    SplitPlanError,
    SplitReason,
    SplitRule,
    SplitSegment,
    rules_digest,
    validate_plan,
)

__all__ = [
    "LEDGER_DETAIL_CHARS",
    "NeverSplit",
    "PageText",
    "SourceSplitter",
    "SplitDecision",
    "SplitJudge",
    "SplitMetadata",
    "SplitPlan",
    "SplitPlanError",
    "SplitReason",
    "SplitRule",
    "SplitSegment",
    "child_source_path",
    "page_range",
    "page_texts",
    "partition_document",
    "rules_digest",
    "validate_plan",
]
