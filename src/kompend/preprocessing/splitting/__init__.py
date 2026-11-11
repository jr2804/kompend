"""Page-range splitting of a parsed document into independent child sources.

A packet (a standard plus amendments, a scanned volume of several papers) is
parsed as one file, so its pages are one source downstream: one chunk tree, one
generation pass, one concept neighbourhood. This package cuts such a source
into per-document virtual sources along page ranges decided by a decision
model, so each becomes a source in its own right.

The seam a model plugs into is :class:`SplitJudge` (``_plan``): page text in,
:class:`SplitPlan` out. The default judge is :class:`NeverSplit`, so the whole
plumbing runs without a model and splitting is off until config constructs a
:class:`SourceSplitter` with a real judge — :func:`build_splitter` (``_build``)
is the one place that turns settings into a judge.

Fail-closed: a judge error, an oversized page, an unpaged source or a plan that
is not a partition leaves the source whole and records why. Nothing here
invents boundaries from heuristics, and no page is ever cropped to fit a
request window.
"""

from __future__ import annotations

from ._build import build_splitter, resolve_judge_model, resolve_rules_file
from ._elements import (
    LEDGER_DETAIL_CHARS,
    SPLIT_FIELDS,
    SourceSplitter,
    SplitDecision,
    child_source_path,
    page_range,
    page_texts,
    partition_document,
)
from ._judge import (
    MAX_QUESTIONS_PER_REQUEST,
    DecisionModelSplitJudge,
    NoDecisionModel,
    SplitJudgeDeps,
    SplitJudgeUnavailable,
    SplitWindowFailed,
    boundary_field_name,
    boundary_model,
    boundary_verdict_model,
    plan_from_answers,
    plan_from_verdict,
    question_windows,
    render_pages,
    render_rules,
    spec_name,
    window_verdict_model,
)
from ._plan import (
    DEFAULT_MIN_SEGMENT_PAGES,
    MIN_SEGMENT_PAGES,
    NeverSplit,
    PageText,
    SegmentTooShortError,
    SplitJudge,
    SplitPlan,
    SplitPlanError,
    SplitReason,
    SplitRule,
    SplitSegment,
    rules_digest,
    segment_length,
    validate_plan,
)
from ._provenance import attachment_owner_stem, parent_source_of, split_keys
from ._rules import BUILTIN_RULES_TEMPLATE, SplitRulesError, load_rules, parse_rules

__all__ = [
    "BUILTIN_RULES_TEMPLATE",
    "DEFAULT_MIN_SEGMENT_PAGES",
    "LEDGER_DETAIL_CHARS",
    "MAX_QUESTIONS_PER_REQUEST",
    "MIN_SEGMENT_PAGES",
    "SPLIT_FIELDS",
    "DecisionModelSplitJudge",
    "NeverSplit",
    "NoDecisionModel",
    "PageText",
    "SegmentTooShortError",
    "SourceSplitter",
    "SplitDecision",
    "SplitJudge",
    "SplitJudgeDeps",
    "SplitJudgeUnavailable",
    "SplitPlan",
    "SplitPlanError",
    "SplitReason",
    "SplitRule",
    "SplitRulesError",
    "SplitSegment",
    "SplitWindowFailed",
    "attachment_owner_stem",
    "boundary_field_name",
    "boundary_model",
    "boundary_verdict_model",
    "build_splitter",
    "child_source_path",
    "load_rules",
    "page_range",
    "page_texts",
    "parent_source_of",
    "parse_rules",
    "partition_document",
    "plan_from_answers",
    "plan_from_verdict",
    "question_windows",
    "render_pages",
    "render_rules",
    "resolve_judge_model",
    "resolve_rules_file",
    "rules_digest",
    "segment_length",
    "spec_name",
    "split_keys",
    "validate_plan",
    "window_verdict_model",
]
