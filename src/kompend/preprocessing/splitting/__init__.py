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
    SourceSplitter,
    SplitDecision,
    SplitMetadata,
    child_source_path,
    page_range,
    page_texts,
    partition_document,
)
from ._judge import (
    DecisionModelSplitJudge,
    NoDecisionModel,
    SplitJudgeDeps,
    SplitJudgeUnavailable,
    boundary_field_name,
    boundary_model,
    boundary_verdict_model,
    plan_from_verdict,
    render_pages,
    render_rules,
    spec_name,
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
from ._provenance import attachment_owner_stem, parent_source_of
from ._rules import BUILTIN_RULES_TEMPLATE, SplitRulesError, load_rules, parse_rules

__all__ = [
    "BUILTIN_RULES_TEMPLATE",
    "LEDGER_DETAIL_CHARS",
    "DecisionModelSplitJudge",
    "NeverSplit",
    "NoDecisionModel",
    "PageText",
    "SourceSplitter",
    "SplitDecision",
    "SplitJudge",
    "SplitJudgeDeps",
    "SplitJudgeUnavailable",
    "SplitMetadata",
    "SplitPlan",
    "SplitPlanError",
    "SplitReason",
    "SplitRule",
    "SplitRulesError",
    "SplitSegment",
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
    "plan_from_verdict",
    "render_pages",
    "render_rules",
    "resolve_judge_model",
    "resolve_rules_file",
    "rules_digest",
    "spec_name",
    "validate_plan",
]
