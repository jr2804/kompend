"""kompend compile lint — structural and LLM-powered quality rules."""

from kompend.lint.base import BaseLintRule, LintResult, LintRuleOutcome, RemediationAction, RemediationStatus, Severity
from kompend.lint.gap_analysis import GapRecommendation, analyze_gaps, gaps_to_text
from kompend.lint.history import (
    LintEntry,
    entry_from_report,
    load_lint_history,
    prune_lint_history,
    save_lint_entry,
)
from kompend.lint.remediation import remediate, remediation_to_text
from kompend.lint.runner import LintRunner
from kompend.lint.structural import STRUCTURAL_RULES

__all__ = [
    "STRUCTURAL_RULES",
    "BaseLintRule",
    "GapRecommendation",
    "LintEntry",
    "LintResult",
    "LintRuleOutcome",
    "LintRunner",
    "RemediationAction",
    "RemediationStatus",
    "Severity",
    "analyze_gaps",
    "entry_from_report",
    "gaps_to_text",
    "load_lint_history",
    "prune_lint_history",
    "remediate",
    "remediation_to_text",
    "save_lint_entry",
]
