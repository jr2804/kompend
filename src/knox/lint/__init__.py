"""KNOX compile lint — structural and LLM-powered quality rules."""

from knox.lint.base import BaseLintRule, LintResult, LintRuleOutcome, Severity
from knox.lint.gap_analysis import GapRecommendation, analyze_gaps, gaps_to_text
from knox.lint.history import (
    entry_from_report,
    load_lint_history,
    prune_lint_history,
    save_lint_entry,
)
from knox.lint.runner import LintRunner
from knox.lint.structural import STRUCTURAL_RULES

__all__ = [
    "STRUCTURAL_RULES",
    "BaseLintRule",
    "GapRecommendation",
    "LintResult",
    "LintRuleOutcome",
    "LintRunner",
    "Severity",
    "analyze_gaps",
    "entry_from_report",
    "gaps_to_text",
    "load_lint_history",
    "prune_lint_history",
    "save_lint_entry",
]
