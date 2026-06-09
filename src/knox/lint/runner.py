"""Lint runner — orchestrates rule execution and reporting."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from knox.lint.base import LintResult, LintRuleOutcome, Severity
from knox.lint.llm_rules import LLM_RULE_BY_NAME
from knox.lint.structural import RULE_BY_NAME, STRUCTURAL_RULES

logger = logging.getLogger(__name__)


@dataclass
class LintReport:
    """Aggregated lint report for a compiled wiki.

    Attributes:
    ----------
    results:
        All individual findings.
    total_checks:
        Number of distinct rules that ran.
    failures:
        Count of FAIL outcomes.
    passes:
        Count of PASS outcomes.
    skips:
        Count of SKIP outcomes.
    health_score:
        0-100 score: ``100 * passes / max(total_checks, 1)``.
    """

    results: list[LintResult] = field(default_factory=list)
    total_checks: int = 0
    failures: int = 0
    passes: int = 0
    skips: int = 0
    health_score: float = 0.0


class LintRunner:
    """Run enabled lint rules against a :class:`CompiledWiki`.

    Parameters
    ----------
    rules:
        Explicit list of rule names to run. ``None`` means all structural.
    enable_llm:
        Also run the 3 LLM-powered rules (requires an LLM client).
    llm_client:
        An :class:`LLMClient` instance for LLM-powered rules.
    fix:
        Attempt auto-fix for supported rules.
    sources_dir:
        Path to sources directory (for file-based checks).
    """

    def __init__(
        self,
        *,
        rules: list[str] | None = None,
        enable_llm: bool = False,
        llm_client: Any | None = None,
        lint_page_budget: int | None = None,
        fix: bool = False,
        sources_dir: str | None = None,
    ) -> None:
        self.requested_rules = rules
        self.enable_llm = enable_llm
        self.llm_client = llm_client
        self.lint_page_budget = lint_page_budget or 100000
        self.fix = fix
        self.sources_dir = sources_dir

    def run(self, wiki: Any) -> LintReport:
        """Execute all enabled rules and return an aggregated report."""
        active_rules = self._resolve_rules()
        all_results: list[LintResult] = []

        for rule in active_rules:
            try:
                results = rule.check(wiki, sources_dir=self.sources_dir)
            except Exception as exc:
                logger.warning("Lint rule '%s' raised: %s", rule.name, exc)
                results = [
                    LintResult(
                        rule=rule.name,
                        outcome=LintRuleOutcome.SKIP,
                        message=f"Rule error: {exc}",
                    )
                ]

            if self.fix:
                fixable = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
                if fixable:
                    try:
                        results = rule.fix(wiki, fixable)
                    except Exception as exc:
                        logger.warning("Fix for '%s' raised: %s", rule.name, exc)

            all_results.extend(results)

        return _build_report(all_results)

    def _resolve_rules(self) -> list[Any]:
        """Build the list of rule instances to run."""
        rules: list[Any] = []

        llm_kwargs = {"llm_client": self.llm_client, "lint_page_budget": self.lint_page_budget}

        if self.requested_rules:
            for name in self.requested_rules:
                if name in RULE_BY_NAME:
                    rules.append(RULE_BY_NAME[name])
                elif name in LLM_RULE_BY_NAME:
                    rules.append(LLM_RULE_BY_NAME[name](**llm_kwargs))
                else:
                    logger.warning("Unknown lint rule: %s", name)
        else:
            rules.extend(STRUCTURAL_RULES)

        if self.enable_llm or (self.requested_rules and any(n in LLM_RULE_BY_NAME for n in self.requested_rules)):
            # Add LLM rules that haven't already been added
            existing_names = {r.name for r in rules}
            for name, cls in LLM_RULE_BY_NAME.items():
                if name not in existing_names:
                    rules.append(cls(**llm_kwargs))

        return rules


def _build_report(results: list[LintResult]) -> LintReport:
    """Compute aggregate stats from a flat list of results.

    Health score is severity-weighted:
      - ERROR failures cost 3 penalty points
      - WARNING failures cost 2 penalty points
      - INFO failures cost 1 penalty point
      - PASS or SKIP results cost nothing

    ``health_score = 100 * max(0, 1 - penalty / max_possible)``
    where ``max_possible = total_checks * 3`` (worst case: all errors).
    """
    failures = sum(1 for r in results if r.outcome == LintRuleOutcome.FAIL)
    passes = sum(1 for r in results if r.outcome == LintRuleOutcome.PASS)
    skips = sum(1 for r in results if r.outcome == LintRuleOutcome.SKIP)
    total = failures + passes + skips

    # Empty result set → zero score (nothing to measure)
    if total == 0:
        return LintReport(
            results=results,
            total_checks=0,
            failures=0,
            passes=0,
            skips=0,
            health_score=0.0,
        )

    # Weighted penalty: ERROR=3, WARNING=2, INFO=1
    penalty = 0
    for r in results:
        if r.outcome != LintRuleOutcome.FAIL:
            continue
        if r.severity == Severity.ERROR:
            penalty += 3
        elif r.severity == Severity.WARNING:
            penalty += 2
        else:
            penalty += 1

    max_possible = total * 3
    health_score = 100.0 * max(0.0, 1.0 - penalty / max_possible)

    return LintReport(
        results=results,
        total_checks=total,
        failures=failures,
        passes=passes,
        skips=skips,
        health_score=round(health_score, 1),
    )


def report_to_json(report: LintReport) -> str:
    """Serialize a lint report to a JSON string."""
    data = {
        "health_score": report.health_score,
        "total_checks": report.total_checks,
        "failures": report.failures,
        "passes": report.passes,
        "skips": report.skips,
        "results": [
            {
                "rule": r.rule,
                "outcome": r.outcome.value,
                "severity": r.severity.value,
                "message": r.message,
                "page_id": r.page_id,
                "fix_applied": r.fix_applied,
                "details": r.details,
            }
            for r in report.results
        ],
    }
    return json.dumps(data, indent=2, default=str)


def report_to_text(report: LintReport) -> str:
    """Format a lint report as a human-readable table."""
    lines: list[str] = []
    lines.append(f"Lint Report — Health Score: {report.health_score:.0f}%")
    lines.append(f"  Checks: {report.total_checks}  |  Pass: {report.passes}  |  Fail: {report.failures}  |  Skip: {report.skips}")
    lines.append("")

    failures = [r for r in report.results if r.outcome == LintRuleOutcome.FAIL]
    if failures:
        lines.append("Failures:")
        for r in failures:
            fix_tag = " [FIXED]" if r.fix_applied else ""
            page = f" (page: {r.page_id})" if r.page_id else ""
            lines.append(f"  [{r.severity.value.upper()}] {r.rule}: {r.message}{page}{fix_tag}")
        lines.append("")

    passes = [r for r in report.results if r.outcome == LintRuleOutcome.PASS]
    if passes:
        lines.append("Passed:")
        for r in passes:
            lines.append(f"  ✓ {r.rule}: {r.message}")

    return "\n".join(lines)
