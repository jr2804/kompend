"""Lint runner — orchestrates rule execution and reporting."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from knox.agents.lint import LintRuleAgent
from knox.core.config._configs import LintConfig
from knox.core.models import CompiledWiki
from knox.lint.base import BaseLintRule, LintReport, LintResult, LintRuleOutcome, Severity
from knox.lint.llm_rules import LLM_RULE_BY_NAME
from knox.lint.structural import RULE_BY_NAME, STRUCTURAL_RULES, DuplicateReferencesRule, RumdlRule
from knox.logging import get_logger

logger = get_logger(__name__)


class LintRunner:
    """Run enabled lint rules against a :class:`CompiledWiki`.

    Parameters
    ----------
    rules:
        Explicit list of rule names to run. ``None`` means all structural.
    enable_llm:
        Also run the LLM-powered rules (requires an LLM client).
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
        lint_agent: LintRuleAgent | None = None,
        lint_page_budget: int | None = None,
        fix: bool = False,
        fix_destructive: bool = False,
        sources_dir: str | None = None,
        lint_config: LintConfig | None = None,
        rumdl_config: Path | None = None,
    ) -> None:
        self.requested_rules = rules
        self.enable_llm = enable_llm
        self._lint_agent = lint_agent
        self.lint_page_budget = lint_page_budget or 100000
        self.fix = fix
        #: Destructive fixes (file deletion) require explicit opt-in — the
        #: post-compile lint and default ``knox lint check`` pass ``fix=True``
        #: but NOT this flag, so rules marked ``destructive`` only report.
        #: See p863 wiki roundup 2026-09-10: the orphan-attachments fix
        #: deleted 248 valid files because the slug check falsely flagged
        #: them and ``fix=True`` auto-ran the deletion.
        self.fix_destructive = fix_destructive
        self.sources_dir = sources_dir
        self._lint_config = lint_config
        self.rumdl_config = rumdl_config

    def run(
        self,
        wiki: CompiledWiki,
        *,
        on_rule: Callable[[str, int, int], None] | None = None,
    ) -> LintReport:
        """Execute all enabled rules and return an aggregated report.

        Parameters
        ----------
        on_rule:
            Optional callback ``(rule_name, current, total)`` called before each
            rule starts. Used by the CLI to drive a progress bar.
        """
        active_rules = self._resolve_rules()
        all_results: list[LintResult] = []
        total = len(active_rules)

        for idx, rule in enumerate(active_rules):
            if on_rule is not None:
                on_rule(rule.name, idx + 1, total)
            try:
                results = rule.check(wiki, sources_dir=self.sources_dir)
            except Exception as exc:
                logger.warning("Lint rule '%s' raised: %s", rule.name, exc, exc_info=True)
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
                    # Destructive rules (file deletion) only run when the
                    # caller explicitly opts in via ``fix_destructive``.
                    if rule.destructive and not self.fix_destructive:
                        logger.info(
                            "Rule '%s' is destructive — skipping auto-fix (pass fix_destructive to enable)",
                            rule.name,
                        )
                    else:
                        try:
                            results = rule.fix(wiki, fixable)
                        except Exception as exc:
                            logger.warning("Fix for '%s' raised: %s", rule.name, exc, exc_info=True)

            all_results.extend(results)

        return _build_report(all_results)

    def _resolve_rules(self) -> list[BaseLintRule]:
        """Build the list of rule instances to run."""
        rules: list[BaseLintRule] = []

        llm_kwargs = {"agent": self._lint_agent, "lint_page_budget": self.lint_page_budget}

        # Config-derived values for DuplicateReferencesRule / RumdlRule
        ref_kwargs: dict[str, Any] = {}
        rumdl_kwargs: dict[str, Any] = {"rumdl_config": self.rumdl_config}
        if self._lint_config is not None:
            ref_kwargs["fuzzy_threshold"] = self._lint_config.fuzzy_threshold

        def _resolve_entry(name: str) -> BaseLintRule | None:
            """Resolve a rule name to an instance, handling class-vs-singleton."""
            if name == "duplicate-references":
                return DuplicateReferencesRule(**ref_kwargs)
            if name == "rumdl":
                return RumdlRule(**rumdl_kwargs)
            entry = RULE_BY_NAME.get(name)
            if entry is None:
                return None
            if isinstance(entry, type):
                return entry()
            return entry

        if self.requested_rules:
            for name in self.requested_rules:
                if name in LLM_RULE_BY_NAME:
                    rules.append(LLM_RULE_BY_NAME[name](**llm_kwargs))
                else:
                    rule = _resolve_entry(name)
                    if rule is not None:
                        rules.append(rule)
                    else:
                        logger.warning("Unknown lint rule: %s", name)
        else:
            rules.extend(STRUCTURAL_RULES)
            # Non-singletons — instantiate with config values
            rules.append(DuplicateReferencesRule(**ref_kwargs))
            rules.append(RumdlRule(**rumdl_kwargs))

        if self.enable_llm:
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
    return json.dumps(report_to_dict(report), indent=2, default=str)


def report_to_dict(report: LintReport) -> dict[str, Any]:
    """Serialize a lint report to a plain dict (the ``lint_history`` shape)."""
    return {
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


def report_to_summary(report: LintReport) -> str:
    """Compact one-line summary for pipeline/CLI status output."""
    return f"Lint: {report.health_score:.0f}% health ({report.passes} pass / {report.failures} fail / {report.skips} skip)"


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
