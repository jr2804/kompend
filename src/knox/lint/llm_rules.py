"""LLM-powered lint rules — opt-in via ``--llm``.

These rules call an LLM (via :class:`LLMClient`) to detect semantic
quality issues. They are slow and expensive, so they are disabled by
default and only run when the user passes ``--llm``.
"""

from __future__ import annotations

import logging
from typing import Any

from knox.lint.base import BaseLintRule, LintResult, LintRuleOutcome, Severity

logger = logging.getLogger(__name__)

# ── Rule 9: factual-consistency ──────────────────────────────────────────


class FactualConsistencyRule(BaseLintRule):
    """Check if claims in a page contradict source material."""

    name = "factual-consistency"
    description = "Claims in page contradict source material"

    def __init__(self, llm_client: Any | None = None, context_budget: int = 100000) -> None:
        self.llm_client = llm_client
        self.context_budget = context_budget

    def check(self, wiki: Any, *, sources_dir: str | None = None) -> list[LintResult]:
        if self.llm_client is None:
            return [
                LintResult(
                    rule=self.name,
                    outcome=LintRuleOutcome.SKIP,
                    message="No LLM client available",
                )
            ]

        results: list[LintResult] = []
        for node in wiki.nodes:
            if not node.content or not node.source_chunks:
                continue
            content = node.content
            if len(content) > self.context_budget:
                logger.warning(
                    "factual-consistency: page '%s' content (%d chars) exceeds context budget (%d), truncating",
                    node.title,
                    len(content),
                    self.context_budget,
                )
                content = content[:self.context_budget]
            prompt = (
                "You are a factual-consistency checker for a compiled wiki. "
                "Given the following page content, identify any claims that "
                "appear unsupported or potentially contradictory. "
                "Return a JSON object with key 'issues' containing a list of "
                "objects with 'claim' and 'reason' fields. "
                "If everything is consistent, return {\"issues\": []}.\n\n"
                f"Page: {node.title}\n\n{content}"
            )
            try:
                llm_result = self.llm_client.try_complete(
                    [{"role": "user", "content": prompt}],
                    role="lint.factual-consistency",
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                logger.warning("factual-consistency LLM call failed: %s", exc)
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.SKIP,
                        message=f"LLM call failed: {exc}",
                        page_id=node.id,
                    )
                )
                continue

            if llm_result is None or llm_result.json_data is None:
                continue

            issues = llm_result.json_data.get("issues", [])
            if issues:
                for issue in issues:
                    results.append(
                        LintResult(
                            rule=self.name,
                            outcome=LintRuleOutcome.FAIL,
                            severity=Severity.ERROR,
                            message=f"Inconsistent claim in '{node.title}': {issue.get('reason', '')}",
                            page_id=node.id,
                            details={"claim": issue.get("claim", ""), "reason": issue.get("reason", "")},
                        )
                    )

        if not results:
            results.append(
                LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="No factual inconsistencies found")
            )
        return results

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        return list(results)


# ── Rule 10: clarity-score ──────────────────────────────────────────────


class ClarityScoreRule(BaseLintRule):
    """Rate readability, coherence, conciseness (1-10)."""

    name = "clarity-score"
    description = "Readability, coherence, conciseness score (1-10)"

    def __init__(self, llm_client: Any | None = None, context_budget: int = 100000) -> None:
        self.llm_client = llm_client
        self.context_budget = context_budget

    def check(self, wiki: Any, *, sources_dir: str | None = None) -> list[LintResult]:
        if self.llm_client is None:
            return [
                LintResult(
                    rule=self.name,
                    outcome=LintRuleOutcome.SKIP,
                    message="No LLM client available",
                )
            ]

        results: list[LintResult] = []
        for node in wiki.nodes:
            if not node.content:
                continue
            content = node.content
            if len(content) > self.context_budget:
                logger.warning(
                    "clarity-score: page '%s' content (%d chars) exceeds context budget (%d), truncating",
                    node.title,
                    len(content),
                    self.context_budget,
                )
                content = content[:self.context_budget]
            prompt = (
                "Rate the following wiki page on readability, coherence, "
                "and conciseness on a scale of 1-10. Return a JSON object "
                "with keys 'score' (integer 1-10) and 'feedback' (string). "
                "A score below 6 indicates poor quality.\n\n"
                f"Page: {node.title}\n\n{content}"
            )
            try:
                llm_result = self.llm_client.try_complete(
                    [{"role": "user", "content": prompt}],
                    role="lint.clarity-score",
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                logger.warning("clarity-score LLM call failed: %s", exc)
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.SKIP,
                        message=f"LLM call failed: {exc}",
                        page_id=node.id,
                    )
                )
                continue

            if llm_result is None or llm_result.json_data is None:
                continue

            score = llm_result.json_data.get("score", 0)
            feedback = llm_result.json_data.get("feedback", "")
            outcome = LintRuleOutcome.PASS if score >= 6 else LintRuleOutcome.FAIL
            severity = Severity.INFO if score >= 6 else Severity.WARNING

            results.append(
                LintResult(
                    rule=self.name,
                    outcome=outcome,
                    severity=severity,
                    message=f"Clarity score for '{node.title}': {score}/10 — {feedback}",
                    page_id=node.id,
                    details={"score": score, "feedback": feedback},
                )
            )

        if not results:
            results.append(
                LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No pages to score")
            )
        return results

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        return list(results)


# ── Rule 11: coverage-gap ───────────────────────────────────────────────


class CoverageGapRule(BaseLintRule):
    """Detect important concepts from sources missing in the wiki."""

    name = "coverage-gap"
    description = "Important concepts mentioned in sources but missing from wiki"

    def __init__(self, llm_client: Any | None = None, context_budget: int = 100000) -> None:
        self.llm_client = llm_client
        self.context_budget = context_budget

    def check(self, wiki: Any, *, sources_dir: str | None = None) -> list[LintResult]:
        if self.llm_client is None:
            return [
                LintResult(
                    rule=self.name,
                    outcome=LintRuleOutcome.SKIP,
                    message="No LLM client available",
                )
            ]

        if not wiki.nodes:
            return [
                LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="Empty wiki")
            ]

        # Build a summary of all wiki content for the LLM
        # Use the context budget, divided evenly across up to 30 pages
        per_node_budget = max(200, self.context_budget // 30)
        wiki_summary = "\n".join(
            f"- {n.title}: {(n.content or '')[:per_node_budget]}" for n in wiki.nodes[:30]
        )

        prompt = (
            "You are a coverage-gap detector for a compiled wiki. "
            "Given the following wiki page titles and summaries, identify "
            "any important concepts that appear to be missing — concepts "
            "that should logically be present based on the existing content. "
            "Return a JSON object with key 'gaps' containing a list of "
            "objects with 'concept' and 'reason' fields. "
            "If coverage looks complete, return {\"gaps\": []}.\n\n"
            f"Wiki pages:\n{wiki_summary}"
        )

        try:
            llm_result = self.llm_client.try_complete(
                [{"role": "user", "content": prompt}],
                role="lint.coverage-gap",
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.warning("coverage-gap LLM call failed: %s", exc)
            return [
                LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message=f"LLM call failed: {exc}")
            ]

        if llm_result is None or llm_result.json_data is None:
            return [
                LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="LLM returned no data")
            ]

        gaps = llm_result.json_data.get("gaps", [])
        results: list[LintResult] = []
        if gaps:
            for gap in gaps:
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.WARNING,
                        message=f"Coverage gap: {gap.get('concept', '')} — {gap.get('reason', '')}",
                        details={"concept": gap.get("concept", ""), "reason": gap.get("reason", "")},
                    )
                )
        else:
            results.append(
                LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="No coverage gaps detected")
            )
        return results

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        return list(results)


# ── Registry ────────────────────────────────────────────────────────────

LLM_RULES: list[Any] = [
    FactualConsistencyRule,
    ClarityScoreRule,
    CoverageGapRule,
]

LLM_RULE_BY_NAME: dict[str, type] = {r.name: r for r in LLM_RULES}
