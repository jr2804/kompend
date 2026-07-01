"""LLM-powered lint rules — opt-in via ``--llm``.

These rules call an LLM (via :class:`LLMClient`) to detect semantic
quality issues. They are slow and expensive, so they are disabled by
default and only run when the user passes ``--llm``.
"""

from __future__ import annotations

from typing import Any

from knox.core.template_loader import get_template_loader
from knox.lint.base import BaseLintRule, LintResult, LintRuleOutcome, Severity
from knox.llm.client import JSON_RESPONSE_FORMAT
from knox.logging import get_logger

logger = get_logger(__name__)

_loader = get_template_loader()

# ── Rule 9: factual-consistency ──────────────────────────────────────────


class FactualConsistencyRule(BaseLintRule):
    """Check if claims in a page contradict source material."""

    name = "factual-consistency"
    description = "Claims in page contradict source material"

    def __init__(self, llm_client: Any | None = None, lint_page_budget: int = 100000) -> None:
        self.llm_client = llm_client
        self.lint_page_budget = lint_page_budget

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
            # Chunk-and-iterate: process content in budget-sized segments.
            content = node.content
            for seg_start in range(0, len(content), self.lint_page_budget):
                segment = content[seg_start : seg_start + self.lint_page_budget]
                prompt = _loader.render("lint/factual-consistency.md.j2", title=node.title, content=segment)
                try:
                    llm_result = self.llm_client.try_complete(
                        [{"role": "user", "content": prompt}],
                        role="lint.factual-consistency",
                        response_format=JSON_RESPONSE_FORMAT,
                    )
                except Exception as exc:
                    logger.warning("factual-consistency LLM call failed: %s", exc, exc_info=True)
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
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="No factual inconsistencies found"))
        return results

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        return list(results)


# ── Rule 10: clarity-score ──────────────────────────────────────────────


class ClarityScoreRule(BaseLintRule):
    """Rate readability, coherence, conciseness (1-10)."""

    name = "clarity-score"
    description = "Readability, coherence, conciseness score (1-10)"

    def __init__(self, llm_client: Any | None = None, lint_page_budget: int = 100000) -> None:
        self.llm_client = llm_client
        self.lint_page_budget = lint_page_budget

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
            # Chunk-and-iterate: process content in budget-sized segments,
            # keeping the best (highest) score.
            content = node.content
            best_score = 0
            best_feedback = ""
            for seg_start in range(0, len(content), self.lint_page_budget):
                segment = content[seg_start : seg_start + self.lint_page_budget]
                prompt = _loader.render("lint/clarity-score.md.j2", title=node.title, content=segment)
                try:
                    llm_result = self.llm_client.try_complete(
                        [{"role": "user", "content": prompt}],
                        role="lint.clarity-score",
                        response_format=JSON_RESPONSE_FORMAT,
                    )
                except Exception as exc:
                    logger.warning("clarity-score LLM call failed: %s", exc, exc_info=True)
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
                if score > best_score:
                    best_score = score
                    best_feedback = feedback

            if best_score > 0:
                outcome = LintRuleOutcome.PASS if best_score >= 6 else LintRuleOutcome.FAIL
                severity = Severity.INFO if best_score >= 6 else Severity.WARNING
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=outcome,
                        severity=severity,
                        message=f"Clarity score for '{node.title}': {best_score}/10 — {best_feedback}",
                        page_id=node.id,
                        details={"score": best_score, "feedback": best_feedback},
                    )
                )

        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No pages to score"))
        return results

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        return list(results)


# ── Rule 11: coverage-gap ───────────────────────────────────────────────


class CoverageGapRule(BaseLintRule):
    """Detect important concepts from sources missing in the wiki."""

    name = "coverage-gap"
    description = "Important concepts mentioned in sources but missing from wiki"

    def __init__(self, llm_client: Any | None = None, lint_page_budget: int = 100000) -> None:
        self.llm_client = llm_client
        self.lint_page_budget = lint_page_budget

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
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="Empty wiki")]

        # Build wiki summary in batches so all nodes are included.
        # Each batch produces a partial coverage check; gaps from any batch
        # are reported.
        all_gaps: list[dict] = []
        batch_size = max(1, self.lint_page_budget // max(200, len(wiki.nodes)))
        batch_size = min(batch_size, 30)  # cap to keep prompts manageable

        for batch_start in range(0, len(wiki.nodes), batch_size):
            batch = wiki.nodes[batch_start : batch_start + batch_size]
            wiki_summary = "\n".join(f"- {n.title}: {n.content or ''}" for n in batch)

            prompt = _loader.render("lint/coverage-gap.md.j2", wiki_summary=wiki_summary)

            try:
                llm_result = self.llm_client.try_complete(
                    [{"role": "user", "content": prompt}],
                    role="lint.coverage-gap",
                    response_format=JSON_RESPONSE_FORMAT,
                )
            except Exception as exc:
                logger.warning("coverage-gap LLM call failed for batch %d: %s", batch_start, exc, exc_info=True)
                continue

            if llm_result is None or llm_result.json_data is None:
                continue

            gaps = llm_result.json_data.get("gaps", [])
            if isinstance(gaps, list):
                all_gaps.extend(gaps)
        results: list[LintResult] = []
        if all_gaps:
            for gap in all_gaps:
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.WARNING,
                        message=f"{gap.get('concept', '')} — {gap.get('reason', '')}",
                        details={"concept": gap.get("concept", ""), "reason": gap.get("reason", "")},
                    )
                )
        else:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="No coverage gaps detected"))
        return results

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        return list(results)


# ── Rule 12: contradiction-detection ────────────────────────────────────


class ContradictionDetectionRule(BaseLintRule):
    """Detect pairs of wiki pages that make contradictory claims."""

    name = "contradiction-detection"
    description = "Contradictory claims across wiki pages"

    def __init__(self, llm_client: Any | None = None, lint_page_budget: int = 100000) -> None:
        self.llm_client = llm_client
        self.lint_page_budget = lint_page_budget

    def check(self, wiki: Any, *, sources_dir: str | None = None) -> list[LintResult]:
        if self.llm_client is None:
            return [
                LintResult(
                    rule=self.name,
                    outcome=LintRuleOutcome.SKIP,
                    message="No LLM client available",
                )
            ]

        if len(wiki.nodes) < 2:
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="Need at least two pages to detect contradictions")]

        results: list[LintResult] = []

        # Each batch contains multiple pages so the LLM can compare them.
        batch_size = max(2, self.lint_page_budget // 2000)
        batch_size = min(batch_size, 20)

        for batch_start in range(0, len(wiki.nodes), batch_size):
            batch = wiki.nodes[batch_start : batch_start + batch_size]
            title_to_node = {n.title.lower(): n for n in batch}
            wiki_summary = "\n\n---\n\n".join(f"Title: {n.title}\n{n.content or ''}" for n in batch)

            prompt = _loader.render("lint/contradiction-detection.md.j2", wiki_summary=wiki_summary)
            try:
                llm_result = self.llm_client.try_complete(
                    [{"role": "user", "content": prompt}],
                    role="lint.contradiction-detection",
                    response_format=JSON_RESPONSE_FORMAT,
                )
            except Exception as exc:
                logger.warning("contradiction-detection LLM call failed for batch %d: %s", batch_start, exc, exc_info=True)
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.SKIP,
                        message=f"LLM call failed: {exc}",
                    )
                )
                continue

            if llm_result is None or llm_result.json_data is None:
                continue

            contradictions = llm_result.json_data.get("contradictions", [])
            if not isinstance(contradictions, list):
                continue

            for contradiction in contradictions:
                if not isinstance(contradiction, dict):
                    continue
                page_a_title = contradiction.get("page_a", "")
                page_b_title = contradiction.get("page_b", "")
                node_a = title_to_node.get(page_a_title.lower())
                node_b = title_to_node.get(page_b_title.lower())
                if node_a is None or node_b is None:
                    continue

                severity = Severity.ERROR if str(contradiction.get("severity", "")).lower() == "error" else Severity.WARNING
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=severity,
                        message=(f"Contradiction between '{node_a.title}' and '{node_b.title}': {contradiction.get('explanation', '')}"),
                        page_id=node_a.id,
                        details={
                            "page_a_id": node_a.id,
                            "page_a_title": node_a.title,
                            "page_b_id": node_b.id,
                            "page_b_title": node_b.title,
                            "claim_a": contradiction.get("claim_a", ""),
                            "claim_b": contradiction.get("claim_b", ""),
                            "explanation": contradiction.get("explanation", ""),
                            "confidence": contradiction.get("confidence"),
                        },
                    )
                )

        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="No contradictions detected"))
        return results

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        return list(results)


# ── Registry ────────────────────────────────────────────────────────────

LLM_RULES: list[Any] = [
    FactualConsistencyRule,
    ClarityScoreRule,
    CoverageGapRule,
    ContradictionDetectionRule,
]

LLM_RULE_BY_NAME: dict[str, type] = {r.name: r for r in LLM_RULES}
