"""LLM-powered lint rules — opt-in via ``--llm``.

These rules call an LLM (via :class:`LLMClient`) to detect semantic
quality issues. They are slow and expensive, so they are disabled by
default and only run when the user passes ``--llm``.
"""

from __future__ import annotations

import json
from typing import Any

from slugify import slugify

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

        # Cross-batch dedup: merge concepts that refer to the same thing
        # under different names (e.g., "ETSI TS 103 558" = "ABLE Model").
        if len(all_gaps) > 1:
            all_gaps = self._dedup_gaps(all_gaps)

        # Suppress false positives: the batched LLM can't see pages outside
        # its own batch, so it flags concepts whose page already exists. Drop
        # any gap matching the full wiki page inventory (deterministic, no LLM).
        if all_gaps:
            all_gaps = self._suppress_existing_gaps(all_gaps, wiki)

        results: list[LintResult] = []
        if all_gaps:
            for gap in all_gaps:
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.WARNING,
                        message=f"{gap.get('concept', '')} — {gap.get('reason', '')}",
                        details={
                            "concept": gap.get("concept", ""),
                            "reason": gap.get("reason", ""),
                            "aliases": gap.get("aliases", []),
                        },
                    )
                )
        else:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="No coverage gaps detected"))
        return results

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:  # noqa: PLR6301
        return list(results)

    def _dedup_gaps(self, gaps: list[dict]) -> list[dict]:
        """Merge semantically equivalent concepts via a single LLM pass.

        Falls back to the original list on any error — never loses gaps.
        Chunks if the list exceeds the page budget, then merges chunk results.
        """
        if len(gaps) <= 1:
            return gaps
        gaps_json = json.dumps(gaps, ensure_ascii=False)
        if len(gaps_json) <= self.lint_page_budget:
            return self._dedup_gaps_batch(gaps)
        chunk_size = max(10, len(gaps) * self.lint_page_budget // (len(gaps_json) + 1))
        merged: list[dict] = []
        for i in range(0, len(gaps), chunk_size):
            merged.extend(self._dedup_gaps_batch(gaps[i : i + chunk_size]))
        if len(merged) > 1:
            return self._dedup_gaps_batch(merged)
        return merged

    def _dedup_gaps_batch(self, gaps: list[dict]) -> list[dict]:
        """Single LLM call to merge a gap list that fits in the context window."""
        if len(gaps) <= 1:
            return gaps
        gaps_json = json.dumps(gaps, ensure_ascii=False, indent=2)
        prompt = _loader.render("lint/coverage-gap-dedup.md.j2", gaps_json=gaps_json)
        try:
            llm_result = self.llm_client.try_complete(
                [{"role": "user", "content": prompt}],
                role="lint.coverage-gap-dedup",
                response_format=JSON_RESPONSE_FORMAT,
            )
        except Exception as exc:
            logger.warning("coverage-gap dedup failed: %s", exc, exc_info=True)
            return gaps
        if llm_result is None or llm_result.json_data is None:
            return gaps
        merged = llm_result.json_data.get("gaps", [])
        return merged if isinstance(merged, list) and merged else gaps

    @staticmethod
    def _suppress_existing_gaps(gaps: list[dict], wiki: Any) -> list[dict]:
        """Drop gaps whose concept already has a wiki page.

        The detection LLM sees only one batch at a time, so it flags any
        concept referenced in batch *N* whose page lives in batch *M*. This
        deterministic pass checks every surviving gap against the full wiki
        inventory — every node id, slugified title, and raw title — and drops
        matches. Gap ``aliases`` are checked too.
        """
        existing: set[str] = set()
        for node in wiki.nodes:
            existing.add(str(node.id).lower().strip())
            title = (getattr(node, "title", "") or "").lower().strip()
            if title:
                existing.add(title)
                existing.add(slugify(title))
        kept: list[dict] = []
        for gap in gaps:
            names = [gap.get("concept", ""), *(gap.get("aliases") or [])]
            if any(n and (slugify(n) in existing or n.lower().strip() in existing) for n in names):
                continue
            kept.append(gap)
        suppressed = len(gaps) - len(kept)
        if suppressed:
            logger.info("coverage-gap suppression dropped %d false positive(s) matching existing pages", suppressed)
        return kept


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
        seen_pairs: set[frozenset[str]] = set()

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

                # Deduplicate: skip if this page pair was already reported.
                pair_key = frozenset({node_a.id, node_b.id})
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

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
