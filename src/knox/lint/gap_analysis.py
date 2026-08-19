"""Analyze lint results to identify knowledge gaps and recommend sources.

Extracts actionable signals from a lint report dict (as stored in
``lint_history.jsonl``) and produces prioritized recommendations for
which source documents should be added or improved.

Gap signals by rule:

- ``broken-wikilinks`` → topics referenced but not covered (add sources)
- ``coverage-gap``     → concepts from sources missing in wiki (add/improve sources)
- ``clarity-score``    → pages with poor content quality (check source completeness)
- ``empty-page``       → pages with no content (source parsing may have failed)
- ``orphan-page``      → isolated topics (add related sources to build connections)
- ``contradiction-detection`` → conflicting claims (add authoritative source)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from knox.core.config import DEFAULTS
from knox.core.config.keys import DISPLAY_LINT_FEEDBACK_CHARS
from knox.lint.base import LintReport, LintReportResult

_CLARITY_LOW_THRESHOLD = 4
_CLARITY_MODERATE_THRESHOLD = 6


class GapPriority(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class GapRecommendation:
    """A single actionable recommendation derived from lint findings."""

    priority: GapPriority
    category: str
    finding: str
    action: str


def analyze_gaps(report: LintReport) -> list[GapRecommendation]:
    """Extract knowledge-gap recommendations from a typed lint report.

    Accepts a :class:`LintReport` (the ``report`` sub-dict as stored in
    ``lint_history.jsonl``, wrapped by :func:`knox.lint.remediation.remediate`).
    """
    results: list[LintReportResult] = report.results
    recs: list[GapRecommendation] = []

    recs.extend(_analyze_broken_wikilinks(results))
    recs.extend(_analyze_coverage_gaps(results))
    recs.extend(_analyze_clarity(results))
    recs.extend(_analyze_empty_pages(results))
    recs.extend(_analyze_orphans(results))
    recs.extend(_analyze_contradictions(results))

    priority_order = {GapPriority.HIGH: 0, GapPriority.MEDIUM: 1, GapPriority.LOW: 2}
    recs.sort(key=lambda r: priority_order.get(r.priority, 3))
    return recs


def gaps_to_text(recs: list[GapRecommendation]) -> str:
    """Format recommendations as a human-readable report."""
    if not recs:
        return "No knowledge gaps detected — all lint checks passed."

    lines: list[str] = [f"Source Gap Analysis ({len(recs)} recommendation{'s' if len(recs) != 1 else ''})"]

    current_priority = ""
    for rec in recs:
        if rec.priority != current_priority:
            current_priority = rec.priority
            label = {GapPriority.HIGH: "HIGH", GapPriority.MEDIUM: "MEDIUM", GapPriority.LOW: "LOW"}.get(rec.priority, rec.priority.upper())
            lines.append(f"\n[{label}]")

        lines.append(f"  {rec.category}: {rec.finding}")
        lines.append(f"    -> {rec.action}")

    return "\n".join(lines)


# ── Per-rule analyzers ───────────────────────────────────────────────────


def _analyze_broken_wikilinks(results: list[LintReportResult]) -> list[GapRecommendation]:
    """Broken wikilinks point to topics the wiki doesn't cover yet."""
    targets: list[str] = []
    for r in results:
        if r.rule != "broken-wikilinks" or r.outcome != "fail":
            continue
        target = r.details.get("target", "")
        if target and not _is_source_reference(target):
            targets.append(target)

    recs: list[GapRecommendation] = []
    for target, count in Counter(targets).most_common():
        pages = f" (referenced by {count} page{'s' if count != 1 else ''})" if count > 1 else ""
        recs.append(
            GapRecommendation(
                priority=GapPriority.HIGH,
                category="Missing topic",
                finding=f"'{target}'{pages}",
                action="Add a source document covering this topic.",
            )
        )
    return recs


def _is_source_reference(target: str) -> bool:
    """Return True if the target is a source-file citation (e.g., sources/foo.pdf)."""
    if target.startswith("sources/"):
        return True
    lower = target.lower()
    return any(lower.endswith(ext) for ext in (".pdf", ".docx", ".doc", ".xlsx", ".pptx"))


def _analyze_coverage_gaps(results: list[LintReportResult]) -> list[GapRecommendation]:
    """LLM-identified concepts missing from the wiki."""
    recs: list[GapRecommendation] = []
    for r in results:
        if r.rule != "coverage-gap" or r.outcome != "fail":
            continue
        details = r.details
        concept = details.get("concept", r.message)
        reason = details.get("reason", "")
        finding = f"'{concept}'"
        if reason:
            finding += f" — {reason}"
        recs.append(
            GapRecommendation(
                priority=GapPriority.MEDIUM,
                category="Coverage gap",
                finding=finding,
                action="Add or improve source material covering this concept.",
            )
        )
    return recs


def _analyze_clarity(results: list[LintReportResult]) -> list[GapRecommendation]:
    """Low clarity scores indicate incomplete or poorly extracted content."""
    recs: list[GapRecommendation] = []
    for r in results:
        if r.rule != "clarity-score" or r.outcome != "fail":
            continue
        details = r.details
        score = details.get("score")
        if not isinstance(score, int | float):
            continue
        page_id = r.page_id or "unknown"
        feedback = details.get("feedback", r.message)

        if score < _CLARITY_LOW_THRESHOLD:
            priority = GapPriority.HIGH
            action = "Source may be incomplete or parsing failed — check the original document."
        elif score < _CLARITY_MODERATE_THRESHOLD:
            priority = GapPriority.MEDIUM
            action = "Content lacks depth or clarity — consider adding supplementary sources."
        else:
            continue

        finding = f"Page '{page_id}' scored {score}/10"
        if feedback:
            feedback_limit = DEFAULTS.get(DISPLAY_LINT_FEEDBACK_CHARS, 200)
            finding += f": {feedback[:feedback_limit]}"
        recs.append(
            GapRecommendation(
                priority=priority,
                category="Poor content quality",
                finding=finding,
                action=action,
            )
        )
    return recs


def _analyze_empty_pages(results: list[LintReportResult]) -> list[GapRecommendation]:
    """Empty pages indicate source parsing failures or empty sources."""
    recs: list[GapRecommendation] = []
    for r in results:
        if r.rule != "empty-page" or r.outcome != "fail":
            continue
        page_id = r.page_id or "unknown"
        recs.append(
            GapRecommendation(
                priority=GapPriority.HIGH,
                category="Empty page",
                finding=f"Page '{page_id}' has no content",
                action="Source file may be empty or unparseable — verify and re-ingest.",
            )
        )
    return recs


def _analyze_orphans(results: list[LintReportResult]) -> list[GapRecommendation]:
    """Orphan pages need related sources to build connections."""
    orphan_ids = [r.page_id for r in results if r.rule == "orphan-page" and r.outcome == "fail"]
    recs: list[GapRecommendation] = []
    for page_id in orphan_ids:
        recs.append(
            GapRecommendation(
                priority=GapPriority.LOW,
                category="Isolated topic",
                finding=f"Page '{page_id}' has no incoming links",
                action="Add sources related to this topic to build cross-references.",
            )
        )
    return recs


def _analyze_contradictions(results: list[LintReportResult]) -> list[GapRecommendation]:
    """Contradictions may need an authoritative source to resolve."""
    recs: list[GapRecommendation] = []
    for r in results:
        if r.rule != "contradiction-detection" or r.outcome != "fail":
            continue
        details = r.details
        page_a = details.get("page_a_title", r.page_id or "unknown")
        page_b = details.get("page_b_title", "unknown")
        recs.append(
            GapRecommendation(
                priority=GapPriority.MEDIUM,
                category="Unresolved contradiction",
                finding=f"'{page_a}' vs '{page_b}'",
                action="Add an authoritative source to resolve the conflicting claims.",
            )
        )
    return recs
