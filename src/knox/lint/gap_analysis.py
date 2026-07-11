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
from typing import Any

from knox.core.config import DEFAULTS
from knox.core.config.keys import DISPLAY_LINT_FEEDBACK_CHARS

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


def analyze_gaps(report_dict: dict[str, Any]) -> list[GapRecommendation]:
    """Extract knowledge-gap recommendations from a lint report dict.

    Accepts the ``report`` sub-dict as stored in ``lint_history.jsonl``
    (i.e. the output of :func:`knox.lint.runner.report_to_dict`).
    """
    results: list[dict[str, Any]] = report_dict.get("results", [])
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


def _analyze_broken_wikilinks(results: list[dict[str, Any]]) -> list[GapRecommendation]:
    """Broken wikilinks point to topics the wiki doesn't cover yet."""
    targets: list[str] = []
    for r in results:
        if r.get("rule") != "broken-wikilinks" or r.get("outcome") != "fail":
            continue
        target = r.get("details", {}).get("target", "")
        if target:
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


def _analyze_coverage_gaps(results: list[dict[str, Any]]) -> list[GapRecommendation]:
    """LLM-identified concepts missing from the wiki."""
    recs: list[GapRecommendation] = []
    for r in results:
        if r.get("rule") != "coverage-gap" or r.get("outcome") != "fail":
            continue
        details = r.get("details", {})
        concept = details.get("concept", r.get("message", ""))
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


def _analyze_clarity(results: list[dict[str, Any]]) -> list[GapRecommendation]:
    """Low clarity scores indicate incomplete or poorly extracted content."""
    recs: list[GapRecommendation] = []
    for r in results:
        if r.get("rule") != "clarity-score" or r.get("outcome") != "fail":
            continue
        details = r.get("details", {})
        score = details.get("score")
        if not isinstance(score, int | float):
            continue
        page_id = r.get("page_id", "unknown")
        feedback = details.get("feedback", r.get("message", ""))

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


def _analyze_empty_pages(results: list[dict[str, Any]]) -> list[GapRecommendation]:
    """Empty pages indicate source parsing failures or empty sources."""
    recs: list[GapRecommendation] = []
    for r in results:
        if r.get("rule") != "empty-page" or r.get("outcome") != "fail":
            continue
        page_id = r.get("page_id", "unknown")
        recs.append(
            GapRecommendation(
                priority=GapPriority.HIGH,
                category="Empty page",
                finding=f"Page '{page_id}' has no content",
                action="Source file may be empty or unparseable — verify and re-ingest.",
            )
        )
    return recs


def _analyze_orphans(results: list[dict[str, Any]]) -> list[GapRecommendation]:
    """Orphan pages need related sources to build connections."""
    orphan_ids = [r.get("page_id", "") for r in results if r.get("rule") == "orphan-page" and r.get("outcome") == "fail"]
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


def _analyze_contradictions(results: list[dict[str, Any]]) -> list[GapRecommendation]:
    """Contradictions may need an authoritative source to resolve."""
    recs: list[GapRecommendation] = []
    for r in results:
        if r.get("rule") != "contradiction-detection" or r.get("outcome") != "fail":
            continue
        details = r.get("details", {})
        page_a = details.get("page_a_title", r.get("page_id", "?"))
        page_b = details.get("page_b_title", "?")
        recs.append(
            GapRecommendation(
                priority=GapPriority.MEDIUM,
                category="Unresolved contradiction",
                finding=f"'{page_a}' vs '{page_b}'",
                action="Add an authoritative source to resolve the conflicting claims.",
            )
        )
    return recs
