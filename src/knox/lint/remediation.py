"""Remediate lint findings — auto-fix, LLM-generate, or recommend.

Three tiers of remediation:

**Tier 1 — Auto-fixable (no LLM):** Deterministic fixes applied directly to
wiki content. Currently handles image embed syntax errors (``[[x.png]]`` →
``![[x.png]]``).

**Tier 2 — LLM-remediable:** Generates new content using cached extractions
as source material. Currently handles coverage gaps by generating concept
pages from phase-1 extraction data.

**Tier 3 — Recommendations:** Issues that need new sources or manual
intervention. Reported but not actioned.
"""

from __future__ import annotations

from pathlib import Path

from knox.agents.remediate import RemediationAgent
from knox.core.models import CompiledWiki
from knox.lint._remediation_gaps import remediate_coverage_gaps
from knox.lint.base import LintReport, LintReportResult, RemediationAction, RemediationStatus
from knox.lint.gap_analysis import analyze_gaps
from knox.logging import get_logger

logger = get_logger(__name__)

_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".bmp",
        ".tiff",
    }
)

_TIER3_CATEGORIES: frozenset[str] = frozenset(
    {
        "Missing topic",
        "Unresolved contradiction",
        "Empty page",
        "Poor content quality",
        "Isolated topic",
        "Coverage gap",
    }
)


# ── Output formatting ──────────────────────────────────────────────────


_TIER_LABELS: dict[int, str] = {
    1: "Tier 1 - Auto-fixed",
    2: "Tier 2 - LLM-generated",
    3: "Tier 3 - Recommendations",
}

_STATUS_ICONS: dict[RemediationStatus, str] = {
    RemediationStatus.APPLIED: "+",
    RemediationStatus.SKIPPED: "~",
    RemediationStatus.RECOMMENDED: "!",
}


def remediate(
    wiki: CompiledWiki,
    report: LintReport,
    *,
    base_dir: Path,
    output_dir: Path | None = None,
    agent: RemediationAgent | None = None,
    dry_run: bool = False,
) -> list[RemediationAction]:
    """Remediate lint findings from a :class:`LintReport`.

    Args:
        wiki: The compiled wiki to modify in-place (tier 1/2).
        report: Typed lint report (the ``report`` sub-dict from
            ``lint_history.jsonl``).
        base_dir: Project root. Required for tier 2 (drives ``sources_dir`` /
            cache resolution via :class:`VaultPaths`).
        output_dir: Wiki output directory. Tier-2 generated concept pages are
            written here as ``.md`` files so they are visible to the user and
            survive a re-organize. When ``None``, pages only update the cache.
        agent: :class:`RemediationAgent` used for tier-2 content generation.
            When ``None``, tier 2 is skipped (only tier 1 auto-fixes and
            tier 3 recommendations run).
        dry_run: If ``True``, report what would be done without modifying.

    Returns a list of :class:`RemediationAction` sorted by tier.
    """
    results: list[LintReportResult] = report.results
    actions: list[RemediationAction] = []

    # ── Tier 1: Auto-fixable ──
    actions.extend(_fix_image_wikilinks(wiki, results, dry_run=dry_run))

    # ── Tier 2: LLM-remediable ──
    handled_concepts: set[str] = set()
    tier2_actions: list[RemediationAction] = []
    if agent is not None:
        tier2_actions = remediate_coverage_gaps(
            wiki,
            results,
            base_dir=base_dir,
            output_dir=output_dir,
            agent=agent,
            dry_run=dry_run,
        )
    actions.extend(tier2_actions)
    # Track which concepts tier 2 looked at (any status) to exclude from tier 3
    for a in tier2_actions:
        concept = a.details.get("concept")
        if concept:
            handled_concepts.add(concept.lower().strip())
        matched = a.details.get("matched_concept")
        if matched:
            handled_concepts.add(matched.lower().strip())

    # ── Tier 3: Recommendations ──
    for rec in analyze_gaps(report):
        if rec.category not in _TIER3_CATEGORIES:
            continue
        # Skip image-extension false positives already handled in tier 1
        finding_lower = rec.finding.lower()
        if any(ext in finding_lower for ext in _IMAGE_EXTENSIONS):
            continue
        # Skip coverage gaps already handled by tier 2
        if rec.category == "Coverage gap" and any(hc in finding_lower for hc in handled_concepts):
            continue
        actions.append(
            RemediationAction(
                tier=3,
                category=rec.category,
                description=rec.finding,
                status=RemediationStatus.RECOMMENDED,
                details={"action": rec.action, "priority": rec.priority},
            )
        )

    actions.sort(key=lambda a: a.tier)
    return actions


# ── Tier 1: Auto-fixable ───────────────────────────────────────────────


def _fix_image_wikilinks(
    wiki: CompiledWiki,
    results: list[LintReportResult],
    *,
    dry_run: bool = False,
) -> list[RemediationAction]:
    """Fix ``[[image.png]]`` → ``![[image.png]]`` (wrong embed syntax)."""
    image_targets: dict[str, list[str]] = {}
    for r in results:
        if r.rule != "broken-wikilinks" or r.outcome != "fail":
            continue
        target = r.details.get("target", "")
        if target and Path(target).suffix.lower() in _IMAGE_EXTENSIONS:
            image_targets.setdefault(r.page_id, []).append(target)

    if not image_targets:
        return []

    actions: list[RemediationAction] = []
    for node in wiki.nodes:
        targets = image_targets.get(node.id)
        if not targets:
            continue
        fixed = 0
        for target in targets:
            old_syntax = f"[[{target}]]"
            if old_syntax in (node.content or ""):
                if not dry_run:
                    node.content = node.content.replace(old_syntax, f"![[{target}]]")
                fixed += 1
        if fixed:
            actions.append(
                RemediationAction(
                    tier=1,
                    category="Image embed syntax",
                    description=f"Fixed {fixed} image wikilink(s) in '{node.title}'",
                    status=RemediationStatus.APPLIED if not dry_run else RemediationStatus.SKIPPED,
                    details={"targets": targets},
                )
            )
    return actions


def remediation_to_text(actions: list[RemediationAction]) -> str:
    """Format remediation actions as a human-readable report."""
    if not actions:
        return "No remediation actions needed."

    lines: list[str] = [f"Remediation Report ({len(actions)} action{'s' if len(actions) != 1 else ''})"]

    current_tier = 0
    for action in actions:
        if action.tier != current_tier:
            current_tier = action.tier
            lines.append(f"\n[{_TIER_LABELS.get(action.tier, f'Tier {action.tier}')}]")

        icon = _STATUS_ICONS.get(action.status, "")
        lines.append(f"  {icon} {action.category}: {action.description}")

    applied = sum(1 for a in actions if a.status == RemediationStatus.APPLIED)
    skipped = sum(1 for a in actions if a.status == RemediationStatus.SKIPPED)
    recommended = sum(1 for a in actions if a.status == RemediationStatus.RECOMMENDED)
    lines.append("")
    lines.append(f"Summary: {applied} applied, {skipped} skipped, {recommended} recommended")

    return "\n".join(lines)
