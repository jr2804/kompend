"""Remediate lint findings — LLM-generate missing pages.

Generates concept pages for coverage-gap findings using cached phase-1
extractions as source material. Recommendations (what needs new sources or
manual review) live in ``kompend lint gaps`` — see :mod:`kompend.lint.gap_analysis`.
"""

from __future__ import annotations

from pathlib import Path

from kompend.agents.remediate import AgenticRemediationAgent, RemediationAgent
from kompend.core.models import CompiledWiki
from kompend.lint._remediation_gaps import remediate_coverage_gaps
from kompend.lint.base import LintReport, LintResult, RemediationAction, RemediationStatus
from kompend.logging import get_logger

# ── Output formatting ──────────────────────────────────────────────────


_STATUS_ICONS: dict[RemediationStatus, str] = {
    RemediationStatus.APPLIED: "+",
    RemediationStatus.SKIPPED: "~",
    RemediationStatus.RECOMMENDED: "!",
}

logger = get_logger(__name__)


def remediate(
    wiki: CompiledWiki,
    report: LintReport,
    *,
    base_dir: Path,
    output_dir: Path | None = None,
    agent: RemediationAgent | AgenticRemediationAgent | None = None,
    dry_run: bool = False,
) -> list[RemediationAction]:
    """Generate concept pages for coverage-gap findings in *report*.

    Args:
        wiki: The compiled wiki to modify in-place.
        report: Typed lint report (the ``report`` sub-dict from
            ``lint_history.jsonl``).
        base_dir: Project root. Drives ``sources_dir`` / cache resolution via
            :class:`VaultPaths`.
        output_dir: Wiki output directory. Generated concept pages are
            written here as ``.md`` files so they are visible to the user and
            survive a re-organize. When ``None``, pages only update the cache.
        agent: Agent used for content generation. Accepts either the legacy
            :class:`RemediationAgent` (single-shot, one LLM call per concept)
            or :class:`AgenticRemediationAgent` (single multi-turn session over
            all gaps). When ``None``, nothing is generated.
        dry_run: If ``True``, report what would be done without modifying.

    Returns a list of :class:`RemediationAction`.
    """
    results: list[LintResult] = report.results

    if agent is None:
        return []

    return remediate_coverage_gaps(
        wiki,
        results,
        base_dir=base_dir,
        output_dir=output_dir,
        agent=agent,
        dry_run=dry_run,
    )


def remediation_to_text(actions: list[RemediationAction]) -> str:
    """Format remediation actions as a human-readable report."""
    if not actions:
        return "No remediation actions needed."

    lines: list[str] = [f"Remediation Report ({len(actions)} action{'s' if len(actions) != 1 else ''})"]

    for action in actions:
        icon = _STATUS_ICONS.get(action.status, "")
        lines.append(f"  {icon} {action.category}: {action.description}")

    applied = sum(1 for a in actions if a.status == RemediationStatus.APPLIED)
    skipped = sum(1 for a in actions if a.status == RemediationStatus.SKIPPED)
    recommended = sum(1 for a in actions if a.status == RemediationStatus.RECOMMENDED)
    lines.append("")
    lines.append(f"Summary: {applied} applied, {skipped} skipped, {recommended} recommended")

    return "\n".join(lines)
