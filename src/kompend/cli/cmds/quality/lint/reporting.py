"""Terminal rendering of a lint report, shared by the ``check`` and ``semantic`` passes."""

from __future__ import annotations

from kompend.core.config._configs import LintOutput
from kompend.core.paths import VaultPaths
from kompend.lint.base import LintReport
from kompend.lint.runner import report_to_compact, report_to_json, report_to_summary, report_to_text
from kompend.logging import get_console
from kompend.output.pages.log import log_stage

__all__ = ["print_report"]


def print_report(report: LintReport, json_output: bool, vp: VaultPaths, level: LintOutput = LintOutput.QUIET) -> None:
    """Print *report* at the requested level and record the score in the stage log."""
    if json_output:
        get_console().print(report_to_json(report))
    elif level is LintOutput.FULL:
        get_console().print(report_to_text(report))
    elif level is LintOutput.SUMMARY:
        get_console().print(report_to_compact(report))
    else:
        get_console().print(report_to_summary(report))
    log_stage(
        vp,
        action="Lint",
        detail=f"Health score {report.health_score:.1f}% ({report.passes}/{report.total_checks} checks passed)",
    )
