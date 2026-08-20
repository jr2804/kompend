"""Tests for :mod:`knox.lint.gap_analysis`."""

from __future__ import annotations

from knox.lint.base import LintReport, LintReportResult
from knox.lint.gap_analysis import (
    GapPriority,
    _analyze_broken_wikilinks,
    analyze_gaps,
    gaps_to_text,
)


def test_alias_mismatch_yields_alias_resolution() -> None:
    """When the broken-wikilinks rule tagged the target with ``alias_of``,
    the gap recommendation says so and points at the existing page.
    """
    results = [_lr("P.863", alias_of="polqa-...")]
    recs = _analyze_broken_wikilinks(results)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.category == "Alias mismatch"
    assert rec.resolution == "alias"
    assert "polqa-..." in rec.finding
    assert "alias map" in rec.action.lower() or "prompt" in rec.action.lower()


def test_genuine_missing_topic_yields_missing_resolution() -> None:
    results = [_lr("No Such Topic XYZ")]
    recs = _analyze_broken_wikilinks(results)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.category == "Missing topic"
    assert rec.resolution == "missing"
    assert "source" in rec.action.lower()


def test_alias_and_missing_mixed() -> None:
    results = [_lr("P.863", alias_of="polqa-..."), _lr("Genuinely Missing")]
    recs = _analyze_broken_wikilinks(results)
    by_resolution = {r.resolution for r in recs}
    assert by_resolution == {"alias", "missing"}


def test_count_aggregated() -> None:
    """Same target referenced from multiple pages is counted once."""
    results = [_lr("P.863", alias_of="polqa-..."), _lr("P.863", alias_of="polqa-..."), _lr("P.863", alias_of="polqa-...")]
    recs = _analyze_broken_wikilinks(results)
    assert len(recs) == 1
    assert "3 pages" in recs[0].finding


def test_source_reference_skipped() -> None:
    """``sources/foo.pdf`` citations are provenance, not broken page links."""
    results = [_lr("sources/foo.pdf")]
    recs = _analyze_broken_wikilinks(results)
    assert recs == []


def test_pass_results_skipped() -> None:
    """Only failing broken-wikilinks become recommendations."""
    results = [LintReportResult(rule="broken-wikilinks", outcome="pass", page_id="n1", message="ok")]
    recs = _analyze_broken_wikilinks(results)
    assert recs == []


def test_gaps_to_text_groups_by_priority() -> None:
    """The text formatter groups recommendations by [HIGH]/[MEDIUM]/[LOW]."""
    report = LintReport(
        results=[
            _lr("P.863", alias_of="polqa-..."),
            _lr("Genuine Missing Topic"),
        ],
    )
    recs = analyze_gaps(report)
    text = gaps_to_text(recs)
    assert "[HIGH]" in text
    assert "Alias mismatch" in text
    assert "Missing topic" in text


def test_gaps_to_text_empty_report() -> None:
    text = gaps_to_text([])
    assert "No knowledge gaps" in text


def test_priority_field_is_high_for_broken_wikilinks() -> None:
    """Alias and missing both surface at HIGH priority — both need attention."""
    results = [_lr("P.863", alias_of="polqa-..."), _lr("Other")]
    recs = _analyze_broken_wikilinks(results)
    assert all(r.priority == GapPriority.HIGH for r in recs)


def _lr(target: str, alias_of: str | None = None) -> LintReportResult:
    """Build a failing broken-wikilinks LintReportResult."""
    details: dict[str, str] = {"target": target}
    if alias_of is not None:
        details["alias_of"] = alias_of
    return LintReportResult(
        rule="broken-wikilinks",
        outcome="fail",
        page_id="n1",
        message=f"Broken wikilink [[{target}]]",
        details=details,
    )
