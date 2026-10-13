"""Tests for :mod:`kompend.lint.gap_analysis`."""

from __future__ import annotations

from kompend.lint.base import LintReport, LintResult, LintRuleOutcome
from kompend.lint.gap_analysis import (
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
    results = [LintResult(rule="broken-wikilinks", outcome=LintRuleOutcome.PASS, page_id="n1", message="ok")]
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
    """Alias surfaces at HIGH; a 2+-referenced missing topic too (weighted)."""
    results = [_lr("P.863", alias_of="polqa-..."), _lr("Other"), _lr("Other")]
    recs = _analyze_broken_wikilinks(results)
    by_finding = {r.finding: r.priority for r in recs}
    assert all(v == GapPriority.HIGH for k, v in by_finding.items() if k.startswith("'P.863'"))
    # 2 refs without degree info → conservative MEDIUM.
    assert by_finding["'Other' (referenced by 2 pages)"] == GapPriority.MEDIUM


def test_missing_topic_weighting() -> None:
    """Coverage weighting: 3+ refs HIGH, 2 refs MEDIUM, 1 leaf ref LOW; hub ref upgrades."""
    results = [_lr(f"T{i}") for i in range(3)]  # T0 x1, T1 x1, T2 x1 — need custom counts
    # Build explicit counts: T0 x3 (HIGH), T1 x2 (MEDIUM), T2 x1 (LOW)
    results = [_lr("T0"), _lr("T0"), _lr("T0"), _lr("T1"), _lr("T1"), _lr("T2")]
    recs = {r.finding: r for r in _analyze_broken_wikilinks(results)}
    assert recs["'T0' (referenced by 3 pages)"].priority == GapPriority.HIGH
    assert recs["'T1' (referenced by 2 pages)"].priority == GapPriority.MEDIUM
    assert recs["'T2'"].priority == GapPriority.LOW


def test_missing_topic_hub_ref_upgrades_priority() -> None:
    """A single reference from a hub page (degree above median) is MEDIUM."""
    degree = {"n1": 20, "n2": 1, "n3": 1, "n4": 2, "n5": 1}
    results = [_lr("CoreHole", page_id="n1"), _lr("Fringe", page_id="n2")]
    recs = {r.finding: r for r in _analyze_broken_wikilinks(results, degree)}
    assert recs["'CoreHole'"].priority == GapPriority.MEDIUM
    assert recs["'Fringe'"].priority == GapPriority.LOW


def _lr(target: str, alias_of: str | None = None, page_id: str = "n1") -> LintResult:
    """Build a failing broken-wikilinks LintResult."""
    details: dict[str, str] = {"target": target}
    if alias_of is not None:
        details["alias_of"] = alias_of
    return LintResult(
        rule="broken-wikilinks",
        outcome=LintRuleOutcome.FAIL,
        page_id=page_id,
        message=f"Broken wikilink [[{target}]]",
        details=details,
    )
