"""Contract pins for ``kompend explain`` (knox-ty8f.11).

Explain reads the per-item ledger and answers "what happened to this
file": the stage rows of the chosen run, where the time went, the
attempt's facts and every degradation with its reason — without
``--verbose``. The ledger rows are seeded through the real
:class:`~kompend.core.reporting.ledger.StageLedger` writer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kompend.cli.tree.registry import COMMANDS, CommandModule, CommandName, HelpPanel
from kompend.core.paths import VaultPaths
from kompend.core.reporting.explain import (
    AmbiguousSourceError,
    build_explain_report,
    render_explain,
)
from kompend.core.reporting.ledger import StageLedger
from kompend.core.types import ReasonCode, StageOutcome

_SOURCE = "sources/stuck.md"
_FINE = "sources/fine.md"


def test_latest_run_is_the_default_and_counts_runs(tmp_path: Path) -> None:
    report = build_explain_report(_seed(tmp_path), _SOURCE)
    assert report is not None
    assert report["run_id"] == "r2"
    assert report["runs_seen"] == 2
    assert [row["stage"] for row in report["rows"]] == ["parse"]
    assert report["rows"][0]["outcome"] == StageOutcome.CACHED.value


def test_run_selects_that_run_and_names_the_slowest_stage(tmp_path: Path) -> None:
    report = build_explain_report(_seed(tmp_path), _SOURCE, run_id="r1")
    assert report is not None
    assert [row["stage"] for row in report["rows"]] == ["parse", "chunk", "enrich"]
    assert report["slowest_stage"] == "parse"
    assert report["stage_totals_ms"]["parse"] == pytest.approx(800.0)
    assert report["total_ms"] == pytest.approx(965.0)


def test_facts_merge_first_non_null(tmp_path: Path) -> None:
    report = build_explain_report(_seed(tmp_path), _SOURCE, run_id="r1")
    assert report is not None
    assert report["facts"]["parser"] == "parsecraft"
    assert report["facts"]["judge_model"] == "gpt-test"
    assert report["facts"]["cache_key"] == "k1"


def test_degradations_carry_reason_and_detail(tmp_path: Path) -> None:
    report = build_explain_report(_seed(tmp_path), _SOURCE, run_id="r1")
    assert report is not None
    assert report["degradations"] == [
        {
            "stage": "enrich",
            "attempt": 1,
            "reason_code": ReasonCode.JUDGE_DEGRADED.value,
            "detail": "clef 400 — fell back to the rule table",
        }
    ]


def test_render_names_where_the_time_went(tmp_path: Path) -> None:
    report = build_explain_report(_seed(tmp_path), _SOURCE, run_id="r1")
    assert report is not None
    text = render_explain(report)
    assert "time went to: parse (800.0 of 965.0 ms tracked)" in text
    assert "judge_degraded" in text
    assert "fell back to the rule table" in text


def test_render_says_none_for_a_clean_run(tmp_path: Path) -> None:
    report = build_explain_report(_seed(tmp_path), _FINE)
    assert report is not None
    assert "degradations: none" in render_explain(report)


def test_file_name_and_fragment_resolve(tmp_path: Path) -> None:
    base = _seed(tmp_path)
    report = build_explain_report(base, "stuck.md")
    assert report is not None
    assert report["source"] == _SOURCE
    report = build_explain_report(base, "fine")
    assert report is not None
    assert report["source"] == _FINE


def test_ambiguous_fragment_raises_with_candidates(tmp_path: Path) -> None:
    base = _seed(tmp_path)
    with StageLedger.open(VaultPaths(base).kompend_db_file) as ledger:
        ledger.record("sources/a/stuck.md", "parse", outcome=StageOutcome.OK, run_id="r3")
        ledger.record("sources/b/stuck.md", "parse", outcome=StageOutcome.OK, run_id="r3")
    with pytest.raises(AmbiguousSourceError) as excinfo:
        build_explain_report(base, "stuck")
    assert "sources/a/stuck.md" in excinfo.value.candidates
    assert "sources/b/stuck.md" in excinfo.value.candidates
    assert _SOURCE in excinfo.value.candidates


def test_unknown_source_and_missing_db_return_none(tmp_path: Path) -> None:
    base = _seed(tmp_path)
    assert build_explain_report(base, "nope.md") is None
    assert build_explain_report(tmp_path / "empty-vault", "x.md") is None


def _seed(tmp_path: Path) -> Path:
    """A vault whose ledger saw _SOURCE in two runs and _FINE in one."""
    base = tmp_path / "vault"
    base.mkdir()
    db_path = VaultPaths(base).kompend_db_file
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with StageLedger.open(db_path) as ledger:
        ledger.record(
            _SOURCE,
            "parse",
            outcome=StageOutcome.OK,
            duration_ms=800.0,
            parser="parsecraft",
            parser_version="2026.11.7",
            backend="text",
            cache_key="k1",
            sha="abc",
            size_bytes=100,
            run_id="r1",
        )
        ledger.record(_SOURCE, "chunk", outcome=StageOutcome.OK, duration_ms=45.0, run_id="r1")
        ledger.record(
            _SOURCE,
            "enrich",
            outcome=StageOutcome.DEGRADED,
            duration_ms=120.0,
            reason_code=ReasonCode.JUDGE_DEGRADED,
            detail="clef 400 — fell back to the rule table",
            judge_model="gpt-test",
            run_id="r1",
        )
        ledger.record(_SOURCE, "parse", outcome=StageOutcome.CACHED, cache_key="k1", run_id="r2")
        ledger.record(_FINE, "parse", outcome=StageOutcome.OK, duration_ms=10.0, run_id="r1")
    return base


def test_explain_is_registered_on_the_quality_panel() -> None:
    spec = COMMANDS[CommandName.EXPLAIN]
    assert spec.module is CommandModule.EXPLAIN
    assert spec.module.value.endswith("explain_cmd")
    assert spec.panel is HelpPanel.QUALITY_INSPECTION
