"""Tests for the doctor's last-run report."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from kompend.core.paths import VaultPaths
from kompend.core.reporting.last_run import build_last_run_report, render_last_run
from kompend.core.reporting.ledger import StageLedger
from kompend.core.store import open_store
from kompend.core.types import ReasonCode, StageOutcome


def test_empty_vault_has_no_runs(tmp_path: Path) -> None:
    report = build_last_run_report(tmp_path)
    assert not report.has_runs
    assert report.run_id is None
    assert "no ledger rows" in " ".join(render_last_run(report))


def test_counts_and_cache_hit_rate(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        [
            {"source": "a.md", "stage": "parse", "outcome": StageOutcome.OK, "duration_ms": 10.0},
            {"source": "b.md", "stage": "parse", "outcome": StageOutcome.CACHED, "duration_ms": 1.0},
            {"source": "a.md", "stage": "chunk", "outcome": StageOutcome.OK, "duration_ms": 5.0},
        ],
    )
    report = build_last_run_report(tmp_path)
    assert report.run_id == "run-1"
    assert report.rows == 3
    assert report.sources == 2
    assert report.cache_hit_rate == 0.5


def test_degradations_keep_the_call_denominator(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        [
            {"source": "a.md", "stage": "parse", "outcome": StageOutcome.OK},
            {"source": "b.md", "stage": "parse", "outcome": StageOutcome.DEGRADED},
            {"source": "c.md", "stage": "parse", "outcome": StageOutcome.OK, "reason_code": ReasonCode.DECISION_FALLBACK},
            {"source": "d.md", "stage": "chunk", "outcome": StageOutcome.OK},
        ],
    )
    report = build_last_run_report(tmp_path)
    counts = {entry.reason: entry for entry in report.degradations}
    assert counts["decision_fallback"].count == 1
    assert counts["decision_fallback"].calls == 3
    assert counts["degraded"].count == 1
    assert counts["degraded"].calls == 3


def test_log_only_degradation_is_counted(tmp_path: Path) -> None:
    _seed(tmp_path, [{"source": "a.md", "stage": "parse", "outcome": StageOutcome.OK}])
    log_path = VaultPaths(tmp_path).run_log_file("run-1")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "WARNING Embedding merge disabled (test); falling back to string similarity\n",
        encoding="utf-8",
    )
    report = build_last_run_report(tmp_path)
    counts = {entry.reason: entry for entry in report.degradations}
    assert counts["embedding_merge_disabled"].count == 1
    assert counts["embedding_merge_disabled"].source == "run log"
    assert report.log_path == str(log_path)


def test_slowest_and_per_stage_items(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        [
            {"source": "a.md", "stage": "parse", "outcome": StageOutcome.OK, "duration_ms": 300.0},
            {"source": "b.md", "stage": "parse", "outcome": StageOutcome.OK, "duration_ms": 100.0},
            {"source": "a.md", "stage": "chunk", "outcome": StageOutcome.OK, "duration_ms": 200.0},
        ],
    )
    report = build_last_run_report(tmp_path)
    assert [item.duration_ms for item in report.slowest] == [300.0, 200.0, 100.0]
    by_stage = {item.stage: item for item in report.slowest_by_stage}
    assert by_stage["parse"].duration_ms == 300.0
    assert by_stage["chunk"].duration_ms == 200.0


def test_failure_histogram_by_reason_code(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        [
            {"source": "a.md", "stage": "parse", "outcome": StageOutcome.TIMEOUT, "reason_code": ReasonCode.TIMEOUT},
            {"source": "b.md", "stage": "parse", "outcome": StageOutcome.FAILED, "reason_code": ReasonCode.TIMEOUT},
            {"source": "c.md", "stage": "parse", "outcome": StageOutcome.OK},
        ],
    )
    report = build_last_run_report(tmp_path)
    assert {(bucket.outcome, bucket.reason_code, bucket.count) for bucket in report.failures} == {
        ("failed", "timeout", 1),
        ("timeout", "timeout", 1),
    }


def test_run_facts_and_missing_log(tmp_path: Path) -> None:
    _seed(tmp_path, [{"source": "a.md", "stage": "parse", "outcome": StageOutcome.OK}])
    _run_row(tmp_path)
    report = build_last_run_report(tmp_path)
    assert report.command == "ingest"
    assert report.status == "ok"
    assert report.duration_s == 12.5
    assert report.log_path is None


def _run_row(tmp_path: Path, *, run_id: str = "run-1") -> None:
    db_path = VaultPaths(tmp_path).kompend_db_file
    with open_store(db_path) as conn:
        conn.execute(
            "INSERT INTO runs (event, run_id, cmd, ts, status, duration_s) VALUES (?, ?, ?, ?, ?, ?)",
            ("end", run_id, "ingest", "2026-01-01T00:00:00", "ok", 12.5),
        )
        conn.commit()


def test_store_without_ledger_table_is_empty(tmp_path: Path) -> None:
    db_path = VaultPaths(tmp_path).kompend_db_file
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(db_path).close()
    report = build_last_run_report(tmp_path)
    assert not report.has_runs
    assert report.degradations == ()


def test_to_dict_round_trips_the_record(tmp_path: Path) -> None:
    _seed(tmp_path, [{"source": "a.md", "stage": "parse", "outcome": StageOutcome.OK}])
    data = build_last_run_report(tmp_path).to_dict()
    assert data["run_id"] == "run-1"
    assert data["rows"] == 1
    assert isinstance(data["slowest"], list)


def _seed(tmp_path: Path, rows: list[dict[str, Any]], *, run_id: str = "run-1") -> Path:
    db_path = VaultPaths(tmp_path).kompend_db_file
    ledger = StageLedger.open(db_path)
    try:
        for row in rows:
            ledger.record(run_id=run_id, **row)
    finally:
        ledger.close()
    return db_path
