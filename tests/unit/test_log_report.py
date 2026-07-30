"""Unit tests for the on-demand pipeline log report (``knox log``)."""

from __future__ import annotations

import json
from pathlib import Path

from knox.core.paths import LINT_HISTORY_FILE, RUNS_FILE
from knox.core.reporting.log import build_log_report


class TestBuildLogReport:
    def test_empty_vault_returns_placeholder(self, tmp_path: Path) -> None:
        report = build_log_report(tmp_path, last_n=5)
        assert "No compile" in report or "KNOX Log" in report

    def test_runs_timeline_shows_completed_and_flags_dangling(self, tmp_path: Path) -> None:
        state_dir = tmp_path / ".knox"
        _write_runs(
            state_dir,
            _make_run("aaa", "compile", status="ok")
            + _make_run("bbb", "lint run", status="error")
            + _make_run("ccc", "query x"),  # start only — never finished
        )
        report = build_log_report(tmp_path, last_n=10)
        assert "Command Timeline" in report
        assert "compile" in report
        assert "lint run" in report
        # the dangling start is flagged
        assert "never finished" in report

    def test_lint_trend_rendered(self, tmp_path: Path) -> None:
        state_dir = tmp_path / ".knox"
        _write_lint(
            state_dir,
            [
                {"timestamp": "2026-07-05T10:00:00+02:00", "report": {"health_score": 40.0, "passes": 5, "failures": 10, "skips": 0}},
                {"timestamp": "2026-07-05T11:00:00+02:00", "report": {"health_score": 70.0, "passes": 8, "failures": 4, "skips": 0}},
            ],
        )
        report = build_log_report(tmp_path, last_n=10)
        assert "Lint Trend" in report
        assert "score=70%" in report
        assert "Trend" in report  # sparkline + range


def _write_runs(state_dir: Path, events: list[dict]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / RUNS_FILE).open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")


def _write_lint(state_dir: Path, entries: list[dict]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / LINT_HISTORY_FILE).open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def _make_run(run_id: str, cmd: str, *, status: str | None = None, dur: float = 1.5) -> list[dict]:
    events = [{"event": "start", "run_id": run_id, "cmd": cmd, "ts": "2026-07-05T10:00:00+02:00"}]
    if status is not None:
        events.append({"event": "end", "run_id": run_id, "cmd": cmd, "status": status, "duration_s": dur, "ts": "2026-07-05T10:00:01+02:00"})
    return events
