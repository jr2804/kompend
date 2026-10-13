"""Unit tests for the on-demand pipeline log report (``kompend log``)."""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants.

from __future__ import annotations

import json
from pathlib import Path

from kompend.core.reporting.log import build_log_report
from kompend.core.store import open_store
from kompend.core.store.schema import LINT_RUNS_TABLE, RUNS_TABLE


class TestBuildLogReport:
    def test_empty_vault_returns_placeholder(self, tmp_path: Path) -> None:
        report = build_log_report(tmp_path, last_n=5)
        assert "No compile" in report or "kompend Log" in report

    def test_runs_timeline_shows_completed_and_flags_dangling(self, tmp_path: Path) -> None:
        state_dir = tmp_path / ".kompend"
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
        state_dir = tmp_path / ".kompend"
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
    with open_store(state_dir / "kompend.db") as conn:
        for ev in events:
            conn.execute(
                f"INSERT INTO {RUNS_TABLE} (event, run_id, cmd, ts, status, duration_s, exit, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",  # noqa: S608
                (ev["event"], ev["run_id"], ev["cmd"], ev["ts"], ev.get("status"), ev.get("duration_s"), ev.get("exit"), ev.get("error")),
            )


def _write_lint(state_dir: Path, entries: list[dict]) -> None:
    with open_store(state_dir / "kompend.db") as conn:
        for e in entries:
            conn.execute(
                f"INSERT INTO {LINT_RUNS_TABLE} (timestamp, duration_seconds, options, llm, config_snapshot, wiki_meta, report) VALUES (?, ?, ?, ?, ?, ?, ?)",  # noqa: S608
                (e["timestamp"], e.get("duration_seconds", 0.0), "{}", None, "{}", "{}", json.dumps(e["report"], default=str)),
            )


def _make_run(run_id: str, cmd: str, *, status: str | None = None, dur: float = 1.5) -> list[dict]:
    events = [{"event": "start", "run_id": run_id, "cmd": cmd, "ts": "2026-07-05T10:00:00+02:00"}]
    if status is not None:
        events.append({"event": "end", "run_id": run_id, "cmd": cmd, "status": status, "duration_s": dur, "ts": "2026-07-05T10:00:01+02:00"})
    return events
