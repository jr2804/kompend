"""The last run, summarised from the ledger and its flight-recorder log.

``kompend doctor`` ends with a "last run" section built from the per-item
ledger (:mod:`kompend.core.reporting.ledger`) and the run's log file. The
facts it gathers are already written — one row per stage attempt, the
degradation rows, the failure reasons — but they only reach the terminal as
WARNING lines that scroll away. This module turns them into one block:

* degradations counted against the calls that could have degraded,
* the slowest items, overall and per stage,
* a failure histogram by reason code,
* the path to the full flight-recorder log.

Read-only. A missing store, a missing table, or a run with no ledger rows
yields an empty report rather than an exception — ``doctor`` must still
answer on a fresh vault.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from kompend.core.paths import VaultPaths
from kompend.core.reporting.explain import DEGRADATION_REASONS
from kompend.core.reporting.ledger import failure_histogram, slowest_items
from kompend.core.types import ReasonCode, StageOutcome

#: Warnings that exist only in the run log: they are emitted by best-effort
#: enrichment paths that never open a ledger row.
_LOG_DEGRADATION_PHRASES: Final[dict[ReasonCode, str]] = {
    ReasonCode.EMBEDDING_MERGE_DISABLED: "Embedding merge disabled",
    ReasonCode.JUDGE_DEGRADED: "falling back to the deterministic rule table",
}
#: Rows shown as the overall slowest list.
_SLOWEST_LIMIT: Final[int] = 5


@dataclass(frozen=True, slots=True)
class DegradationCount:
    """One degradation class, counted against the calls that could degrade."""

    reason: str
    count: int
    calls: int | None = None
    source: str = "ledger"

    def to_dict(self) -> dict[str, object]:
        return {"reason": self.reason, "count": self.count, "calls": self.calls, "source": self.source}


@dataclass(frozen=True, slots=True)
class SlowItem:
    """One slow ledger attempt."""

    source: str
    stage: str
    duration_ms: float
    outcome: str
    reason_code: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "stage": self.stage,
            "duration_ms": self.duration_ms,
            "outcome": self.outcome,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class FailureCount:
    """One (stage, outcome, reason) bucket of the failure histogram."""

    stage: str
    outcome: str
    reason_code: str | None
    count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "outcome": self.outcome,
            "reason_code": self.reason_code,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class LastRunReport:
    """The last run's ledger summary plus the path to its full log."""

    run_id: str | None = None
    command: str | None = None
    status: str | None = None
    duration_s: float | None = None
    log_path: str | None = None
    rows: int = 0
    sources: int = 0
    cache_hit_rate: float | None = None
    degradations: tuple[DegradationCount, ...] = ()
    slowest: tuple[SlowItem, ...] = ()
    slowest_by_stage: tuple[SlowItem, ...] = ()
    failures: tuple[FailureCount, ...] = ()

    @property
    def has_runs(self) -> bool:
        """Whether the vault's ledger holds at least one run."""
        return self.run_id is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "command": self.command,
            "status": self.status,
            "duration_s": self.duration_s,
            "log_path": self.log_path,
            "rows": self.rows,
            "sources": self.sources,
            "cache_hit_rate": self.cache_hit_rate,
            "degradations": [entry.to_dict() for entry in self.degradations],
            "slowest": [entry.to_dict() for entry in self.slowest],
            "slowest_by_stage": [entry.to_dict() for entry in self.slowest_by_stage],
            "failures": [entry.to_dict() for entry in self.failures],
        }


def build_last_run_report(base_dir: Path) -> LastRunReport:
    """Build the report for the latest run that wrote ledger rows."""
    paths = VaultPaths(base_dir)
    db_path = paths.kompend_db_file
    if not db_path.exists():
        return LastRunReport()
    try:
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)) as conn:
            conn.row_factory = sqlite3.Row
            run_id = _latest_ledger_run(conn)
            if run_id is None:
                return LastRunReport()
            command, status, duration_s = _run_facts(conn, run_id)
            rows, sources = _counts(conn, run_id)
            report = LastRunReport(
                run_id=run_id,
                command=command,
                status=status,
                duration_s=duration_s,
                rows=rows,
                sources=sources,
                cache_hit_rate=_cache_hit_rate(conn, run_id),
                degradations=_degradations(conn, run_id),
                slowest=_slowest(conn, run_id),
                slowest_by_stage=_slowest_by_stage(conn, run_id),
                failures=_failures(conn, run_id),
            )
    except sqlite3.Error:
        return LastRunReport()
    log_path = paths.run_log_file(run_id)
    log_text = _read_log(log_path)
    degradations = _with_log_signals(report.degradations, log_text)
    return LastRunReport(
        run_id=report.run_id,
        command=report.command,
        status=report.status,
        duration_s=report.duration_s,
        log_path=str(log_path) if log_path.is_file() else None,
        rows=report.rows,
        sources=report.sources,
        cache_hit_rate=report.cache_hit_rate,
        degradations=degradations,
        slowest=report.slowest,
        slowest_by_stage=report.slowest_by_stage,
        failures=report.failures,
    )


def render_last_run(report: LastRunReport) -> list[str]:
    """Render the report as indented lines for the doctor's text output."""
    if not report.has_runs:
        return ["  last run: no ledger rows recorded yet"]
    header = f"  last run: {report.run_id}"
    if report.command:
        header += f" ({report.command})"
    if report.status:
        header += f" · {report.status}"
    if report.duration_s is not None:
        header += f" · {report.duration_s:.1f}s"
    lines = [header]
    if report.log_path:
        lines.append(f"    log: {report.log_path}")
    lines.append(f"    {report.rows} row(s) across {report.sources} source(s)")
    if report.cache_hit_rate is not None:
        lines.append(f"    parse cache hits: {report.cache_hit_rate:.0%}")
    lines.append("    degradations: " + _render_degradations(report.degradations))
    if report.slowest:
        lines.append("    slowest:")
        for item in report.slowest:
            reason = f" [{item.reason_code}]" if item.reason_code else ""
            lines.append(f"      {item.stage:<10} {item.source:<32} {item.duration_ms:>10.1f} ms {item.outcome}{reason}")
    if report.failures:
        lines.append("    failures:")
        for bucket in report.failures:
            reason = bucket.reason_code or "-"
            lines.append(f"      {bucket.stage:<10} {bucket.outcome:<10} {reason:<20} {bucket.count}")
    return lines


def _latest_ledger_run(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT run_id FROM stages WHERE run_id IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
    return row[0] if row is not None else None


def _run_facts(conn: sqlite3.Connection, run_id: str) -> tuple[str | None, str | None, float | None]:
    row = conn.execute(
        "SELECT cmd, status, duration_s FROM runs WHERE run_id = ? AND event = 'end' ORDER BY id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT cmd, status, duration_s FROM runs WHERE run_id = ? ORDER BY id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
    if row is None:
        return None, None, None
    return row["cmd"], row["status"], row["duration_s"]


def _counts(conn: sqlite3.Connection, run_id: str) -> tuple[int, int]:
    row = conn.execute(
        "SELECT COUNT(*) AS rows, COUNT(DISTINCT source) AS sources FROM stages WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    return (row["rows"], row["sources"]) if row is not None else (0, 0)


def _cache_hit_rate(conn: sqlite3.Connection, run_id: str) -> float | None:
    rows = conn.execute(
        "SELECT outcome, COUNT(*) AS n FROM stages WHERE run_id = ? AND stage = 'parse' GROUP BY outcome",
        (run_id,),
    ).fetchall()
    counts = {row["outcome"]: row["n"] for row in rows}
    parsed = counts.get(StageOutcome.CACHED.value, 0) + counts.get(StageOutcome.OK.value, 0)
    if parsed == 0:
        return None
    return counts.get(StageOutcome.CACHED.value, 0) / parsed


def _degradations(conn: sqlite3.Connection, run_id: str) -> tuple[DegradationCount, ...]:
    rows = conn.execute(
        "SELECT reason_code, stage, COUNT(*) AS n FROM stages "
        "WHERE run_id = ? AND (outcome = ? OR reason_code IN (?, ?)) "
        "GROUP BY reason_code, stage ORDER BY n DESC",
        (run_id, StageOutcome.DEGRADED.value, *sorted(DEGRADATION_REASONS)),
    ).fetchall()
    if not rows:
        return ()
    totals = dict(conn.execute("SELECT stage, COUNT(*) FROM stages WHERE run_id = ? GROUP BY stage", (run_id,)).fetchall())
    by_reason: dict[str, int] = {}
    stages_by_reason: dict[str, set[str]] = {}
    for row in rows:
        reason = row["reason_code"] or StageOutcome.DEGRADED.value
        by_reason[reason] = by_reason.get(reason, 0) + row["n"]
        stages_by_reason.setdefault(reason, set()).add(row["stage"])
    return tuple(
        DegradationCount(
            reason=reason,
            count=count,
            calls=sum(totals.get(stage, 0) for stage in stages_by_reason[reason]),
        )
        for reason, count in sorted(by_reason.items(), key=lambda item: -item[1])
    )


def _with_log_signals(
    counts: tuple[DegradationCount, ...],
    log_text: str,
) -> tuple[DegradationCount, ...]:
    by_reason = {entry.reason: entry for entry in counts}
    for reason, phrase in _LOG_DEGRADATION_PHRASES.items():
        found = log_text.count(phrase)
        if found == 0:
            continue
        value = reason.value
        existing = by_reason.get(value)
        if existing is None:
            by_reason[value] = DegradationCount(reason=value, count=found, source="run log")
        elif existing.source == "ledger":
            # The warning and the ledger row describe the same degradation;
            # keep the larger count rather than summing two views.
            by_reason[value] = DegradationCount(reason=value, count=max(existing.count, found), calls=existing.calls)
    return tuple(sorted(by_reason.values(), key=lambda entry: -entry.count))


def _slowest(conn: sqlite3.Connection, run_id: str) -> tuple[SlowItem, ...]:
    return tuple(_slow_item(row) for row in slowest_items(conn, run_id, limit=_SLOWEST_LIMIT))


def _slowest_by_stage(conn: sqlite3.Connection, run_id: str) -> tuple[SlowItem, ...]:
    rows = conn.execute(
        "SELECT source, stage, duration_ms, outcome, reason_code FROM stages WHERE run_id = ? AND duration_ms IS NOT NULL ORDER BY duration_ms DESC",
        (run_id,),
    ).fetchall()
    seen: set[str] = set()
    items: list[SlowItem] = []
    for row in rows:
        if row["stage"] in seen:
            continue
        seen.add(row["stage"])
        items.append(_slow_item(row))
    return tuple(items)


def _slow_item(row: sqlite3.Row) -> SlowItem:
    return SlowItem(
        source=row["source"],
        stage=row["stage"],
        duration_ms=float(row["duration_ms"]),
        outcome=row["outcome"],
        reason_code=row["reason_code"],
    )


def _failures(conn: sqlite3.Connection, run_id: str) -> tuple[FailureCount, ...]:
    return tuple(
        FailureCount(
            stage=row["stage"],
            outcome=row["outcome"],
            reason_code=row["reason_code"],
            count=row["n"],
        )
        for row in failure_histogram(conn, run_id)
    )


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _render_degradations(counts: tuple[DegradationCount, ...]) -> str:
    if not counts:
        return "none"
    parts: list[str] = []
    for entry in counts:
        if entry.calls is None:
            parts.append(f"{entry.reason} {entry.count} ({entry.source})")
        else:
            parts.append(f"{entry.reason} {entry.count}/{entry.calls} ({entry.source})")
    return ", ".join(parts)
