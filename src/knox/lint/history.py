"""Persistent lint history — one row per lint run in the ``lint_runs`` table.

Each row holds: timestamp, duration_seconds, options, llm, config_snapshot,
wiki_meta, report.

The :class:`LintEntry` TypedDict is the schema for one row — it types
key access without a runtime wrapper. Use save_lint_entry()/
load_lint_history() for I/O; callers pass ``LintEntry`` dicts.
"""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants.

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict

from knox.core.config.keys import (
    COMPILE_COMPILER,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_FALLBACK_CHAIN,
)
from knox.core.store import open_store, vault_db_path
from knox.core.store.schema import LINT_RUNS_TABLE
from knox.core.utils.time import iso_local, now_local
from knox.lint.base import LintReport, LintResult, LintRuleOutcome, Severity
from knox.lint.runner import report_to_dict

# Config keys captured in the snapshot
_SNAPSHOT_KEYS = (
    COMPILE_COMPILER,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_FALLBACK_CHAIN,
)


# ── Helpers ──────────────────────────────────────────────────────────────

_SPARK_GLYPHS = "▁▂▃▄▅▆▇█"


class LintEntry(TypedDict):
    """Schema for one JSONL line in the lint history file."""

    timestamp: str
    duration_seconds: float
    options: dict[str, Any]
    llm: dict[str, Any] | None
    config_snapshot: dict[str, Any]
    wiki_meta: dict[str, Any]
    report: dict[str, Any]


def entry_from_report(
    report: LintReport,
    *,
    duration_seconds: float,
    options: dict[str, Any],
    config: dict[str, Any],
    node_count: int,
    llm: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> LintEntry:
    """Build a lint history entry from a lint run."""
    return {
        "timestamp": timestamp or iso_local(),
        "duration_seconds": round(duration_seconds, 3),  # nocheck — lint-history schema, not WikiField
        "options": dict(options),
        "llm": llm,
        "config_snapshot": _build_config_snapshot(config),
        "wiki_meta": {"node_count": node_count},  # nocheck: lint-history schema, not wiki metadata
        "report": report_to_dict(report),
    }


def report_from_dict(report_dict: dict[str, Any]) -> LintReport:
    """Parse a ``lint_history.jsonl`` report dict back into :class:`LintReport`.

    The inverse of :func:`knox.lint.runner.report_to_dict`, and the one place
    history entries become typed findings again — the CLI and MCP each carried
    their own copy of this conversion and both dropped ``severity`` and
    ``fix_applied`` on the way in.
    """
    results = [
        LintResult(
            rule=str(r.get("rule", "")),
            outcome=LintRuleOutcome(r.get("outcome", "") or "skip"),
            severity=Severity(r.get("severity", "") or "warning"),
            message=str(r.get("message", "")),
            page_id=str(r.get("page_id", "")),
            fix_applied=bool(r.get("fix_applied", False)),
            details=dict(r.get("details") or {}),
        )
        for r in report_dict.get("results", [])
    ]
    return LintReport(
        results=results,
        total_checks=int(report_dict.get("total_checks", 0) or 0),
        failures=int(report_dict.get("failures", 0) or 0),
        passes=int(report_dict.get("passes", 0) or 0),
        skips=int(report_dict.get("skips", 0) or 0),
        health_score=float(report_dict.get("health_score", 0.0) or 0.0),
    )


def _build_config_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract the lint-relevant subset of a resolved config."""
    return {key: getattr(cfg, "get", lambda k: None)(key) for key in _SNAPSHOT_KEYS}


def save_lint_entry(entry: LintEntry, path: Path) -> None:
    """Append *entry* as one row in the ``lint_runs`` table for the vault at *path*."""
    with open_store(vault_db_path(path)) as conn:
        conn.execute(
            f"INSERT INTO {LINT_RUNS_TABLE} (timestamp, duration_seconds, options, llm, config_snapshot, wiki_meta, report) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                entry["timestamp"],
                entry["duration_seconds"],  # nocheck: lint_history column/payload key
                json.dumps(entry["options"], default=str, ensure_ascii=False),
                json.dumps(entry["llm"], default=str, ensure_ascii=False) if entry["llm"] is not None else None,
                json.dumps(entry["config_snapshot"], default=str, ensure_ascii=False),
                json.dumps(entry["wiki_meta"], default=str, ensure_ascii=False),
                json.dumps(entry["report"], default=str, ensure_ascii=False),
            ),
        )


def prune_lint_history(
    path: Path,
    *,
    keep_last: int | None = None,
    max_age_days: int | None = None,
) -> int:
    """Remove old rows from the ``lint_runs`` table for the vault at *path*.

    Returns the number of entries removed.
    """
    if keep_last is None and max_age_days is None:
        raise ValueError("At least one of keep_last or max_age_days must be set.")

    db_path = vault_db_path(path)
    if not db_path.is_file():
        return 0

    with open_store(db_path) as conn:
        rows = conn.execute(f"SELECT id, timestamp FROM {LINT_RUNS_TABLE} ORDER BY id").fetchall()
        if not rows:
            return 0

        surviving = [(r["id"], r["timestamp"]) for r in rows]
        if max_age_days is not None:
            cutoff = now_local() - timedelta(days=max_age_days)
            surviving = [(rid, ts) for rid, ts in surviving if _parse_timestamp(ts) >= cutoff]
        if keep_last is not None:
            surviving = surviving[-keep_last:] if keep_last > 0 else []

        keep_ids = {rid for rid, _ in surviving}
        doomed = [r["id"] for r in rows if r["id"] not in keep_ids]
        if doomed:
            conn.executemany(f"DELETE FROM {LINT_RUNS_TABLE} WHERE id = ?", [(rid,) for rid in doomed])
        return len(doomed)


def load_lint_history(
    path: Path,
    *,
    last_n: int | None = None,
) -> list[LintEntry]:
    """Load lint history rows for the vault at *path* (oldest first).

    A missing store returns an empty list.
    """
    db_path = vault_db_path(path)
    entries: list[LintEntry] = []
    if db_path.is_file():
        with open_store(db_path) as conn:
            rows = conn.execute(
                f"SELECT timestamp, duration_seconds, options, llm, config_snapshot, wiki_meta, report FROM {LINT_RUNS_TABLE} ORDER BY id"
            ).fetchall()
        entries = [_row_to_entry(row) for row in rows]

    if last_n is not None and last_n >= 0:
        entries = entries[-last_n:] if last_n > 0 else []
    return entries


def _row_to_entry(row: sqlite3.Row) -> LintEntry:
    """Rebuild a :class:`LintEntry` from a ``lint_runs`` row."""
    return {
        "timestamp": row["timestamp"],
        "duration_seconds": row["duration_seconds"],  # nocheck: lint_history column/payload key
        "options": json.loads(row["options"]),
        "llm": json.loads(row["llm"]) if row["llm"] is not None else None,
        "config_snapshot": json.loads(row["config_snapshot"]),
        "wiki_meta": json.loads(row["wiki_meta"]),
        "report": json.loads(row["report"]),
    }


def history_to_json(entries: list[LintEntry]) -> str:
    """Serialise a list of entries to a pretty JSON string."""
    return json.dumps(entries, indent=2, default=str, ensure_ascii=False)


def history_to_text(
    entries: list[LintEntry],
    *,
    show_trend: bool = False,
) -> str:
    """Format lint history as a human-readable summary."""
    if not entries:
        return "No lint history found. Run 'knox lint run' first."

    lines: list[str] = []
    lines.append(f"Lint History ({len(entries)} entr{'y' if len(entries) == 1 else 'ies'})")
    lines.append("")

    for entry in entries:
        report = entry.get("report", {})
        score = report.get("health_score", 0.0)
        fails = report.get("failures", 0)
        passes = report.get("passes", 0)
        skips = report.get("skips", 0)
        llm = entry.get("llm")
        model_tag = f"  [{llm.get('model')}]" if llm else ""  # nocheck: lint-history schema, not wiki metadata
        lines.append(f"  {entry['timestamp']}  score={score:.0f}%  {passes} pass / {fails} fail / {skips} skip{model_tag}")

    if show_trend and len(entries) >= 2:
        scores = [e.get("report", {}).get("health_score", 0.0) for e in entries]
        lines.append("")
        lines.append(f"  Trend: {_sparkline(scores)}")
        lines.append(f"  Range: {min(scores):.0f}% — {max(scores):.0f}%")

    return "\n".join(lines)


def _sparkline(values: list[float]) -> str:
    """Render a list of numbers as a Unicode sparkline."""
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return _SPARK_GLYPHS[-1] * len(values)
    span = hi - lo
    return "".join(_SPARK_GLYPHS[min(len(_SPARK_GLYPHS) - 1, int((v - lo) / span * (len(_SPARK_GLYPHS) - 1)))] for v in values)


def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp, returning a timezone-aware datetime."""
    return datetime.fromisoformat(ts)
