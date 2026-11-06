"""Persistent lint history — one row per lint run in the ``lint_runs`` table.

Each row holds: timestamp, duration_seconds, options, llm, config_snapshot,
wiki_meta, report. Every JSON column stores the matching record below, so the
table shape and the type shape cannot drift apart: add a column by adding a
field to :class:`LintEntry` and a name to :data:`_COLUMNS`. Rows are
validated on load, so a corrupt store fails here with the field name
instead of downstream with a KeyError.

The :class:`LintEntry` TypedDict is the schema for one row — it types
key access without a runtime wrapper. Use save_lint_entry()/
load_lint_history() for I/O; callers pass ``LintEntry`` dicts.
"""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants.

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, TypedDict

from pydantic import TypeAdapter

from kompend.core.keys import (
    COMPILE_COMPILER,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_FALLBACK_CHAIN,
)
from kompend.core.paths import LINT_HISTORY_FILE, VaultPaths
from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import LINT_RUNS_TABLE
from kompend.core.utils.time import iso_local, now_local
from kompend.lint.base import LintReport, LintResult, LintRuleOutcome, Severity
from kompend.lint.runner import ReportDict, report_to_dict
from kompend.llm.usage import UsageSnapshot

# Config keys captured in the snapshot
_SNAPSHOT_KEYS = (
    COMPILE_COMPILER,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_FALLBACK_CHAIN,
)

_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "duration_seconds",
    "options",
    "llm",
    "config_snapshot",
    "wiki_meta",
    "report",
)
_COLUMNS_SQL: Final[str] = ", ".join(_COLUMNS)
_PLACEHOLDERS_SQL: Final[str] = ", ".join("?" * len(_COLUMNS))


# ── Helpers ──────────────────────────────────────────────────────────────

_SPARK_GLYPHS = "▁▂▃▄▅▆▇█"


LintConfigSnapshot = TypedDict(
    "LintConfigSnapshot",
    {
        "compile.compiler": str | None,
        "compile.llm.default_model": str | None,
        "compile.llm.fallback_chain": list[str] | None,
    },
)


class LintRunOptions(TypedDict):
    """Which slice of the lint run this entry records."""

    context: str
    fix: bool


class LintLlmMeta(TypedDict, total=False):
    """Model identity for a lint run that used LLM rules."""

    model: str
    usage: UsageSnapshot


class LintWikiMeta(TypedDict):
    """Corpus size the lint run evaluated."""

    node_count: int


class LintEntry(TypedDict):
    """Schema for one row in the ``lint_runs`` table."""

    timestamp: str
    duration_seconds: float
    options: LintRunOptions
    llm: LintLlmMeta | None
    config_snapshot: LintConfigSnapshot
    wiki_meta: LintWikiMeta
    report: ReportDict


_LINT_ENTRY_ADAPTER = TypeAdapter(LintEntry)


def persist_history(  # noqa: PLR0917 — a history entry needs every run fact
    report: LintReport,
    *,
    vp: VaultPaths,
    config: Mapping[str, Any],
    context: str,
    elapsed: float,
    fix: bool,
    node_count: int,
    llm: LintLlmMeta | None,
) -> None:
    """Append the run as a history entry, so ``lint history`` and ``lint gaps`` see it.

    The CLI commands and the MCP tools both record runs; this is the one place
    that entry is built and stored.
    """
    entry = entry_from_report(
        report,
        duration_seconds=elapsed,
        options={"context": context, "fix": fix},
        config=config,
        node_count=node_count,
        llm=llm,
    )
    save_lint_entry(entry, vp.kompend_dir / LINT_HISTORY_FILE)


def entry_from_report(
    report: LintReport,
    *,
    duration_seconds: float,
    options: LintRunOptions,
    config: Mapping[str, Any],
    node_count: int,
    llm: LintLlmMeta | None = None,
    timestamp: str | None = None,
) -> LintEntry:
    """Build a lint history entry from a lint run."""
    return {
        "timestamp": timestamp or iso_local(),
        "duration_seconds": round(duration_seconds, 3),  # nocheck — lint-history schema, not WikiField
        "options": options,
        "llm": llm,
        "config_snapshot": _build_config_snapshot(config),
        "wiki_meta": {"node_count": node_count},  # nocheck: lint-history schema, not wiki metadata
        "report": report_to_dict(report),
    }


def report_from_dict(report_dict: ReportDict) -> LintReport:
    """Parse a ``lint_runs`` report payload back into :class:`LintReport`.

    The inverse of :func:`kompend.lint.runner.report_to_dict`, and the one place
    history entries become typed findings again — the CLI and MCP each carried
    their own copy of this conversion and both dropped ``severity`` and
    ``fix_applied`` on the way in.
    """
    results = [
        LintResult(
            rule=r.get("rule", ""),
            outcome=LintRuleOutcome(r.get("outcome", "") or "skip"),
            severity=Severity(r.get("severity", "") or "warning"),
            message=r.get("message", ""),
            page_id=r.get("page_id", ""),
            fix_applied=r.get("fix_applied", False),
            details=dict(r.get("details") or {}),
        )
        for r in report_dict.get("results", [])
    ]
    return LintReport(
        results=results,
        total_checks=report_dict.get("total_checks", 0) or 0,
        failures=report_dict.get("failures", 0) or 0,
        passes=report_dict.get("passes", 0) or 0,
        skips=report_dict.get("skips", 0) or 0,
        health_score=report_dict.get("health_score", 0.0) or 0.0,
    )


def _build_config_snapshot(cfg: Mapping[str, Any]) -> LintConfigSnapshot:
    """Extract the lint-relevant subset of a resolved config."""
    raw = {key: cfg.get(key) for key in _SNAPSHOT_KEYS}
    return {
        "compile.compiler": raw[COMPILE_COMPILER],
        "compile.llm.default_model": raw[COMPILE_LLM_DEFAULT_MODEL],
        "compile.llm.fallback_chain": raw[COMPILE_LLM_FALLBACK_CHAIN],
    }


def save_lint_entry(entry: LintEntry, path: Path) -> None:
    """Append *entry* as one row in the ``lint_runs`` table for the vault at *path*."""
    with open_store(vault_db_path(path)) as conn:
        conn.execute(
            f"INSERT INTO {LINT_RUNS_TABLE} ({_COLUMNS_SQL}) VALUES ({_PLACEHOLDERS_SQL})",
            (
                entry["timestamp"],
                entry["duration_seconds"],  # nocheck — lint_history column/payload key
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

        # The survivor computation stays in Python: stored timestamps carry
        # the machine's local offset, and only parsed datetimes compare
        # mixed offsets correctly. The DELETE itself is one statement.
        surviving = [(r["id"], r["timestamp"]) for r in rows]
        if max_age_days is not None:
            cutoff = now_local() - timedelta(days=max_age_days)
            surviving = [(rid, ts) for rid, ts in surviving if _parse_timestamp(ts) >= cutoff]
        if keep_last is not None:
            surviving = surviving[-keep_last:] if keep_last > 0 else []

        keep_ids = {rid for rid, _ in surviving}
        doomed = [r["id"] for r in rows if r["id"] not in keep_ids]
        if not doomed:
            return 0
        placeholders = ", ".join("?" * len(doomed))
        cursor = conn.execute(f"DELETE FROM {LINT_RUNS_TABLE} WHERE id IN ({placeholders})", doomed)
        return cursor.rowcount


def load_lint_history(
    path: Path,
    *,
    last_n: int | None = None,
) -> list[LintEntry]:
    """Load lint history rows for the vault at *path* (oldest first).

    A missing store returns an empty list.
    """
    if last_n is not None and last_n <= 0:
        return []

    db_path = vault_db_path(path)
    entries: list[LintEntry] = []
    if db_path.is_file():
        with open_store(db_path) as conn:
            if last_n is None:
                rows = conn.execute(f"SELECT {_COLUMNS_SQL} FROM {LINT_RUNS_TABLE} ORDER BY id").fetchall()
            else:
                rows = conn.execute(f"SELECT {_COLUMNS_SQL} FROM {LINT_RUNS_TABLE} ORDER BY id DESC LIMIT ?", (last_n,)).fetchall()
                rows = rows[::-1]
        entries = [_row_to_entry(row) for row in rows]

    return entries


def _row_to_entry(row: sqlite3.Row | Mapping[str, Any]) -> LintEntry:
    """Rebuild a :class:`LintEntry` from a ``lint_runs`` row.

    The assembled entry is validated — including the nested report and
    usage records — so a corrupt row raises here instead of poisoning
    every history consumer.
    """
    return _LINT_ENTRY_ADAPTER.validate_python(
        {
            "timestamp": row["timestamp"],
            "duration_seconds": row["duration_seconds"],  # nocheck: lint_history column/payload key
            "options": json.loads(row["options"]),
            "llm": json.loads(row["llm"]) if row["llm"] is not None else None,
            "config_snapshot": json.loads(row["config_snapshot"]),
            "wiki_meta": json.loads(row["wiki_meta"]),
            "report": json.loads(row["report"]),
        }
    )


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
        return "No lint history found. Run 'kompend lint run' first."

    lines: list[str] = []
    lines.append(f"Lint History ({len(entries)} entr{'y' if len(entries) == 1 else 'ies'})")
    lines.append("")

    for entry in entries:
        report = entry["report"]
        score = report["health_score"]
        fails = report["failures"]
        passes = report["passes"]
        skips = report["skips"]
        llm = entry["llm"]
        model_tag = f"  [{llm['model']}]" if llm else ""  # nocheck: lint-history schema, not wiki metadata
        lines.append(f"  {entry['timestamp']}  score={score:.0f}%  {passes} pass / {fails} fail / {skips} skip{model_tag}")

    if show_trend and len(entries) >= 2:
        scores = [e["report"]["health_score"] for e in entries]
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
