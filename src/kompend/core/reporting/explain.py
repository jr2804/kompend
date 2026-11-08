"""Explain one source file's journey through the pipeline.

``kompend explain <path>`` answers "what happened to this file" without
``--verbose``: the stage-by-stage ledger rows of the chosen run, where the
wall-clock time went, the attempt's facts (parser, backend, judge model,
cache identity) and every degradation with its reason code. Presentation
only — no LLM calls; all data comes from the per-item ledger
(:mod:`kompend.core.reporting.ledger`).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Final

from kompend.core.paths import VaultPaths
from kompend.core.reporting.ledger import matching_sources, source_attempts, source_runs
from kompend.core.store import open_store
from kompend.core.types import ReasonCode, StageOutcome

#: Reason codes that mark a row as a degradation even when the outcome is OK.
DEGRADATION_REASONS: Final[frozenset[str]] = frozenset({ReasonCode.JUDGE_DEGRADED.value, ReasonCode.DECISION_FALLBACK.value})

#: Row facts merged first-non-null into the report (attempt-level identity).
_FACT_KEYS: Final[tuple[str, ...]] = (
    "parser",
    "parser_version",
    "backend",
    "judge_model",
    "cache_key",
    "sha",
    "size_bytes",
)


class AmbiguousSourceError(ValueError):
    """Raised when *source* matches several ledger sources.

    Attributes:
        candidates: The matching ledger sources, sorted.
    """

    def __init__(self, source: str, candidates: list[str]) -> None:
        super().__init__(f"ambiguous source {source!r}: matches {len(candidates)} ledger files")
        self.candidates = candidates


def build_explain_report(base_dir: Path, source: str, *, run_id: str | None = None) -> dict[str, Any] | None:
    """Build the explain report for *source* in the vault at *base_dir*.

    Args:
        base_dir: Vault root (the wiki dir; ``.kompend`` lives beneath it).
        source: Ledger source name, file name, or unambiguous fragment.
        run_id: Run to explain; ``None`` picks the latest run that touched
            the file.

    Returns:
        ``None`` when the ledger database is missing or *source* matches
        nothing; :class:`AmbiguousSourceError` propagates on fragment
        collisions.

    """
    paths = VaultPaths(base_dir)
    db_path = paths.kompend_db_file
    if not db_path.exists():
        return None
    with open_store(db_path) as conn:
        resolved = resolve_source(conn, source)
        if resolved is None:
            return None
        run_ids = source_runs(conn, resolved)
        chosen_run = run_id if run_id is not None else (run_ids[-1] if run_ids else None)
        attempts = source_attempts(conn, resolved, run_id=chosen_run)

    rows = [dict(row) for row in attempts]
    stage_totals: dict[str, float] = {}
    for row in rows:
        duration = row["duration_ms"]
        if duration is not None:
            stage_totals[row["stage"]] = stage_totals.get(row["stage"], 0.0) + float(duration)
    slowest = max(stage_totals, key=lambda stage: stage_totals[stage]) if stage_totals else None
    degradations = [
        {
            "stage": row["stage"],
            "attempt": row["attempt"],
            "reason_code": row["reason_code"],
            "detail": row["detail"],
        }
        for row in rows
        if row["outcome"] == StageOutcome.DEGRADED.value or row["reason_code"] in DEGRADATION_REASONS
    ]
    facts = {key: next((row[key] for row in rows if row.get(key) is not None), None) for key in _FACT_KEYS}
    return {
        "source": resolved,
        "run_id": chosen_run,
        "runs_seen": len(run_ids),
        "rows": rows,
        "stage_totals_ms": dict(sorted(stage_totals.items(), key=lambda item: -item[1])),
        "slowest_stage": slowest,
        "total_ms": sum(stage_totals.values()),
        "degradations": degradations,
        "facts": facts,
    }


def resolve_source(conn: sqlite3.Connection, source: str) -> str | None:
    """Resolve *source* to a ledger source name.

    Exact ledger name first, then a unique file-name match, then a unique
    substring match. ``None`` when nothing matches;
    :class:`AmbiguousSourceError` when several do.
    """
    matches = matching_sources(conn, source)
    if not matches:
        return None
    if source in matches:
        return source
    name = source.rsplit("/", 1)[-1]
    by_name = [m for m in matches if m.rsplit("/", 1)[-1] == name]
    if len(by_name) == 1:
        return by_name[0]
    if len(matches) == 1:
        return matches[0]
    raise AmbiguousSourceError(source, matches)


def render_explain(report: dict[str, Any]) -> str:
    """Render a :func:`build_explain_report` result as plain text."""
    lines = [
        f"source: {report['source']}",
        f"run: {report['run_id']} (latest of {report['runs_seen']} run(s) that touched this file)",
        "",
    ]
    lines.append(f"{'stage':<12} {'attempt':>7}  {'outcome':<9} {'ms':>9}  reason")
    for row in report["rows"]:
        duration = "" if row["duration_ms"] is None else f"{row['duration_ms']:.1f}"
        reason = row["reason_code"] or ""
        lines.append(f"{row['stage']:<12} {row['attempt']:>7}  {row['outcome']:<9} {duration:>9}  {reason}".rstrip())
    lines.append("")
    if report["slowest_stage"] is not None:
        share = report["stage_totals_ms"][report["slowest_stage"]]
        lines.append(f"time went to: {report['slowest_stage']} ({share:.1f} of {report['total_ms']:.1f} ms tracked)")
    else:
        lines.append("time went to: (no durations tracked)")
    facts = report["facts"]
    rendered_facts = " ".join(f"{key}={facts[key]}" for key in _FACT_KEYS if facts[key] is not None)
    if rendered_facts:
        lines.append(f"facts: {rendered_facts}")
    if report["degradations"]:
        lines.append("")
        lines.append("degradations:")
        for entry in report["degradations"]:
            reason = entry["reason_code"] or StageOutcome.DEGRADED.value
            suffix = f" — {entry['detail']}" if entry["detail"] else ""
            lines.append(f"  {entry['stage']} (attempt {entry['attempt']}): {reason}{suffix}")
    else:
        lines.append("degradations: none")
    return "\n".join(lines)
