"""Persistent lint history — one row per lint run in the ``lint_runs`` table.

Each row holds: timestamp, duration_seconds, options, llm, config_snapshot,
wiki_meta, report. The stored shape is exactly :meth:`LintEntry.to_columns`,
and :meth:`LintEntry.from_row` is its inverse, so the table and the model
cannot drift apart: add a column by adding a field to :class:`LintEntry` (a
mapping key in ``to_columns``). Rows are validated on load, so a corrupt store
fails here with the field name instead of downstream with a KeyError.

Use save_lint_entry()/load_lint_history() for I/O.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from kompend.core.keys import (
    COMPILE_COMPILER,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_FALLBACK_CHAIN,
)
from kompend.core.paths import LINT_HISTORY_FILE, VaultPaths
from kompend.core.store import core_connection, open_store, vault_db_path
from kompend.core.store.tables import lint_runs
from kompend.core.utils.time import iso_local, now_local
from kompend.lint.base import LintReport, LintResult, LintRuleOutcome, Severity
from kompend.lint.details import details_from_storage
from kompend.lint.runner import LintReportPayload, report_to_payload
from kompend.llm.usage import UsageSnapshot
from kompend.types import JSONDict, JSONValue, RawConfigValue

_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "duration_seconds",
    "options",
    "llm",
    "config_snapshot",
    "wiki_meta",
    "report",
)


# ── Helpers ──────────────────────────────────────────────────────────────

_SPARK_GLYPHS = "▁▂▃▄▅▆▇█"


class LintConfigSnapshot(BaseModel):
    """The lint-relevant subset of a resolved config, keyed by dotted config key."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    compiler: str | None = Field(default=None, alias=COMPILE_COMPILER)
    default_model: str | None = Field(default=None, alias=COMPILE_LLM_DEFAULT_MODEL)
    fallback_chain: list[str] | None = Field(default=None, alias=COMPILE_LLM_FALLBACK_CHAIN)


class LintRunOptions(BaseModel):
    """Which slice of the lint run this entry records."""

    context: str
    fix: bool


class LintLlmMeta(BaseModel):
    """Model identity for a lint run that used LLM rules."""

    model: str
    usage: UsageSnapshot | None = None


class LintWikiMeta(BaseModel):
    """Corpus size the lint run evaluated."""

    node_count: int


class LintEntry(BaseModel):
    """Schema for one row in the ``lint_runs`` table."""

    timestamp: str
    duration_seconds: float
    options: LintRunOptions
    llm: LintLlmMeta | None = None
    config_snapshot: LintConfigSnapshot
    wiki_meta: LintWikiMeta
    report: LintReportPayload

    def to_columns(self) -> JSONDict:
        """Serialise to the column-keyed values of one ``lint_runs`` row."""
        return {
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,  # nocheck: lint_runs column, not a wiki field
            "options": _json_column(self.options),
            "llm": _json_column(self.llm) if self.llm is not None else None,
            "config_snapshot": _json_column(self.config_snapshot),
            "wiki_meta": _json_column(self.wiki_meta),
            "report": _json_column(self.report),
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row | Mapping[str, JSONValue]) -> LintEntry:
        """Rebuild an entry from a ``lint_runs`` row, validating every JSON column.

        The nested records are validated too, so a corrupt row raises here
        instead of poisoning every history consumer.
        """
        return cls.model_validate(
            {
                "timestamp": row["timestamp"],
                "duration_seconds": row["duration_seconds"],  # nocheck: lint_history column/payload key
                "options": _load_column(row, "options"),
                "llm": _load_column(row, "llm") if row["llm"] is not None else None,
                "config_snapshot": _load_column(row, "config_snapshot"),
                "wiki_meta": _load_column(row, "wiki_meta"),
                "report": _load_column(row, "report"),
            }
        )


def _load_column(row: sqlite3.Row | Mapping[str, JSONValue], key: str) -> JSONValue:
    """Decode one JSON ``TEXT`` column, naming the column when it is not text.

    A ``TEXT`` column always yields ``str``; anything else means the row was
    written by something other than this schema, so it raises instead of
    silently feeding a non-JSON value into ``model_validate``.
    """
    raw = row[key]
    if not isinstance(raw, str):
        raise TypeError(f"lint_runs.{key} must be JSON text, not {type(raw).__name__}: {raw!r}")
    return json.loads(raw)


def _json_column(value: BaseModel) -> str:
    """Encode one nested record the way its ``TEXT`` column stores it."""
    return json.dumps(value.model_dump(by_alias=True), default=str, ensure_ascii=False)


def persist_history(  # noqa: PLR0917 — a history entry needs every run fact
    report: LintReport,
    *,
    vp: VaultPaths,
    config: Mapping[str, RawConfigValue],
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
        options=LintRunOptions(context=context, fix=fix),
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
    config: Mapping[str, RawConfigValue],
    node_count: int,
    llm: LintLlmMeta | None = None,
    timestamp: str | None = None,
) -> LintEntry:
    """Build a lint history entry from a lint run."""
    return LintEntry(
        timestamp=timestamp or iso_local(),
        duration_seconds=round(duration_seconds, 3),  # nocheck — lint-history schema, not WikiField
        options=options,
        llm=llm,
        config_snapshot=_build_config_snapshot(config),
        wiki_meta=LintWikiMeta(node_count=node_count),
        report=report_to_payload(report),
    )


def report_from_dict(report_dict: Mapping[str, JSONValue] | LintReportPayload) -> LintReport:
    """Parse a ``lint_runs`` report payload back into :class:`LintReport`.

    The inverse of :func:`kompend.lint.runner.report_to_payload`, and the one
    place history entries become typed findings again — the CLI and MCP each
    carried their own copy of this conversion and both dropped ``severity`` and
    ``fix_applied`` on the way in. Parsing is validated: a malformed payload
    raises instead of silently defaulting missing fields.
    """
    payload = report_dict if isinstance(report_dict, LintReportPayload) else LintReportPayload.model_validate(report_dict)
    return LintReport(
        results=[
            LintResult(
                rule=r.rule,
                outcome=LintRuleOutcome(r.outcome),
                severity=Severity(r.severity),
                message=r.message,
                page_id=r.page_id,
                fix_applied=r.fix_applied,
                details=details_from_storage(r.details),
            )
            for r in payload.results
        ],
        total_checks=payload.total_checks,
        failures=payload.failures,
        passes=payload.passes,
        skips=payload.skips,
        health_score=payload.health_score,
    )


def _build_config_snapshot(cfg: Mapping[str, RawConfigValue]) -> LintConfigSnapshot:
    """Extract the lint-relevant subset of a resolved config."""
    return LintConfigSnapshot(
        compiler=_config_text(cfg, COMPILE_COMPILER),
        default_model=_config_text(cfg, COMPILE_LLM_DEFAULT_MODEL),
        fallback_chain=_config_text_list(cfg, COMPILE_LLM_FALLBACK_CHAIN),
    )


def _config_text(cfg: Mapping[str, RawConfigValue], key: str) -> str | None:
    """Read one config value as text, refusing anything else.

    A resolved config carries JSON values or live ``Path`` objects (the
    ``paths`` section); the snapshot records the three keys the history view
    compares. A key holding a non-text value means the caller passed a
    different mapping, so it raises instead of recording a snapshot that
    renders as a wrong type.
    """
    value = cfg.get(key)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"{key} must be text in the config snapshot, not {type(value).__name__}: {value!r}")


def _config_text_list(cfg: Mapping[str, RawConfigValue], key: str) -> list[str] | None:
    """Read one config list value as text entries, refusing anything else."""
    value = cfg.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list in the config snapshot, not {type(value).__name__}: {value!r}")
    entries: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise TypeError(f"{key} entries must be text, not {type(entry).__name__}: {entry!r}")
        entries.append(entry)
    return entries


def save_lint_entry(entry: LintEntry, path: Path) -> None:
    """Append *entry* as one row in the ``lint_runs`` table for the vault at *path*."""
    with open_store(vault_db_path(path)) as conn, core_connection(conn) as core:
        core.execute(sqlite_insert(lint_runs).values(**entry.to_columns()))


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

    with open_store(db_path) as conn, core_connection(conn) as core:
        rows = core.execute(select(lint_runs.c.id, lint_runs.c.timestamp).order_by(lint_runs.c.id)).mappings().all()
        if not rows:
            return 0

        # The survivor computation stays in Python: stored timestamps carry
        # the machine's local offset, and only parsed datetimes compare
        # mixed offsets correctly. The DELETE itself is one statement.
        surviving = [(row["id"], row["timestamp"]) for row in rows]
        if max_age_days is not None:
            cutoff = now_local() - timedelta(days=max_age_days)
            surviving = [(rid, ts) for rid, ts in surviving if _parse_timestamp(ts) >= cutoff]
        if keep_last is not None:
            surviving = surviving[-keep_last:] if keep_last > 0 else []

        keep_ids = {rid for rid, _ in surviving}
        doomed = [row["id"] for row in rows if row["id"] not in keep_ids]
        if not doomed:
            return 0
        result = core.execute(delete(lint_runs).where(lint_runs.c.id.in_(doomed)))
        return result.rowcount


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
        with open_store(db_path) as conn, core_connection(conn) as core:
            if last_n is None:
                rows = core.execute(select(lint_runs).order_by(lint_runs.c.id)).mappings().all()
            else:
                rows = core.execute(select(lint_runs).order_by(lint_runs.c.id.desc()).limit(last_n)).mappings().all()
                rows = rows[::-1]
        entries = [LintEntry.from_row(row) for row in rows]

    return entries


def history_to_json(entries: list[LintEntry]) -> str:
    """Serialise a list of entries to a pretty JSON string."""
    payload = [entry.model_dump(mode="json", by_alias=True) for entry in entries]
    return json.dumps(payload, indent=2, ensure_ascii=False)


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
        report = entry.report
        score = report.health_score
        fails = report.failures
        passes = report.passes
        skips = report.skips
        llm = entry.llm
        model_tag = f"  [{llm.model}]" if llm else ""
        lines.append(f"  {entry.timestamp}  score={score:.0f}%  {passes} pass / {fails} fail / {skips} skip{model_tag}")

    if show_trend and len(entries) >= 2:
        scores = [entry.report.health_score for entry in entries]
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
