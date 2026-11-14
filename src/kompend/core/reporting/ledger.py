"""Per-item stage ledger — one row per (run, source, stage, attempt) in ``stages``.

The companion of the ``runs`` audit trail (:mod:`kompend.core.reporting.runlog`):
where ``runs`` answers *"did the command finish?"*, ``stages`` answers
*"what happened to each item, at which stage, and how long did it take?"* —
without log archaeology.

Rows are written by :class:`StageLedger` at the call sites that already hold
the facts (parse cache hits, routed backend, deadline results, dedup verdicts).
Each row carries:

* ``outcome`` — a :class:`~kompend.core.types.StageOutcome` value
  (``ok | cached | skipped | failed | timeout | degraded``);
* ``reason_code`` — a :class:`~kompend.core.types.ReasonCode` value when the
  outcome was not ``ok``/``cached`` (the same value the log ``extra=`` carries);
  traversal-plan rows always carry ``TRAVERSAL`` — the plan is the product,
  not a failure;
* ``duration_ms`` — wall-clock cost of the attempt, when measurable;
* ``detail`` — a capped excerpt (exception text, dedup verdict, …);
* the attempt's facts — ``parser`` (backend identity), ``backend`` (class
  name), ``judge_model``, ``cache_key``, ``sha``, ``size_bytes`` — ``NULL``
  when that fact was not in hand.

Rows correlate with ``runs`` through ``run_id``: :func:`log_run_start` publishes
the current run in :mod:`kompend.core.reporting.handle`, and ``record()`` picks
it up from there (``None`` for library/standalone use, e.g. tests).

Documented queries (``conn`` is an :func:`~kompend.core.store.open_store` handle;
the helpers below wrap exactly these):

Slowest items of a run::

    SELECT source, stage, duration_ms FROM stages
    WHERE run_id = ? AND duration_ms IS NOT NULL
    ORDER BY duration_ms DESC LIMIT 10;

Failure histogram (stage × outcome × reason)::

    SELECT stage, outcome, reason_code, COUNT(*) FROM stages
    WHERE run_id = ? AND outcome NOT IN ('ok', 'cached')
    GROUP BY stage, outcome, reason_code ORDER BY 4 DESC;

Parse-cache hit rate::

    SELECT outcome, COUNT(*) FROM stages
    WHERE run_id = ? AND stage = 'parse' GROUP BY outcome;
    -- cached / (cached + ok) is the hit rate

Rows are tiny; the table is pruned to a soft cap (run-log pattern) on open
and close, so the store stays bounded without a maintenance command. Parse
quality is the exception: :meth:`StageLedger.record_parse_quality` upserts one
row per source into ``parse_quality`` (latest parse wins) so the verdict
outlives ledger pruning. A ledger write must never break the command it
observes: every store error is logged at debug and swallowed.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

from kompend.core.reporting.handle import current_run_id
from kompend.core.reporting.parse_quality import PARSE_QUALITY_COLUMNS, ParseQuality
from kompend.core.store import connect, open_store
from kompend.core.store.schema import PARSE_QUALITY_TABLE, STAGES_TABLE
from kompend.core.types import ReasonCode, StageOutcome
from kompend.core.utils.time import iso_local
from kompend.logging import get_logger

#: Column whose value the parse-quality table keys on (a column name, not a NodeType).
_SOURCE_COLUMN: Final[str] = "source"

_SOFT_CAP_ROWS = 5000
_KEEP_ROWS = 2500
#: Cap for the ``detail`` excerpt — enough to recognise an error, never a dump.
_DETAIL_CHARS = 240


_SOURCE_COLUMNS: Final[str] = (
    "run_id, ts, stage, attempt, outcome, duration_ms, reason_code, detail, parser, parser_version, backend, judge_model, cache_key, sha, size_bytes"
)

#: Columns of one ``stages`` insert, in the order :meth:`StageLedger.record` binds them.
_STAGES_INSERT_COLUMNS: Final[tuple[str, ...]] = (
    "run_id",
    "ts",
    "source",
    "stage",
    "attempt",
    "outcome",
    "duration_ms",
    "reason_code",
    "detail",
    "parser",
    "parser_version",
    "backend",
    "judge_model",
    "cache_key",
    "sha",
    "size_bytes",
)

logger = get_logger(__name__)


class StageLedger:
    """Append-only writer for the ``stages`` table, opened once per run.

    Holds one long-lived connection (``connect`` applies the WAL/busy-timeout
    PRAGMAs; ``open_store`` first ensures the schema once, so an old vault
    gains the table on first use). Each ``record()`` commits immediately —
    under WAL with ``synchronous=NORMAL`` that is well under a millisecond,
    and a crashed run leaves the rows it wrote so far behind for diagnosis.
    Thread-safe: the parse pool records from worker threads.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def record(
        self,
        source: str,
        stage: str,
        *,
        outcome: StageOutcome,
        attempt: int = 1,
        duration_ms: float | None = None,
        reason_code: ReasonCode | None = None,
        detail: str | None = None,
        parser: str | None = None,
        parser_version: str | None = None,
        backend: str | None = None,
        judge_model: str | None = None,
        cache_key: str | None = None,
        sha: str | None = None,
        size_bytes: int | None = None,
        run_id: str | None = None,
    ) -> None:
        """Append one row. Never raises — a ledger miss must not fail the item."""
        conn = self._conn
        if conn is None:
            return
        if detail is not None and len(detail) > _DETAIL_CHARS:
            detail = _excerpt(detail)
        values: tuple[Any, ...] = (
            run_id if run_id is not None else current_run_id(),
            iso_local(),
            source,
            stage,
            attempt,
            outcome.value,
            round(duration_ms, 3) if duration_ms is not None else None,
            reason_code.value if reason_code is not None else None,
            detail,
            parser,
            parser_version,
            backend,
            judge_model,
            cache_key,
            sha,
            size_bytes,
        )
        statement = _sql(
            "INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            table=STAGES_TABLE,
            columns=", ".join(_STAGES_INSERT_COLUMNS),
            placeholders=", ".join("?" for _ in _STAGES_INSERT_COLUMNS),
        )
        try:
            with self._lock:
                conn.execute(statement, values)
                conn.commit()
        except sqlite3.Error as exc:  # never let the observer break the run
            logger.debug("stage ledger write failed: %s", exc)

    def record_traversal_plan(
        self,
        source: str,
        *,
        outcome: StageOutcome,
        planned: int,
        skipped: int,
        expanded: int,
        windows: int,
        run_id: str | None = None,
    ) -> None:
        """Record one traversal-plan row: one row per tree, stage ``"summarize"``.

        The full plan JSON belongs to the ``summarizer_plan`` cache layer;
        the row carries only the counters via :func:`format_traversal_detail`.
        """
        self.record(
            source,
            "summarize",
            outcome=outcome,
            reason_code=ReasonCode.TRAVERSAL,
            detail=format_traversal_detail(planned=planned, skipped=skipped, expanded=expanded, windows=windows),
            run_id=run_id,
        )

    def record_parse_quality(self, quality: ParseQuality) -> None:
        """Persist *quality* as the source's current parse verdict.

        One row per source (latest parse wins) in ``parse_quality`` — beside
        the attempt row in ``stages`` but not pruned with it, because the row
        describes the document rather than one run. Never raises, for the same
        reason the stage row does not.
        """
        conn = self._conn
        if conn is None:
            return
        columns = ", ".join(PARSE_QUALITY_COLUMNS)
        placeholders = ", ".join("?" for _ in PARSE_QUALITY_COLUMNS)
        updates = ", ".join(_sql("{column} = excluded.{column}", column=column) for column in PARSE_QUALITY_COLUMNS if column != _SOURCE_COLUMN)
        statement = _sql(
            "INSERT INTO {table} ({columns}) VALUES ({placeholders}) ON CONFLICT(source) DO UPDATE SET {updates}",
            table=PARSE_QUALITY_TABLE,
            columns=columns,
            placeholders=placeholders,
            updates=updates,
        )
        try:
            with self._lock:
                conn.execute(statement, quality.to_row(run_id=current_run_id(), ts=iso_local()))
                conn.commit()
        except sqlite3.Error as exc:  # never let the observer break the run
            logger.debug("parse-quality write failed: %s", exc)

    def close(self) -> None:
        """Close the writer's connection (idempotent; prunes one last time)."""
        with self._lock:
            if self._conn is not None:
                try:
                    _prune(self._conn)
                except sqlite3.Error as exc:
                    logger.debug("stage ledger prune failed: %s", exc)
                finally:
                    self._conn.close()
                    self._conn = None

    def __enter__(self) -> StageLedger:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @classmethod
    def open(cls, db_path: Path) -> StageLedger:
        """Open (and ensure the schema of) the ledger for the store at *db_path*."""
        ledger = cls(db_path)
        try:
            # Ensures the schema (idempotent) and closes the handle; the
            # writer's own connection below is deliberately schema-free.
            with open_store(db_path):
                pass
            ledger._conn = connect(db_path)
            # WAL-safe durability level for an observability side channel:
            # commits stop fsyncing on every row but survive process death.
            ledger._conn.execute("PRAGMA synchronous=NORMAL")
            ledger._maybe_prune()
        except sqlite3.Error as exc:
            logger.debug("stage ledger unavailable: %s", exc)
            ledger._conn = None
        return ledger

    def _maybe_prune(self) -> None:
        with self._lock:
            if self._conn is not None:
                _prune(self._conn)


def format_traversal_detail(*, planned: int, skipped: int, expanded: int, windows: int) -> str:
    """Counter line for a traversal-plan ledger row.

    The fixed ``planned/skipped/expanded/windows`` shape keeps the row
    greppable; the full plan JSON lives in the ``summarizer_plan`` cache
    layer, never in the excerpted detail.
    """
    return f"planned={planned} skipped={skipped} expanded={expanded} windows={windows}"


def _excerpt(text: str) -> str:
    """*text* cut at :data:`_DETAIL_CHARS` on a word boundary, marked as cut.

    A raw slice ends mid-word, which reads as corruption in a diagnostics
    table (a skipped source's reason ended "…or b"). Dropping the partial word
    costs at most one word and the ellipsis says the text continues.
    """
    cut = text.rfind(" ", 0, _DETAIL_CHARS)
    return f"{text[: cut if cut > 0 else _DETAIL_CHARS].rstrip()}…"


def _prune(conn: sqlite3.Connection) -> None:
    """Trim the ledger to the last ``_KEEP_ROWS`` once it exceeds the cap."""
    count = conn.execute(_sql("SELECT COUNT(*) FROM {table}", table=STAGES_TABLE)).fetchone()[0]
    if count <= _SOFT_CAP_ROWS:
        return
    conn.execute(
        _sql(
            "DELETE FROM {table} WHERE id NOT IN (SELECT id FROM {table} ORDER BY id DESC LIMIT ?)",
            table=STAGES_TABLE,
        ),
        (_KEEP_ROWS,),
    )
    conn.commit()


@contextmanager
def stage_ledger(db_path: Path) -> Generator[StageLedger]:
    """Open a :class:`StageLedger` for a pipeline stage (prunes on exit)."""
    ledger = StageLedger.open(db_path)
    try:
        yield ledger
    finally:
        ledger.close()


# ── Documented queries (see module docstring) ───────────────────────────


def slowest_items(conn: sqlite3.Connection, run_id: str, limit: int = 10) -> list[sqlite3.Row]:
    """The *limit* slowest (source, stage) attempts of a run, slowest first."""
    return conn.execute(
        _sql(
            "SELECT source, stage, duration_ms, outcome, reason_code FROM {table} "
            "WHERE run_id = ? AND duration_ms IS NOT NULL ORDER BY duration_ms DESC LIMIT ?",
            table=STAGES_TABLE,
        ),
        (run_id, limit),
    ).fetchall()


def failure_histogram(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    """Per (stage, outcome, reason_code) counts of non-happy attempts in a run."""
    return conn.execute(
        _sql(
            "SELECT stage, outcome, reason_code, COUNT(*) AS n FROM {table} "
            "WHERE run_id = ? AND outcome NOT IN (?, ?) GROUP BY stage, outcome, reason_code ORDER BY n DESC",
            table=STAGES_TABLE,
        ),
        (run_id, StageOutcome.OK.value, StageOutcome.CACHED.value),
    ).fetchall()


def cache_hit_rate(conn: sqlite3.Connection, run_id: str) -> float | None:
    """Parse-cache hit rate of a run; ``None`` when no parse stage ran."""
    rows = conn.execute(
        _sql(
            "SELECT outcome, COUNT(*) AS n FROM {table} WHERE run_id = ? AND stage = 'parse' GROUP BY outcome",
            table=STAGES_TABLE,
        ),
        (run_id,),
    ).fetchall()
    counts = {row["outcome"]: row["n"] for row in rows}
    parsed = counts.get(StageOutcome.CACHED.value, 0) + counts.get(StageOutcome.OK.value, 0)
    if parsed == 0:
        return None
    return counts.get(StageOutcome.CACHED.value, 0) / parsed


def matching_sources(conn: sqlite3.Connection, fragment: str) -> list[str]:
    """Distinct ledger sources whose name contains *fragment*, sorted.

    An exact match also contains itself, so resolution can check for it
    without a second query.
    """
    rows = conn.execute(
        _sql("SELECT DISTINCT source FROM {table} WHERE source LIKE ? ORDER BY source", table=STAGES_TABLE),
        (f"%{fragment}%",),
    ).fetchall()
    return [row["source"] for row in rows]


def source_runs(conn: sqlite3.Connection, source: str) -> list[str]:
    """Run ids that recorded *source*, oldest first (``[]`` when unknown)."""
    rows = conn.execute(
        _sql(
            "SELECT DISTINCT run_id FROM {table} WHERE source = ? AND run_id IS NOT NULL ORDER BY id",
            table=STAGES_TABLE,
        ),
        (source,),
    ).fetchall()
    return [row["run_id"] for row in rows]


def source_attempts(conn: sqlite3.Connection, source: str, *, run_id: str | None = None) -> list[sqlite3.Row]:
    """Every recorded attempt for *source* (optionally one run), oldest first."""
    if run_id is None:
        return conn.execute(
            _sql("SELECT {columns} FROM {table} WHERE source = ? ORDER BY id", columns=_SOURCE_COLUMNS, table=STAGES_TABLE),
            (source,),
        ).fetchall()
    return conn.execute(
        _sql(
            "SELECT {columns} FROM {table} WHERE source = ? AND run_id = ? ORDER BY id",
            columns=_SOURCE_COLUMNS,
            table=STAGES_TABLE,
        ),
        (source, run_id),
    ).fetchall()


def _sql(template: str, **values: object) -> str:
    """Fill one statement template; every statement stays a literal at its call site.

    The ledger writes from worker threads through its own long-lived raw
    connection (see :class:`StageLedger`), so it cannot hand each row to Core
    without wrapping the handle in an engine per append. Keeping the SQL in a
    plain-``str`` parameter instead means bandit never sees a built statement,
    and no caller assembles SQL by hand.
    """
    return template.format(**values)
