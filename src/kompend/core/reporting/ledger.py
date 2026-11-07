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
and close, so the store stays bounded without a maintenance command. A ledger
write must never break the command it observes: every store error is logged at
debug and swallowed.
"""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants.

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from kompend.core.reporting.handle import current_run_id
from kompend.core.store import connect, open_store
from kompend.core.store.schema import STAGES_TABLE
from kompend.core.types import ReasonCode, StageOutcome
from kompend.core.utils.time import iso_local
from kompend.logging import get_logger
from collections.abc import Generator
from typing import Any

_SOFT_CAP_ROWS = 5000
_KEEP_ROWS = 2500
#: Cap for the ``detail`` excerpt — enough to recognise an error, never a dump.
_DETAIL_CHARS = 240

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
            detail = detail[:_DETAIL_CHARS]
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
        try:
            with self._lock:
                conn.execute(
                    f"INSERT INTO {STAGES_TABLE} (run_id, ts, source, stage, attempt, outcome, duration_ms, "
                    f"reason_code, detail, parser, parser_version, backend, judge_model, cache_key, sha, size_bytes) "
                    f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )
                conn.commit()
        except sqlite3.Error as exc:  # never let the observer break the run
            logger.debug("stage ledger write failed: %s", exc)

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


def _prune(conn: sqlite3.Connection) -> None:
    """Trim the ledger to the last ``_KEEP_ROWS`` once it exceeds the cap."""
    count = conn.execute(f"SELECT COUNT(*) FROM {STAGES_TABLE}").fetchone()[0]
    if count <= _SOFT_CAP_ROWS:
        return
    conn.execute(
        f"DELETE FROM {STAGES_TABLE} WHERE id NOT IN (SELECT id FROM {STAGES_TABLE} ORDER BY id DESC LIMIT ?)",
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
        f"SELECT source, stage, duration_ms, outcome, reason_code FROM {STAGES_TABLE} "
        f"WHERE run_id = ? AND duration_ms IS NOT NULL ORDER BY duration_ms DESC LIMIT ?",
        (run_id, limit),
    ).fetchall()


def failure_histogram(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    """Per (stage, outcome, reason_code) counts of non-happy attempts in a run."""
    return conn.execute(
        f"SELECT stage, outcome, reason_code, COUNT(*) AS n FROM {STAGES_TABLE} "
        f"WHERE run_id = ? AND outcome NOT IN (?, ?) GROUP BY stage, outcome, reason_code ORDER BY n DESC",
        (run_id, StageOutcome.OK.value, StageOutcome.CACHED.value),
    ).fetchall()


def cache_hit_rate(conn: sqlite3.Connection, run_id: str) -> float | None:
    """Parse-cache hit rate of a run; ``None`` when no parse stage ran."""
    rows = conn.execute(
        f"SELECT outcome, COUNT(*) AS n FROM {STAGES_TABLE} WHERE run_id = ? AND stage = 'parse' GROUP BY outcome",
        (run_id,),
    ).fetchall()
    counts = {row["outcome"]: row["n"] for row in rows}
    parsed = counts.get(StageOutcome.CACHED.value, 0) + counts.get(StageOutcome.OK.value, 0)
    if parsed == 0:
        return None
    return counts.get(StageOutcome.CACHED.value, 0) / parsed
