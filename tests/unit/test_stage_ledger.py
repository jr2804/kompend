"""The per-item stage ledger — table, writer, queries, pruning, overhead."""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import select
from sqlalchemy.engine import RowMapping

import kompend.core.reporting.ledger as ledger_mod
from kompend.core.models import ParsedDocument
from kompend.core.pipeline.ingest import ingest_documents
from kompend.core.reporting.handle import set_current_run_id
from kompend.core.reporting.ledger import (
    StageLedger,
    cache_hit_rate,
    failure_histogram,
    format_traversal_detail,
    slowest_items,
    stage_ledger,
)
from kompend.core.store import core_connection, open_store
from kompend.core.store.tables import stages
from kompend.core.types import ReasonCode, StageOutcome
from kompend.preprocessing.parsers.base import DocumentParser

# --- ingest integration ---------------------------------------------------------


class _LedgerBackend:
    """Trivial backend: any file parses to an empty document."""

    def parse(self, target: Path) -> ParsedDocument:
        return ParsedDocument(source_file=Path(target), elements=[], title=Path(target).stem)


class _LedgerParser:
    """Parser facade satisfying ingest_documents' routing contract."""

    def backend_identity(self) -> str:
        return "stub-backend|1.0"

    def route(self, source_file: Path) -> _LedgerBackend:  # noqa: ARG002
        return _LedgerBackend()


def test_record_roundtrips_every_column(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with stage_ledger(db) as ledger:
        ledger.record(
            "notes/report.md",
            "parse",
            outcome=StageOutcome.FAILED,
            attempt=2,
            duration_ms=123.456,
            reason_code=ReasonCode.TIMEOUT,
            detail="deadline exceeded",
            parser="docling|judge:balanced",
            parser_version="0.9",
            backend="ParsecraftParser",
            judge_model="gpt-judge",
            cache_key="report-a1b2c3d4",
            sha="a" * 64,
            size_bytes=1024,
            run_id="deadbeef",
        )
    rows = _rows(db, "deadbeef")
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "notes/report.md"
    assert row["stage"] == "parse"
    assert row["attempt"] == 2
    assert row["outcome"] == "failed"
    assert row["duration_ms"] == pytest.approx(123.456)
    assert row["reason_code"] == "timeout"
    assert row["detail"] == "deadline exceeded"
    assert row["parser"] == "docling|judge:balanced"
    assert row["parser_version"] == "0.9"
    assert row["backend"] == "ParsecraftParser"
    assert row["judge_model"] == "gpt-judge"
    assert row["cache_key"] == "report-a1b2c3d4"
    assert row["sha"] == "a" * 64
    assert row["size_bytes"] == 1024
    assert row["ts"]


def test_run_id_defaults_to_the_current_run_context(tmp_path: Path) -> None:
    db = _db(tmp_path)
    set_current_run_id("ctx42")
    try:
        with stage_ledger(db) as ledger:
            ledger.record("a.md", "parse", outcome=StageOutcome.OK)
    finally:
        set_current_run_id(None)
    rows = _rows(db, "ctx42")
    assert len(rows) == 1
    assert rows[0]["outcome"] == "ok"
    assert rows[0]["run_id"] == "ctx42"


def test_detail_excerpt_is_capped_on_a_word_boundary(tmp_path: Path) -> None:
    # A raw slice ended mid-word (a skipped source read "…or b"); the excerpt
    # keeps whole words and says it continues.
    db = _db(tmp_path)
    detail = " ".join(f"word{index}" for index in range(120))
    with stage_ledger(db) as ledger:
        ledger.record("big.md", "parse", outcome=StageOutcome.FAILED, detail=detail, run_id="r")
    stored = _rows(db, "r")[0]["detail"]
    assert stored.endswith("…")
    assert len(stored) <= 241
    assert stored[:-1] == detail[: len(stored) - 1]
    assert not stored[:-1].endswith("wor")


def test_detail_excerpt_without_spaces_is_a_hard_cut(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with stage_ledger(db) as ledger:
        ledger.record("big.md", "parse", outcome=StageOutcome.FAILED, detail="x" * 10_000, run_id="r")
    row = _rows(db, "r")[0]
    assert len(row["detail"]) == 241
    assert row["detail"].endswith("…")


def test_closed_ledger_records_are_silent_no_ops(tmp_path: Path) -> None:
    db = _db(tmp_path)
    ledger = StageLedger.open(db)
    ledger.close()
    ledger.record("a.md", "parse", outcome=StageOutcome.OK, run_id="r")  # must not raise
    ledger.close()  # idempotent
    assert _rows(db, "r") == []


def test_prune_keeps_the_newest_rows_beyond_the_soft_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ledger_mod, "_SOFT_CAP_ROWS", 10)
    monkeypatch.setattr(ledger_mod, "_KEEP_ROWS", 5)
    db = _db(tmp_path)
    with stage_ledger(db) as ledger:
        for i in range(12):
            ledger.record(f"f{i}.md", "parse", outcome=StageOutcome.OK, run_id="r")
    with open_store(db) as conn, core_connection(conn) as core:
        sources = [row["source"] for row in core.execute(select(stages.c.source).order_by(stages.c.id)).mappings()]
    assert sources == [f"f{i}.md" for i in range(7, 12)]


def test_documented_queries_answer_slowest_failures_and_cache_rate(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with stage_ledger(db) as ledger:
        ledger.record("slow.md", "parse", outcome=StageOutcome.OK, duration_ms=900, run_id="r")
        ledger.record("fast.md", "parse", outcome=StageOutcome.CACHED, duration_ms=10, run_id="r")
        ledger.record("bad.md", "parse", outcome=StageOutcome.FAILED, duration_ms=500, reason_code=ReasonCode.BACKEND_UNAVAILABLE, run_id="r")
        ledger.record("bad.md", "chunk", outcome=StageOutcome.TIMEOUT, duration_ms=400, reason_code=ReasonCode.TIMEOUT, run_id="r")
        ledger.record("other.md", "parse", outcome=StageOutcome.OK, run_id="other-run")
    with open_store(db) as conn:
        slowest = slowest_items(conn, "r", limit=2)
        assert [(row["source"], row["stage"]) for row in slowest] == [("slow.md", "parse"), ("bad.md", "parse")]

        histogram = failure_histogram(conn, "r")
        assert {(row["stage"], row["outcome"], row["reason_code"], row["n"]) for row in histogram} == {
            ("parse", "failed", "backend_unavailable", 1),
            ("chunk", "timeout", "timeout", 1),
        }

        assert cache_hit_rate(conn, "r") == pytest.approx(1 / 2)
        assert cache_hit_rate(conn, "other-run") == pytest.approx(0.0)
        assert cache_hit_rate(conn, "never-ran") is None


def test_ledger_survives_a_run_that_never_closed(tmp_path: Path) -> None:
    """Rows committed per record survive a crashed run (no close/prune)."""
    db = _db(tmp_path)
    ledger = StageLedger.open(db)
    ledger.record("crash.md", "parse", outcome=StageOutcome.OK, run_id="r")
    del ledger  # no close() — the process "died"
    assert len(_rows(db, "r")) == 1


def test_record_overhead_stays_under_one_millisecond_per_item(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with stage_ledger(db) as ledger:
        samples = 200
        start = time.perf_counter()
        for i in range(samples):
            ledger.record(f"f{i}.md", "parse", outcome=StageOutcome.OK, duration_ms=1.0, run_id="perf")
        elapsed = time.perf_counter() - start
    mean_ms = elapsed / samples * 1000.0
    # The acceptance bound is ~1 ms per item; measured locally at tens of µs.
    # The 4× headroom absorbs slow CI disks without lying about the budget.
    assert mean_ms < 1.0, f"mean ledger write {mean_ms:.3f} ms exceeds the 1 ms budget"


def test_a_broken_store_never_breaks_the_command(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with stage_ledger(db) as ledger:
        ledger.record("ok.md", "parse", outcome=StageOutcome.OK, run_id="r")
        # Simulate the connection dying under the writer: every execute on a
        # closed sqlite handle raises (ProgrammingError ⊂ sqlite3.Error).
        assert ledger._conn is not None
        ledger._conn.close()
        ledger.record("broken.md", "parse", outcome=StageOutcome.OK, run_id="r")  # swallowed
    assert len(_rows(db, "r")) == 1


def test_ingest_documents_fills_the_ledger(tmp_path: Path) -> None:
    """Acceptance: a real ingest run writes one ok-row per parsed source,
    correlated with the ambient run id (the same one runlog assigns).
    """
    base = tmp_path / "vault"
    (base / "sources").mkdir(parents=True)
    (base / ".kompend").mkdir()
    (base / ".kompend" / "config.toml").write_text("", encoding="utf-8")
    src = base / "sources" / "note.md"
    src.write_bytes(b"# note\n")  # bytes: text mode would translate to CRLF and change size_bytes

    set_current_run_id("ing42")
    try:
        result = ingest_documents([src], base, cast(DocumentParser, _LedgerParser()))
    finally:
        set_current_run_id(None)

    assert result.count == 1
    rows = _rows(_db(base), "ing42")
    assert [(row["stage"], row["outcome"]) for row in rows] == [("parse", "ok")]
    row = rows[0]
    assert row["source"].endswith("note.md")
    assert row["parser"] == "stub-backend|1.0"
    assert row["backend"] == "_LedgerBackend"
    assert len(row["sha"]) == 64
    assert row["size_bytes"] == len(b"# note\n")


def test_the_ledger_carries_no_ruff_waiver() -> None:
    """A module-wide waiver hid eleven statements from bandit; it must stay gone.

    Deleting it exposed eleven string-built statements — every write, the prune
    and all six documented queries. They now go through one plain-``str``
    template helper, so bandit sees no built statement; a waiver over the module
    would let the next hand-built one land unseen.
    """
    source = Path(ledger_mod.__file__).read_text(encoding="utf-8")
    assert "ruff: noqa" not in source
    assert "noqa: S608" not in source


def test_traversal_plan_run_writes_one_summarize_row(tmp_path: Path) -> None:
    """knox-u84v.3: a plan run writes exactly one summarize row with counters."""
    db = _db(tmp_path)
    with stage_ledger(db) as ledger:
        ledger.record_traversal_plan(
            "notes/report.md",
            outcome=StageOutcome.OK,
            planned=12,
            skipped=3,
            expanded=9,
            windows=4,
            run_id="plan1",
        )
    rows = _rows(db, "plan1")
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "notes/report.md"
    assert row["stage"] == "summarize"
    assert row["outcome"] == "ok"
    assert row["reason_code"] == ReasonCode.TRAVERSAL.value
    assert row["detail"] == "planned=12 skipped=3 expanded=9 windows=4"
    assert format_traversal_detail(planned=12, skipped=3, expanded=9, windows=4) == row["detail"]


def test_traversal_plan_partial_outcome(tmp_path: Path) -> None:
    """A partially-covered tree records PARTIAL, same shape, same row count."""
    db = _db(tmp_path)
    with stage_ledger(db) as ledger:
        ledger.record_traversal_plan(
            "notes/report.md",
            outcome=StageOutcome.PARTIAL,
            planned=12,
            skipped=3,
            expanded=9,
            windows=4,
            run_id="plan2",
        )
    rows = _rows(db, "plan2")
    assert len(rows) == 1
    assert rows[0]["outcome"] == "partial"
    assert rows[0]["reason_code"] == ReasonCode.TRAVERSAL.value


def _db(tmp_path: Path) -> Path:
    return tmp_path / ".kompend" / "kompend.db"


def _rows(db: Path, run_id: str) -> list[RowMapping]:
    """Every ledger row of *run_id*, oldest first, as materialised mappings."""
    with open_store(db) as conn, core_connection(conn) as core:
        statement = select(stages).where(stages.c.run_id == run_id).order_by(stages.c.id)
        return list(core.execute(statement).mappings().all())
