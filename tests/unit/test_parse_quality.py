"""Parse quality metrics: measurement, suspicion, persistence, read-back."""

from __future__ import annotations

from pathlib import Path

import pytest

from kompend.core.models import DocumentElement, ElementType, ParsedDocument
from kompend.core.reporting.ledger import StageLedger
from kompend.core.reporting.parse_quality import (
    PARSE_QUALITY_COLUMNS,
    ParseQuality,
    measure_parse_quality,
    read_parse_quality,
)


def _doc(*blocks: str, pages: int = 1) -> ParsedDocument:
    return ParsedDocument(
        source_file=Path("paper.md"),
        elements=[DocumentElement(type=ElementType.TEXT, content=block) for block in blocks],
        page_count=pages,
    )


def _quality(doc: ParsedDocument, *, source: str = "paper.md", sha: str = "abc") -> ParseQuality:
    return measure_parse_quality(doc, source=source, sha=sha, parser="docling", backend="Docling")


def test_replacement_ratio_flags_a_garbled_document() -> None:
    doc = _doc("a" * 100 + "\ufffd" * 10, "b" * 100 + "\ufffd" * 10)

    quality = _quality(doc)

    assert quality.text_chars == 221  # 110 + 110 + the "\n".join separator
    assert quality.replacement_chars == 20
    assert quality.replacement_ratio == pytest.approx(20 / 221)
    assert quality.suspicious is True


def test_small_text_is_not_judged_by_ratio() -> None:
    doc = _doc("a\ufffd", pages=0)

    quality = _quality(doc)

    assert quality.replacement_ratio == pytest.approx(0.5)
    assert quality.suspicious is False


def test_empty_blocks_flag_a_failed_extraction() -> None:
    doc = _doc("only this block has text", "", "   ", "", "")

    quality = _quality(doc)

    assert quality.block_count == 5
    assert quality.empty_blocks == 4
    assert quality.block_coverage == pytest.approx(1 / 5)
    assert quality.suspicious is True


def test_healthy_document_is_clean() -> None:
    doc = _doc("a" * 100, "b" * 100, "c" * 100, pages=2)

    quality = _quality(doc)

    assert quality.replacement_ratio == 0.0
    assert quality.empty_blocks == 0
    assert quality.block_coverage == 1.0
    assert quality.chars_per_page == pytest.approx(151.0)  # 302 joined chars / 2 pages
    assert quality.suspicious is False


def test_to_row_follows_the_column_contract() -> None:
    quality = _quality(_doc("a" * 300), sha="sha")

    row = quality.to_row(run_id="run-1", ts="2026-11-08T12:00:00")

    assert len(row) == len(PARSE_QUALITY_COLUMNS)
    assert row[PARSE_QUALITY_COLUMNS.index("source")] == "paper.md"
    assert row[PARSE_QUALITY_COLUMNS.index("run_id")] == "run-1"
    assert row[-1] == 0


def test_ledger_round_trip_keeps_the_latest_row(tmp_path: Path) -> None:
    db_path = tmp_path / "kompend.db"
    ledger = StageLedger.open(db_path)
    try:
        ledger.record_parse_quality(_quality(_doc("a" * 300, "\ufffd" * 300)))
        ledger.record_parse_quality(_quality(_doc("a" * 400)))
        ledger.record_parse_quality(_quality(_doc("b\ufffd"), source="other.md"))
    finally:
        ledger.close()

    rows = read_parse_quality(db_path)

    assert len(rows) == 2
    assert rows[0].source == "other.md"
    assert rows[0].suspicious is True
    current = next(row for row in rows if row.source == "paper.md")
    assert current.text_chars == 400
    assert current.suspicious is False


def test_missing_store_reads_as_empty(tmp_path: Path) -> None:
    assert read_parse_quality(tmp_path / "does-not-exist.db") == []
