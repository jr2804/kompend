"""Per-document parse quality — the signals a bad parse leaves behind.

A parse that "succeeds" can still be garbage: a PDF renamed to ``.md`` decodes
to replacement characters instead of failing, and every downstream stage
happily builds a wiki from the soup (knox-rtmn, knox-ty8f.10). The parser
cannot know what the document was supposed to contain, but the *shape* of its
output is measurable:

* ``replacement_ratio`` — share of U+FFFD in the extracted text;
* ``empty_blocks`` — elements whose text is blank;
* ``block_coverage`` — share of elements that produced text;
* ``chars_per_page`` — extracted characters per declared page.

These are persisted one row per source in the ``parse_quality`` table, written
by :meth:`kompend.core.reporting.ledger.StageLedger.record_parse_quality` at
the parse call sites (the same places that write the ``stages`` row). A source
re-parsed later replaces its previous row, so a query always reads the latest
verdict.

The flag is deliberately a *suspicion*: the numbers diagnose, the user decides.
No parse is rejected and no run fails because of a quality row.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from kompend.core.models import ParsedDocument, element_text
from kompend.core.store.schema import PARSE_QUALITY_TABLE

#: Replacement characters are suspicious only past this share of the text.
REPLACEMENT_RATIO_LIMIT: Final[float] = 0.01
#: A document this small cannot support a ratio verdict either way.
MIN_TEXT_CHARS_FOR_RATIO: Final[int] = 200
#: Share of blank elements that suggests the backend extracted nothing.
EMPTY_BLOCK_RATIO_LIMIT: Final[float] = 0.60
#: Fewer elements than this make the empty-block share meaningless.
MIN_BLOCKS_FOR_EMPTY: Final[int] = 4
#: A page scan with almost no text and some replacement characters.
MIN_CHARS_PER_PAGE: Final[float] = 40.0

#: Column order for the persisted row; kept beside the dataclass so the
#: INSERT in the ledger and the SELECT here cannot drift.
PARSE_QUALITY_COLUMNS: Final[tuple[str, ...]] = (
    "source",
    "sha",
    "run_id",
    "ts",
    "parser",
    "backend",
    "page_count",
    "text_chars",
    "replacement_chars",
    "replacement_ratio",
    "empty_blocks",
    "block_count",
    "block_coverage",
    "chars_per_page",
    "suspicious",
)


@dataclass(frozen=True, slots=True)
class ParseQuality:
    """The measured quality of one parsed document."""

    source: str
    sha: str
    parser: str
    backend: str
    page_count: int
    text_chars: int
    replacement_chars: int
    replacement_ratio: float
    empty_blocks: int
    block_count: int
    block_coverage: float
    chars_per_page: float
    suspicious: bool

    def to_dict(self) -> dict[str, object]:
        """JSON-ready record for status output and ledger reads."""
        return {column: getattr(self, column) for column in PARSE_QUALITY_COLUMNS}

    def to_row(self, *, run_id: str | None, ts: str) -> tuple[object, ...]:
        """Values in :data:`PARSE_QUALITY_COLUMNS` order, for the ledger INSERT."""
        return (
            self.source,
            self.sha,
            run_id,
            ts,
            self.parser,
            self.backend,
            self.page_count,
            self.text_chars,
            self.replacement_chars,
            self.replacement_ratio,
            self.empty_blocks,
            self.block_count,
            self.block_coverage,
            self.chars_per_page,
            int(self.suspicious),
        )


def measure_parse_quality(
    document: ParsedDocument,
    *,
    source: str,
    sha: str = "",
    parser: str = "",
    backend: str = "",
) -> ParseQuality:
    """Measure *document* and return the persisted record.

    *source* is the vault-relative spelling used by the ledger, so a status
    query can join the row to the source tree without path translation.
    """
    texts = [element_text(element.content) for element in document.elements]
    text = "\n".join(texts)
    text_chars = len(text)
    replacement_chars = text.count("\ufffd")
    replacement_ratio = replacement_chars / text_chars if text_chars else 0.0
    block_count = len(texts)
    empty_blocks = sum(1 for block in texts if not block.strip())
    block_coverage = (block_count - empty_blocks) / block_count if block_count else 1.0
    page_count = max(document.page_count, 0)
    chars_per_page = text_chars / page_count if page_count else 0.0
    suspicious = _is_suspicious(
        text_chars=text_chars,
        replacement_chars=replacement_chars,
        replacement_ratio=replacement_ratio,
        empty_blocks=empty_blocks,
        block_count=block_count,
        page_count=page_count,
        chars_per_page=chars_per_page,
    )
    return ParseQuality(
        source=source,
        sha=sha,
        parser=parser,
        backend=backend,
        page_count=page_count,
        text_chars=text_chars,
        replacement_chars=replacement_chars,
        replacement_ratio=replacement_ratio,
        empty_blocks=empty_blocks,
        block_count=block_count,
        block_coverage=block_coverage,
        chars_per_page=chars_per_page,
        suspicious=suspicious,
    )


def _is_suspicious(
    *,
    text_chars: int,
    replacement_chars: int,
    replacement_ratio: float,
    empty_blocks: int,
    block_count: int,
    page_count: int,
    chars_per_page: float,
) -> bool:
    """Classify a measurement as worth surfacing.

    A suspicion needs the shape of a failed extraction, not one odd character:
    the ratio needs enough text to mean anything, the empty-block share needs
    enough blocks, and the page test needs both pages and replacement
    characters (a legitimate mostly-blank page is not garbage).
    """
    if text_chars == 0 and block_count > 0:
        return True
    if text_chars >= MIN_TEXT_CHARS_FOR_RATIO and replacement_ratio >= REPLACEMENT_RATIO_LIMIT:
        return True
    if block_count >= MIN_BLOCKS_FOR_EMPTY and empty_blocks / block_count >= EMPTY_BLOCK_RATIO_LIMIT:
        return True
    return page_count > 0 and chars_per_page < MIN_CHARS_PER_PAGE and replacement_chars > 0


def read_parse_quality(db_path: Path) -> list[ParseQuality]:
    """Read every persisted parse-quality row, most suspicious first.

    Never creates the store: a missing file or an old store without the table
    yields an empty list, so ``kompend status`` stays read-only.
    """
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT {', '.join(PARSE_QUALITY_COLUMNS)} FROM {PARSE_QUALITY_TABLE} "  # noqa: S608 — module constants
            "ORDER BY suspicious DESC, replacement_ratio DESC, empty_blocks DESC, source"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [_from_row(row) for row in rows]


def _from_row(row: sqlite3.Row) -> ParseQuality:
    return ParseQuality(
        source=str(row["source"]),
        sha=str(row["sha"]),
        parser=str(row["parser"] or ""),
        backend=str(row["backend"] or ""),
        page_count=int(row["page_count"]),
        text_chars=int(row["text_chars"]),
        replacement_chars=int(row["replacement_chars"]),
        replacement_ratio=float(row["replacement_ratio"]),
        empty_blocks=int(row["empty_blocks"]),
        block_count=int(row["block_count"]),
        block_coverage=float(row["block_coverage"]),
        chars_per_page=float(row["chars_per_page"]),
        suspicious=bool(row["suspicious"]),
    )
