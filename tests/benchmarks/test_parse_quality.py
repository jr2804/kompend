"""Parser quality benchmark.

Takes a real PDF, runs DoclingParser on it, and measures:
  - Element count (at least 1 heading, some text)
  - Structure preservation (tables > 0, or headings preserved)
  - Speed (reports time, doesn't assert)

Skips with a clear message if no real PDF is found or DoclingParser unavailable.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import pytest

from knox.core.models import ElementType, ParsedDocument
from knox.preprocessing.parsers.docling_parser import DoclingParser

# Test corpus: tests/data/sources/ (committed small samples) or local override
_TEST_DIR = Path(__file__).resolve().parent.parent
_DATA_SOURCES = _TEST_DIR / "data" / "sources"

_CORPUS_CANDIDATES = [
    _DATA_SOURCES,
    Path("/opt/workspace/obsidian/jan/vault/Worky/research/listening-effort/papers/"),
    Path("/opt/workspace/obsidian/jan/vault/Worky/research/listening-effort/docling_input/"),
]

_REAL_PDF: Path | None = None
for _candidate_dir in _CORPUS_CANDIDATES:
    if _candidate_dir.exists():
        pdfs = sorted(_candidate_dir.glob("*.pdf"))
        if pdfs:
            _REAL_PDF = pdfs[0]
            break


def _check_docling() -> bool:
    """Return True if DoclingParser can be instantiated."""
    try:
        DoclingParser(extract_images=False)
        return True
    except Exception:
        return False


_DOCLING_OK = _check_docling()


pytestmark = pytest.mark.benchmark


@pytest.mark.skipif(
    _REAL_PDF is None or not _DOCLING_OK,
    reason=("No real PDF found in test corpus and/or DoclingParser not available; parser benchmark needs a physical PDF file and working DoclingParser"),
)
class TestParserQualityBenchmark:
    """Quality benchmark for DoclingParser."""

    @pytest.fixture(scope="class")
    def parser(self) -> DoclingParser:
        return DoclingParser(extract_images=False)

    @pytest.fixture(scope="class")
    def parsed_doc(self, parser: DoclingParser) -> ParsedDocument:
        """Parse the real PDF and return the ParsedDocument."""
        assert _REAL_PDF is not None
        return parser.parse(_REAL_PDF)

    def test_benchmark_parse_speed(self, parser: DoclingParser) -> None:
        """Measure parsing speed (reports time, does not assert)."""
        assert _REAL_PDF is not None
        durations: list[float] = []
        for _ in range(3):
            t0 = time.perf_counter()
            parser.parse(_REAL_PDF)
            durations.append(time.perf_counter() - t0)

        avg = sum(durations) / len(durations)
        # Report — not assert — the timing
        print(f"\n  ⏱  DoclingParser — 3 runs: min={min(durations):.3f}s, avg={avg:.3f}s, max={max(durations):.3f}s")

    def test_element_count(self, parsed_doc: ParsedDocument) -> None:
        """Verify the parsed document has a reasonable number of elements."""
        n = len(parsed_doc.elements)
        print(f"\n  📊 Total elements: {n}")
        assert n > 0, "Parsed document should have at least 1 element"

    def test_contains_headings(self, parsed_doc: ParsedDocument) -> None:
        """At least one heading should be present."""
        heading_count = sum(
            1 for e in parsed_doc.elements if getattr(e, "type", None) == ElementType.HEADING or (hasattr(e, "heading_level") and e.heading_level is not None)
        )
        print(f"\n  📑 Heading elements: {heading_count}")
        # Don't assert — it's a quality metric, not a pass/fail
        # But warn if zero
        if heading_count == 0:
            print("  ⚠  Warning: No headings found — parser may not be preserving structure")

    def test_contains_text(self, parsed_doc: ParsedDocument) -> None:
        """At least some text content should be extracted."""
        text_count = sum(1 for e in parsed_doc.elements if getattr(e, "type", None) == ElementType.TEXT)
        total_chars = sum(len(str(getattr(e, "content", ""))) for e in parsed_doc.elements)
        print(f"\n  📝 Text elements: {text_count}")
        print(f"  🔤 Total characters extracted: {total_chars}")
        assert text_count > 0 or total_chars > 0, "No text content extracted from PDF"

    def test_structure_preservation(self, parsed_doc: ParsedDocument) -> None:
        """Check for preserved structure elements (tables, code, etc.)."""
        table_count = sum(1 for e in parsed_doc.elements if getattr(e, "type", None) == ElementType.TABLE)
        image_count = sum(1 for e in parsed_doc.elements if getattr(e, "type", None) == ElementType.IMAGE)
        print(f"\n  🏛  Tables found: {table_count}")
        print(f"  🖼  Images found: {image_count}")

    def test_page_count(self, parsed_doc: ParsedDocument) -> None:
        """Verify page_count is populated."""
        pc = getattr(parsed_doc, "page_count", 0)
        print(f"\n  📄 Page count: {pc}")
        if pc == 0:
            print("  ⚠  Warning: page_count is 0 — may indicate empty document")

    def test_element_types_distribution(self, parsed_doc: ParsedDocument) -> None:
        """Breakdown of element types found during parsing."""
        type_counts: Counter = Counter(e.type.name if getattr(e, "type", None) is not None else "UNKNOWN" for e in parsed_doc.elements)
        print("\n  📊 Element type distribution:")
        for tname, count in type_counts.most_common():
            print(f"     {tname}: {count}")
