"""Integration tests using the real test corpus (tests/data/sources/ and urls.txt).

These tests require:
  - PDF files in tests/data/sources/
  - URLs in tests/data/urls.txt
  - DoclingParser (or fallback parser) available

They are skipped if the required resources are not present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from knox.chunking.hybrid import HybridChunker
from knox.compilation.graph import GraphCompiler
from knox.core.models import ParsedDocument, ResolvedChunk
from knox.preprocessing.downloaders.http_downloader import HTTPDownloader
from knox.preprocessing.parsers.docling_parser import DoclingParser
from knox.preprocessing.parsers.pdfplumber_parser import PdfPlumberParser


def _check_parser() -> bool:
    """Return True if a parser can be instantiated."""
    try:
        DoclingParser(extract_images=False)
        return True
    except Exception:
        pass
    try:
        PdfPlumberParser()
        return True
    except Exception:
        pass
    return False


def _get_parser():
    """Return the best available parser instance."""
    try:
        return DoclingParser(extract_images=False)
    except Exception:
        pass
    try:
        return PdfPlumberParser()
    except Exception:
        pass
    pytest.skip("No PDF parser available (DoclingParser or PdfPlumberParser)")


_PARSER_OK = _check_parser()

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Parse local PDFs from tests/data/sources/
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _PARSER_OK,
    reason="No PDF parser available (DoclingParser or PDFPlumberParser)",
)
class TestRealCorpusParse:
    """Parse real PDFs from the test corpus and validate structure."""

    @pytest.fixture(scope="class")
    def parser(self):
        return _get_parser()

    def test_parse_first_pdf(self, source_pdfs: list[Path], parser) -> None:
        """Parse the first PDF in the corpus — basic sanity."""
        assert len(source_pdfs) > 0, "No PDFs in tests/data/sources/"
        pdf = source_pdfs[0]
        doc = parser.parse(pdf)
        assert isinstance(doc, ParsedDocument)
        assert len(doc.elements) > 0, f"No elements parsed from {pdf.name}"
        assert doc.page_count > 0, f"page_count is 0 for {pdf.name}"

    def test_parse_all_pdfs_succeed(self, source_pdfs: list[Path], parser) -> None:
        """Every PDF in the corpus must parse without raising."""
        for pdf in source_pdfs:
            doc = parser.parse(pdf)
            assert isinstance(doc, ParsedDocument)
            assert len(doc.elements) > 0, f"No elements from {pdf.name}"

    def test_all_pdfs_have_text(self, source_pdfs: list[Path], parser) -> None:
        """Every PDF must produce at least some text content."""
        for pdf in source_pdfs:
            doc = parser.parse(pdf)
            text_chars = sum(len(e.content) for e in doc.elements if e.content)
            assert text_chars > 50, f"{pdf.name}: only {text_chars} chars extracted"


# ---------------------------------------------------------------------------
# Chunk + compile a real PDF into a wiki
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _PARSER_OK,
    reason="No PDF parser available",
)
class TestRealCorpusWiki:
    """Build a compiled wiki from real PDFs."""

    def test_compile_single_pdf_to_wiki(self, source_pdfs: list[Path]) -> None:
        """Parse → chunk → resolve → compile a single PDF into a CompiledWiki."""
        assert len(source_pdfs) > 0
        parser = _get_parser()

        doc = parser.parse(source_pdfs[0])
        chunker = HybridChunker(max_tokens=512)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 0

        # Wrap raw chunks into ResolvedChunks (skip resolution tier for this test)
        resolved = [ResolvedChunk(chunk=c) for c in chunks]

        compiler = GraphCompiler()
        wiki = compiler.compile(resolved)

        assert len(wiki.nodes) > 0
        assert len(wiki.toc) > 0
        assert wiki.quality_score >= 0.0

    def test_compile_all_pdfs_to_wiki(self, source_pdfs: list[Path]) -> None:
        """Parse and compile all corpus PDFs — verify no crashes."""
        parser = _get_parser()
        chunker = HybridChunker(max_tokens=512)
        compiler = GraphCompiler()

        for pdf in source_pdfs:
            doc = parser.parse(pdf)
            chunks = chunker.chunk(doc)
            resolved = [ResolvedChunk(chunk=c) for c in chunks]
            wiki = compiler.compile(resolved)
            assert len(wiki.nodes) > 0, f"No nodes from {pdf.name}"


# ---------------------------------------------------------------------------
# URL download integration
# ---------------------------------------------------------------------------


class TestURLDownload:
    """Download URLs from tests/data/urls.txt."""

    def test_url_file_exists_and_has_urls(self, test_urls: list[str]) -> None:
        """urls.txt must exist and contain at least one URL."""
        assert len(test_urls) > 0, "tests/data/urls.txt is empty or missing"

    @pytest.mark.skipif(
        not _PARSER_OK,
        reason="No PDF parser available",
    )
    def test_download_first_url(self, test_urls: list[str], tmp_path: Path) -> None:
        """Download the first URL and verify it produces a parseable file."""
        pytest.importorskip("knox.preprocessing.downloaders.http_downloader")

        assert len(test_urls) > 0
        downloader = HTTPDownloader(timeout=30)
        path = downloader.download(test_urls[0], tmp_path)

        if path is None:
            pytest.skip(f"Could not download {test_urls[0]} (network or access issue)")

        assert path.exists()
        assert path.stat().st_size > 0

        # Try to parse the downloaded file
        parser = _get_parser()
        doc = parser.parse(path)
        assert isinstance(doc, ParsedDocument)
        assert len(doc.elements) > 0
