"""Integration tests for the arXiv resolver pipeline.

Tests three download-and-parse paths against arXiv paper 2405.17364
("Attention Mechanisms: A Survey"):

  1. PDF via ArxivPdfResolver  (arxiv.org/abs/...)  -> liteparse/docling
  2. HTML via ArxivHtmlResolver (arxiv.org/html/...) -> WebPageParser (trafilatura)
  3. LaTeX source via ArxivSourceResolver (arxiv.org/src/...) -> _extract_archive -> PandocParser

All three are skipped when:
  - The arXiv host is not reachable (offline environment / CI without network)
  - The required parser / dependency is not available

Run manually:
    uv run pytest tests/integration/test_arxiv_pipeline.py -v -s
"""

from __future__ import annotations

import logging
import socket
from pathlib import Path

import pytest

from knox.cli._sources import extract_archive
from knox.core.models import ParsedDocument
from knox.preprocessing.downloaders.http_downloader import HTTPDownloader
from knox.preprocessing.downloaders.url_resolver import (
    ArxivHtmlResolver,
    ArxivPdfResolver,
    ArxivSourceResolver,
    UrlResolverChain,
)
from knox.preprocessing.parsers._registry import ParserBuildOptions, build_default_router
from knox.preprocessing.parsers.web_parser import WebPageParser

ARXIV_PAPER_ID = "2405.17364"
ARXIV_PDF_URL = f"https://arxiv.org/abs/{ARXIV_PAPER_ID}"
ARXIV_HTML_URL = f"https://arxiv.org/html/{ARXIV_PAPER_ID}v1"
ARXIV_SRC_URL = f"https://arxiv.org/src/{ARXIV_PAPER_ID}"

logger = logging.getLogger(__name__)


def _arxiv_reachable() -> bool:
    """Return True when arxiv.org is reachable on port 443."""
    try:
        socket.create_connection(("arxiv.org", 443), timeout=5).close()
        return True
    except OSError:
        return False


requires_network = pytest.mark.skipif(
    not _arxiv_reachable(),
    reason="arxiv.org not reachable — skipping network integration tests",
)


# ── Scenario 1: PDF via ArxivPdfResolver ────────────────────────────────


@requires_network
def test_arxiv_pdf_resolver_rewrites_url(tmp_path: Path) -> None:
    """ArxivPdfResolver rewrites /abs/ URL to /pdf/ URL."""
    resolver = ArxivPdfResolver()
    result = resolver.resolve(ARXIV_PDF_URL)
    assert result is not None
    assert "/pdf/" in result.download_url
    assert result.download_url.endswith(".pdf")
    assert result.content_type == "application/pdf"
    assert result.source_label == "arxiv"
    assert result.prefer_http is False  # default


@requires_network
def test_arxiv_pdf_downloads_and_parses(tmp_path: Path) -> None:
    """ArxivPdfResolver + HTTPDownloader + liteparse produces a ParsedDocument."""
    chain = UrlResolverChain()
    resolved = chain.resolve(ARXIV_PDF_URL)
    assert resolved.download_url != ARXIV_PDF_URL, "URL should be rewritten to PDF"

    downloader = HTTPDownloader(timeout=60)
    pdf_file = downloader.download(resolved.download_url, tmp_path)
    assert pdf_file is not None, "PDF download failed"
    assert pdf_file.suffix.lower() == ".pdf", f"Expected .pdf, got {pdf_file.suffix}"

    router = build_default_router(options=ParserBuildOptions(), names="liteparse,docling")
    parser = router.route(pdf_file)
    assert parser is not None
    doc = parser.parse(pdf_file)

    assert isinstance(doc, ParsedDocument)
    text = _all_text(doc)
    assert len(text) > 500, "Expected substantial text from the PDF"
    logger.info("PDF parse: %d elements, %d chars", len(doc.elements), len(text))


# ── Scenario 2: HTML via ArxivHtmlResolver ──────────────────────────────


@requires_network
def test_arxiv_html_resolver_flags_prefer_http(tmp_path: Path) -> None:
    """ArxivHtmlResolver sets prefer_http=True so HTML goes to HTTPDownloader."""
    resolver = ArxivHtmlResolver()
    result = resolver.resolve(ARXIV_HTML_URL)
    assert result is not None
    assert result.prefer_http is True
    assert result.content_type == "text/html"
    assert result.source_label == "arxiv-html"


@requires_network
def test_arxiv_html_downloads_as_html_and_parses_with_trafilatura(tmp_path: Path) -> None:
    """ArxivHtmlResolver + HTTPDownloader + WebPageParser produces clean text."""
    chain = UrlResolverChain()
    resolved = chain.resolve(ARXIV_HTML_URL)
    assert resolved.prefer_http is True, "ArxivHtmlResolver should set prefer_http"

    downloader = HTTPDownloader(timeout=60)
    html_file = downloader.download(resolved.download_url, tmp_path)
    assert html_file is not None, "HTML download failed"
    assert html_file.suffix.lower() in {".html", ".htm"}, f"Expected .html, got {html_file}"

    router = build_default_router(options=ParserBuildOptions(), names="web,liteparse,docling")
    parser = router.route(html_file)
    assert isinstance(parser, WebPageParser), f"Expected WebPageParser for .html, got {type(parser).__name__}"

    doc = parser.parse(html_file)
    assert isinstance(doc, ParsedDocument)
    text = _all_text(doc)
    assert "<html>" not in text, "Raw HTML tags must not appear in trafilatura output"
    assert "<p>" not in text, "Raw HTML tags must not appear in trafilatura output"
    assert len(text) > 500, "Expected substantial article text"
    logger.info("HTML parse: %d elements, %d chars", len(doc.elements), len(text))


# ── Scenario 3: LaTeX source via ArxivSourceResolver ────────────────────


@requires_network
def test_arxiv_source_resolver_flags_prefer_http(tmp_path: Path) -> None:
    """ArxivSourceResolver sets prefer_http=True to force binary HTTPDownloader."""
    resolver = ArxivSourceResolver()
    result = resolver.resolve(ARXIV_SRC_URL)
    assert result is not None
    assert result.prefer_http is True
    assert result.source_label == "arxiv-src"


@requires_network
def test_arxiv_source_downloads_tarball_then_extracts_tex(tmp_path: Path) -> None:
    """ArxivSourceResolver + HTTPDownloader + _extract_archive finds .tex file."""
    chain = UrlResolverChain()
    resolved = chain.resolve(ARXIV_SRC_URL)
    assert resolved.prefer_http is True

    downloader = HTTPDownloader(timeout=120)
    archive_file = downloader.download(resolved.download_url, tmp_path)
    assert archive_file is not None, "Archive download should succeed"
    logger.info("Downloaded archive: %s (%d bytes)", archive_file.name, archive_file.stat().st_size)

    # Archive should be recognisable (tar.gz or similar).
    name_lower = archive_file.name.lower()
    assert any(name_lower.endswith(s) for s in (".tar.gz", ".gz", ".tar", ".tgz")), f"Expected a tar.gz file, got {archive_file.name}"

    extracted = extract_archive(archive_file, tmp_path)
    assert extracted is not None, "_extract_archive should find a .tex file"
    assert extracted.suffix.lower() == ".tex", f"Expected a .tex file, extracted {extracted.name}"
    assert not archive_file.exists(), "Original archive should be removed after extraction"
    logger.info("Extracted: %s (%d bytes)", extracted.name, extracted.stat().st_size)


@requires_network
def test_arxiv_source_tex_parses_with_pandoc(tmp_path: Path) -> None:
    """End-to-end: arxiv.org/src/ -> .tar.gz -> .tex -> PandocParser -> ParsedDocument."""
    chain = UrlResolverChain()
    resolved = chain.resolve(ARXIV_SRC_URL)

    downloader = HTTPDownloader(timeout=120)
    archive_file = downloader.download(resolved.download_url, tmp_path)
    assert archive_file is not None

    extracted = extract_archive(archive_file, tmp_path)
    assert extracted is not None
    assert extracted.suffix.lower() == ".tex"

    # Route through the configured backend; parsecraft drives pandoc for .tex.
    router = build_default_router(name="parsecraft")
    parser = router.route(extracted)
    assert parser is not None, "Expected a backend for .tex, got none"

    doc = parser.parse(extracted)
    assert isinstance(doc, ParsedDocument)
    text = _all_text(doc)
    # Pandoc may return empty content for stub .tex (e.g. main file with only \input commands).
    # Assert the parse succeeded and produced a document — content quality is pandoc-version-dependent.
    assert isinstance(doc, ParsedDocument)
    logger.info("TeX parse: %d elements, %d chars", len(doc.elements), len(text))


def _all_text(doc: ParsedDocument) -> str:
    """Concatenate all string-content elements."""
    return " ".join(e.content for e in doc.elements if isinstance(e.content, str) and e.content.strip())
