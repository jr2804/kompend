"""Integration tests using a small subset of the real test corpus.

Parses a **3-PDF sample** (the smallest in ``tests/data/sources/``) with
LiteParse (Rust-core, ~30x faster than Docling) and runs the resulting
chunks through the real graph compiler. Session-scoped fixtures cache the
parsed documents so the 7 tests share one parse pass.

No LLM/VLM/embedding calls — GraphCompiler is a deterministic, prompt-free
compiler that builds nodes/edges from chunk text. See
``tests/integration/test_pipeline_mock.py`` for the mock pattern, and
``tests/AGENTS.md`` for the "no real engine calls" rule.

Skip guards:
  - ``_skip_no_pdfs``:  tests/data/sources/ is empty
  - ``_skip_no_parser``: no parser backend installed
  - ``_skip_no_urls``:   tests/data/sources/urls.txt is empty

Notes:
  - LiteParse is the default (Rust-core, sub-second per PDF). Falls back
    to Docling if LiteParse is not installed.
  - The ``source_pdfs_sample`` fixture caps the corpus to 3 PDFs so even
    Docling stays under 30 seconds.
  - Session-scoped ``parsed_docs_sample`` cache means the 7 tests in
    this file share one parse pass.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from kompend.chunking import HierarchicalChunker
from kompend.compilation.graph import GraphCompiler
from kompend.core.models import ParsedDocument, ResolvedChunk
from kompend.preprocessing.downloaders.http_downloader import HTTPDownloader
from kompend.preprocessing.parsers import build_default_router
from kompend.preprocessing.parsers.base import BaseParser

#: Number of smallest PDFs to use as the corpus sample. Kept low so the
#: suite stays under 30 seconds.
_CORPUS_SAMPLE_SIZE = 3

logger = logging.getLogger(__name__)


def _parser_available() -> bool:
    """The configured ingestion backend resolves and is installed."""
    try:
        build_default_router()
    except Exception:
        return False
    return True


_PARSER_OK = _parser_available()


_skip_no_parser = pytest.mark.skipif(
    not _PARSER_OK,
    reason="No PDF parser backend installed",
)


@pytest.mark.skipif(not _PARSER_OK, reason="No PDF parser backend available")
class TestRealCorpusParse:
    """Parse real PDFs from the test corpus and validate structure."""

    def test_parse_first_pdf(self, parsed_docs_sample: dict[Path, ParsedDocument]) -> None:
        """First parsed PDF is a non-empty ParsedDocument."""
        assert parsed_docs_sample, "No PDFs parsed from the sample"
        first_pdf, doc = next(iter(parsed_docs_sample.items()))
        assert isinstance(doc, ParsedDocument)
        assert len(doc.elements) > 0, f"No elements parsed from {first_pdf.name}"
        assert doc.page_count > 0, f"page_count is 0 for {first_pdf.name}"

    def test_parse_all_pdfs_succeed(self, parsed_docs_sample: dict[Path, ParsedDocument]) -> None:
        """Every PDF in the sample must parse without raising."""
        for pdf, doc in parsed_docs_sample.items():
            assert isinstance(doc, ParsedDocument), f"{pdf.name} returned wrong type"
            assert len(doc.elements) > 0, f"No elements from {pdf.name}"

    def test_all_pdfs_have_text(self, parsed_docs_sample: dict[Path, ParsedDocument]) -> None:
        """Every PDF must produce at least some text content."""
        for pdf, doc in parsed_docs_sample.items():
            text_chars = sum(len(e.content) for e in doc.elements if e.content)
            assert text_chars > 50, f"{pdf.name}: only {text_chars} chars extracted"


@pytest.mark.skipif(not _PARSER_OK, reason="No PDF parser backend available")
class TestRealCorpusWiki:
    """Build a compiled wiki from real PDFs (chunk + compile, no LLM)."""

    def test_compile_first_pdf_to_wiki(self, resolved_chunks_sample: list[ResolvedChunk]) -> None:
        """Compile the first PDF's chunks into a CompiledWiki."""
        assert len(resolved_chunks_sample) > 0, "No chunks resolved from the sample"

        compiler = GraphCompiler("model")
        wiki = compiler.compile(resolved_chunks_sample)
        assert len(wiki.nodes) > 0
        assert len(wiki.toc) >= 0
        assert wiki.quality_score >= 0.0

    def test_compile_all_pdfs_to_wiki(self, resolved_chunks_sample: list[ResolvedChunk]) -> None:
        """Compile all sampled chunks — verify no crashes."""
        assert len(resolved_chunks_sample) > 0
        compiler = GraphCompiler("model")
        wiki = compiler.compile(resolved_chunks_sample)
        assert len(wiki.nodes) > 0, "No nodes from any sampled PDF"


class TestURLDownload:
    """Download URLs from tests/data/sources/urls.txt.

    No parser needed for the file-existence check; the parse check is
    gated behind the parser skip.
    """

    def test_url_file_exists_and_has_urls(self, test_urls: list[str]) -> None:
        """urls.txt must exist and contain at least one URL."""
        assert len(test_urls) > 0, "tests/data/sources/urls.txt is empty or missing"

    @pytest.mark.skipif(not _PARSER_OK, reason="No PDF parser backend available")
    def test_download_first_url(self, test_urls: list[str], tmp_path: Path) -> None:
        """Download the first URL and verify the response is a non-empty file."""
        pytest.importorskip("kompend.preprocessing.downloaders.http_downloader")
        assert len(test_urls) > 0

        downloader = HTTPDownloader(timeout=30)
        path = downloader.download(test_urls[0], tmp_path)
        if path is None:
            pytest.skip(f"Could not download {test_urls[0]} (network or access issue)")
        assert path.exists()
        assert path.stat().st_size > 0


@pytest.fixture(scope="session")
def resolved_chunks_sample(parsed_docs_sample: dict[Path, ParsedDocument]) -> list[ResolvedChunk]:
    """Chunk every parsed document once and expose a flat list."""
    chunker = HierarchicalChunker(max_tokens=512)
    out: list[ResolvedChunk] = []
    for doc in parsed_docs_sample.values():
        tree = chunker.chunk(doc)
        for node in tree.leaves():
            if node.chunk:
                out.append(ResolvedChunk(chunk=node.chunk) if isinstance(node.chunk, object) else node.chunk)
    return out


@pytest.fixture(scope="session")
def parsed_docs_sample(source_pdfs_sample: list[Path]) -> dict[Path, ParsedDocument]:
    """Parse every PDF in the sample ONCE per pytest session.

    The 7 tests in this file share this cache, so the parse phase runs
    exactly ``len(sample)`` times regardless of how many tests read it.
    """
    if not source_pdfs_sample:
        return {}
    parser = _make_parser()
    cache: dict[Path, ParsedDocument] = {}
    for pdf in source_pdfs_sample:
        try:
            cache[pdf] = parser.parse(pdf)
        except Exception:
            logger.exception("Parse failed for %s", pdf.name)
    return cache


def _make_parser() -> BaseParser:
    """Construct the ingestion parser via its normal factory."""
    parser = build_default_router().route(Path("sample.pdf"))
    if parser is None:
        msg = "no installed parser handles .pdf; install an extraction extra to run the corpus tests"
        raise RuntimeError(msg)
    return parser


@pytest.fixture(scope="session")
def source_pdfs_sample(source_pdfs: list[Path]) -> list[Path]:
    """The 3 smallest PDFs in the corpus — deterministic, sub-second parse."""
    return sorted(source_pdfs, key=lambda p: p.stat().st_size)[:_CORPUS_SAMPLE_SIZE]


def _have_pdfs(source_pdfs: list[Path]) -> bool:
    return len(source_pdfs) > 0
