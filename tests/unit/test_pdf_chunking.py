"""Unit tests for PDF page-range chunking in DoclingParser."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pypdfium2 as pdfium
import pytest

from knox.core.config import PdfConfig
from knox.preprocessing.parsers.docling_parser import DoclingParser


class TestCountPdfPages:
    """Tests for DoclingParser._count_pdf_pages."""

    def test_count_single_page(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "one.pdf"
        _create_pdf(pdf_path, 1)
        assert DoclingParser._count_pdf_pages(pdf_path) == 1

    def test_count_multiple_pages(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "three.pdf"
        _create_pdf(pdf_path, 3)
        assert DoclingParser._count_pdf_pages(pdf_path) == 3


class TestPdfConfigDefaults:
    """Tests for PdfConfig dataclass defaults."""

    def test_defaults(self) -> None:
        cfg = PdfConfig()
        assert cfg.max_pages_before_chunking == 50
        assert cfg.page_chunk_size == 30
        assert cfg.page_chunk_overlap == 5

    def test_custom_values(self) -> None:
        cfg = PdfConfig(max_pages_before_chunking=10, page_chunk_size=5, page_chunk_overlap=1)
        assert cfg.max_pages_before_chunking == 10
        assert cfg.page_chunk_size == 5
        assert cfg.page_chunk_overlap == 1

    def test_chunking_disabled(self) -> None:
        cfg = PdfConfig(max_pages_before_chunking=0)
        parser = DoclingParser(pdf_config=cfg)
        assert parser._pdf_config.max_pages_before_chunking == 0


class TestParseChunked:
    """Tests for the chunked parsing logic."""

    def test_small_pdf_no_chunking(self, tmp_path: Path) -> None:
        """A PDF below the threshold should not trigger chunked parsing."""
        pdf_path = tmp_path / "small.pdf"
        _create_pdf(pdf_path, 3)

        parser = DoclingParser(pdf_config=PdfConfig(max_pages_before_chunking=50))

        fake_result = self._make_fake_result(1, 3)
        with patch.object(parser, "_get_converter") as mock_conv, patch.object(DoclingParser, "_unload_result") as mock_unload:
            mock_conv.return_value.convert = MagicMock(return_value=fake_result)
            doc = parser.parse(pdf_path)
            mock_conv.return_value.convert.assert_called_once_with(str(pdf_path))
            mock_unload.assert_called_once()

        assert doc.metadata.get("chunked") is None
        assert doc.page_count == 3

    def test_chunked_parsing_dispatch(self, tmp_path: Path) -> None:
        """A PDF above the threshold should dispatch to _parse_chunked."""
        pdf_path = tmp_path / "large.pdf"
        _create_pdf(pdf_path, 5)

        parser = DoclingParser(
            pdf_config=PdfConfig(max_pages_before_chunking=2, page_chunk_size=2),
        )

        call_log: list[tuple[int, int]] = []

        def fake_convert(source: str, page_range: tuple[int, int] | None = None) -> MagicMock:
            if page_range is not None:
                call_log.append(page_range)
                return self._make_fake_result(page_range[0], page_range[1])
            call_log.append((0, 0))
            return self._make_fake_result(1, 5)

        with patch.object(parser, "_get_converter") as mock_conv, patch.object(DoclingParser, "_unload_result"):
            mock_conv.return_value.convert = fake_convert
            doc = parser.parse(pdf_path)

        assert call_log == [(1, 2), (3, 4), (5, 5)]
        assert doc.metadata.get("chunked") is True
        assert doc.page_count == 5
        assert len(doc.elements) == 5

    def test_chunked_reading_order_monotonic(self, tmp_path: Path) -> None:
        """Reading order should be monotonically increasing across chunks."""
        pdf_path = tmp_path / "large.pdf"
        _create_pdf(pdf_path, 6)

        parser = DoclingParser(
            pdf_config=PdfConfig(max_pages_before_chunking=2, page_chunk_size=2),
        )

        def fake_convert(source: str, page_range: tuple[int, int] | None = None) -> MagicMock:
            if page_range is not None:
                return self._make_fake_result(page_range[0], page_range[1])
            return self._make_fake_result(1, 6)

        with patch.object(parser, "_get_converter") as mock_conv, patch.object(DoclingParser, "_unload_result"):
            mock_conv.return_value.convert = fake_convert
            doc = parser.parse(pdf_path)

        orders = [e.reading_order for e in doc.elements]
        assert orders == sorted(orders)
        assert len(orders) == 6

    def test_chunking_disabled_via_config(self, tmp_path: Path) -> None:
        """When max_pages_before_chunking=0, chunking never triggers."""
        pdf_path = tmp_path / "big.pdf"
        _create_pdf(pdf_path, 100)

        parser = DoclingParser(pdf_config=PdfConfig(max_pages_before_chunking=0))

        fake_result = self._make_fake_result(1, 100)
        with patch.object(parser, "_get_converter") as mock_conv, patch.object(DoclingParser, "_unload_result"):
            mock_conv.return_value.convert = MagicMock(return_value=fake_result)
            doc = parser.parse(pdf_path)

        assert mock_conv.return_value.convert.call_count == 1
        assert doc.metadata.get("chunked") is None

    @staticmethod
    def _make_fake_doc(page_start: int, page_end: int) -> MagicMock:
        """Create a fake Docling document that yields items for given pages."""
        TextItem = type("TextItem", (), {"text": "", "prov": []})
        items: list[tuple[TextItem, int]] = []
        for page_no in range(page_start, page_end + 1):
            item = TextItem()
            object.__setattr__(item, "text", f"Content from page {page_no}")
            object.__setattr__(item, "prov", [SimpleNamespace(page_no=page_no)])
            items.append((item, 0))

        doc = MagicMock()
        doc.name = "test_doc"
        doc.pages = list(range(page_start, page_end + 1))
        doc.iterate_items = MagicMock(return_value=iter(items))
        return doc

    @staticmethod
    def _make_fake_result(page_start: int, page_end: int) -> MagicMock:
        """Create a fake Docling ConversionResult."""
        result = MagicMock()
        result.document = TestParseChunked._make_fake_doc(page_start, page_end)
        result.input = MagicMock()
        result.input._backend = MagicMock()
        return result


class TestUnloadResult:
    """Tests for DoclingParser._unload_result."""

    def test_unload_calls_backend_unload(self) -> None:
        backend = MagicMock()
        result = MagicMock()
        result.input._backend = backend

        DoclingParser._unload_result(result)
        backend.unload.assert_called_once()

    def test_unload_no_backend(self) -> None:
        result = MagicMock()
        result.input._backend = None

        DoclingParser._unload_result(result)

    def test_unload_swallows_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        backend = MagicMock()
        backend.unload.side_effect = RuntimeError("boom")
        result = MagicMock()
        result.input._backend = backend

        with caplog.at_level(logging.DEBUG):
            DoclingParser._unload_result(result)


def _create_pdf(path: Path, num_pages: int) -> None:
    """Create a minimal PDF with the given number of blank pages."""
    pdf = pdfium.PdfDocument.new()
    for _ in range(num_pages):
        pdf.new_page(500, 800)
    pdf.save(str(path))
