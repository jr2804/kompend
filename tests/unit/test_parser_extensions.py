"""Tests for KNOX parser extensions (Milestone 9).

Covers:
- OpenDataLoaderParser: import guard, ImportError, ParserRouter registration
- VLMParser: image metadata extraction, graceful fallback, ParserRouter registration
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from knox.preprocessing.parsers.base import ParserRouter
from knox.preprocessing.parsers.opendataloader_parser import (
    OPD_AVAILABLE,
    OpenDataLoaderParser,
)
from knox.preprocessing.parsers.vlm_parser import VLM_AVAILABLE, VLMParser


# ============================================================================
# OpenDataLoaderParser
# ============================================================================


class TestOpenDataLoaderParser:
    """Verify the opendataloader-pdf adapter."""

    def test_import_guard_not_available(self) -> None:
        """When opendataloader is not installed, OPD_AVAILABLE is False."""
        assert OPD_AVAILABLE is False, (
            "opendataloader-pdf should not be importable in a standard test env"
        )

    def test_raises_importerror_when_unavailable(self) -> None:
        """Instantiating OpenDataLoaderParser without opendataloader raises."""
        with pytest.raises(ImportError) as excinfo:
            OpenDataLoaderParser()
        assert "opendataloader-pdf" in str(excinfo.value)
        assert "uv pip install" in str(excinfo.value)

    def test_supported_extensions(self) -> None:
        """Only .pdf is supported."""
        assert OpenDataLoaderParser.supported_extensions == (".pdf",)

    def test_can_parse_pdf_with_mock(self) -> None:
        """can_parse returns True for .pdf files when parser is instantiable."""
        with patch.object(
            OpenDataLoaderParser,
            "__init__",
            lambda self: setattr(self, "_loader", None),
        ):
            parser = object.__new__(OpenDataLoaderParser)
            parser.__init__()
            assert parser.can_parse(Path("report.pdf")) is True

    def test_cannot_parse_other_formats_with_mock(self) -> None:
        """can_parse returns False for non-PDF files."""
        with patch.object(
            OpenDataLoaderParser,
            "__init__",
            lambda self: setattr(self, "_loader", None),
        ):
            parser = object.__new__(OpenDataLoaderParser)
            parser.__init__()
            assert parser.can_parse(Path("notes.txt")) is False
            assert parser.can_parse(Path("image.png")) is False
            assert parser.can_parse(Path("doc.docx")) is False


# ============================================================================
# VLMParser
# ============================================================================


class TestVLMParser:
    """Verify the VLM image parser."""

    def test_vlm_not_available(self) -> None:
        """VLM_AVAILABLE should be False since docling_slim is not installed."""
        assert VLM_AVAILABLE is False, (
            "docling_slim.models.VLMInlineModel is not currently available"
        )

    def test_instantiation_without_vlm(self) -> None:
        """VLMParser can be instantiated even without VLM available."""
        parser = VLMParser()
        assert parser.supported_extensions == (".png", ".jpg", ".jpeg", ".gif", ".webp")

    def test_parse_non_existent_image_returns_parsed_document(self) -> None:
        """Parsing a non-existent file returns a ParsedDocument with error info."""
        parser = VLMParser()
        fake_path = Path("/tmp/nonexistent_image_xyz.png")
        doc = parser.parse(fake_path)

        assert doc.source_path == fake_path
        assert doc.page_count == 1
        assert len(doc.elements) == 2  # metadata + text fallback
        assert doc.metadata["parser"] == "vlm"

    def test_parse_real_image_with_pil(self, tmp_path: Path) -> None:
        """Parsing a real image file extracts metadata correctly."""
        png_path = tmp_path / "test_image.png"
        img = Image.new("RGB", (4, 4), color="red")
        img.save(png_path, format="PNG")

        parser = VLMParser()
        doc = parser.parse(png_path)

        assert doc.source_path == png_path
        assert doc.title == "test_image"
        assert doc.page_count == 1
        assert doc.metadata["parser"] == "vlm"
        assert doc.metadata["format"] == "PNG"
        assert doc.metadata["width"] == 4
        assert doc.metadata["height"] == 4

        # Should have 2 elements: metadata + text fallback
        assert len(doc.elements) == 2

        # First element is metadata
        meta_elem = doc.elements[0]
        assert meta_elem.type.name == "METADATA"
        assert isinstance(meta_elem.content, str)
        assert "test_image.png" in meta_elem.content
        assert "PNG" in meta_elem.content

        # Second element is the text fallback (VLM not available)
        text_elem = doc.elements[1]
        assert text_elem.type.name == "TEXT"
        assert isinstance(text_elem.content, str)
        assert "not available" in text_elem.content

    def test_parse_jpg_image(self, tmp_path: Path) -> None:
        """Parsing a JPEG file also works."""
        jpg_path = tmp_path / "photo.jpg"
        img = Image.new("RGB", (8, 8), color="blue")
        img.save(jpg_path, format="JPEG")

        parser = VLMParser()
        doc = parser.parse(jpg_path)

        assert doc.source_path == jpg_path
        assert doc.title == "photo"
        assert doc.metadata["format"] in ("JPEG", "jpeg")
        assert len(doc.elements) == 2

    def test_can_parse(self) -> None:
        """can_parse returns True for supported image extensions."""
        parser = VLMParser()
        assert parser.can_parse(Path("photo.png")) is True
        assert parser.can_parse(Path("photo.jpg")) is True
        assert parser.can_parse(Path("photo.jpeg")) is True
        assert parser.can_parse(Path("photo.gif")) is True
        assert parser.can_parse(Path("photo.webp")) is True

    def test_cannot_parse_non_image(self) -> None:
        """can_parse returns False for unsupported extensions."""
        parser = VLMParser()
        assert parser.can_parse(Path("document.pdf")) is False
        assert parser.can_parse(Path("notes.txt")) is False
        assert parser.can_parse(Path("data.csv")) is False


# ============================================================================
# ParserRouter registration
# ============================================================================


class TestParserRouterRegistration:
    """Verify both new parsers register correctly in ParserRouter."""

    def test_router_registers_vlm(self) -> None:
        """VLMParser registers for image extensions and routes correctly."""
        router = ParserRouter()
        parser = VLMParser()
        router.register(parser)

        assert router.route(Path("photo.png")) is parser
        assert router.route(Path("photo.jpg")) is parser
        assert router.route(Path("photo.jpeg")) is parser
        assert router.route(Path("photo.gif")) is parser
        assert router.route(Path("photo.webp")) is parser
        assert router.route(Path("document.pdf")) is None

    def test_router_parse_vlm(self) -> None:
        """Router.parse() works with VLMParser for an image."""
        router = ParserRouter()
        parser = VLMParser()
        router.register(parser)

        doc = router.parse(Path("/tmp/vlm_router_test.png"))
        assert doc.metadata["parser"] == "vlm"
        assert doc.page_count == 1

    def test_router_parse_fails_for_unregistered_type(self) -> None:
        """Router.parse() raises ValueError for unregistered file types."""
        router = ParserRouter()
        with pytest.raises(ValueError, match="No parser found"):
            router.parse(Path("unknown.jfif"))

    def test_router_opendataloader_cannot_instantiate(self) -> None:
        """OpenDataLoaderParser cannot be registered due to ImportError."""
        with pytest.raises(ImportError):
            OpenDataLoaderParser()
        # Verify the class-level routing info is correct even without instantiation
        assert OpenDataLoaderParser.supported_extensions == (".pdf",)