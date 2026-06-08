"""Tests for KNOX parser extensions (Milestone 9).

Covers:
- OpenDataLoaderParser: instantiation, ParserRouter registration
- VLMParser: image metadata extraction, ParserRouter registration
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from knox.preprocessing.parsers.base import FileExtension, ParserRouter
from knox.preprocessing.parsers.opendataloader_parser import OpenDataLoaderParser
from knox.preprocessing.parsers.vlm_parser import VLMParser

# ============================================================================
# OpenDataLoaderParser
# ============================================================================


class TestOpenDataLoaderParser:
    """Verify the opendataloader-pdf adapter."""

    def test_supported_extensions(self) -> None:
        """Only .pdf is supported."""
        assert OpenDataLoaderParser.supported_extensions == {FileExtension.PDF}

    def test_raises_notimplemented(self) -> None:
        """Instantiating OpenDataLoaderParser raises NotImplementedError (stub)."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            OpenDataLoaderParser()


# ============================================================================
# VLMParser
# ============================================================================


class TestVLMParser:
    """Verify the VLM image parser."""

    def test_instantiation(self) -> None:
        """VLMParser can be instantiated."""
        parser = VLMParser()
        assert parser.supported_extensions == {FileExtension.PNG, FileExtension.JPG, FileExtension.JPEG, FileExtension.GIF, FileExtension.WEBP}

    def test_parse_non_existent_image_returns_parsed_document(self) -> None:
        """Parsing a non-existent file returns a ParsedDocument with error info."""
        parser = VLMParser()
        fake_path = Path("/tmp/nonexistent_image_xyz.png")
        doc = parser.parse(fake_path)

        assert doc.source_path == fake_path
        assert doc.page_count == 1
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

        # Should have 2 elements: metadata + text
        assert len(doc.elements) == 2

        # First element is metadata
        meta_elem = doc.elements[0]
        assert meta_elem.type.name == "METADATA"
        assert isinstance(meta_elem.content, str)
        assert "test_image.png" in meta_elem.content
        assert "PNG" in meta_elem.content

        # Second element is the text (placeholder caption from _generate_caption)
        text_elem = doc.elements[1]
        assert text_elem.type.name == "TEXT"
        assert isinstance(text_elem.content, str)

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

    def test_router_opendataloader_not_implemented(self) -> None:
        """OpenDataLoaderParser raises NotImplementedError on instantiation."""
        with pytest.raises(NotImplementedError):
            OpenDataLoaderParser()
        assert OpenDataLoaderParser.supported_extensions == {FileExtension.PDF}
