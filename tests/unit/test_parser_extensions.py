"""Tests for KNOX parser extensions (Milestone 9).

Covers:
- OpenDataLoaderParser: lazy import, JSON mapping, ParserRouter registration
- VLMParser: image metadata extraction, VLM captioning, ParserRouter registration
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from knox.core.models import ElementType, ParsedDocument
from knox.preprocessing.parsers.base import FileExtension, ParserRouter
from knox.preprocessing.parsers.opendataloader_parser import (
    _OD_TYPE_MAP,
    OpenDataLoaderParser,
)
from knox.preprocessing.parsers.vlm_parser import VLMParser

# ============================================================================
# OpenDataLoaderParser
# ============================================================================


def _make_mock_opendataloader(tmp_json: dict) -> ModuleType:
    """Create a minimal mock ``opendataloader_pdf`` module."""
    mod = ModuleType("opendataloader_pdf")

    def _convert(*, input_path, output_dir, format) -> None:  # noqa: A002, ARG001
        Path(output_dir).joinpath("result.json").write_text(__import__("json").dumps(tmp_json), encoding="utf-8")

    mod.convert = _convert
    return mod


class TestOpenDataLoaderParser:
    """Verify the opendataloader-pdf adapter."""

    def test_supported_extensions(self) -> None:
        assert OpenDataLoaderParser.supported_extensions == {FileExtension.PDF}

    def test_import_error_without_package(self) -> None:
        with pytest.raises(ImportError, match="opendataloader-pdf"):
            # Force the import path to fail
            with patch.dict("sys.modules", {"opendataloader_pdf": None}):
                # Remove opendataloader_pdf from sys.modules if present
                import sys  # noqa: PLC0415

                sys.modules.pop("opendataloader_pdf", None)
                OpenDataLoaderParser()

    def test_parse_maps_json_to_document(self, tmp_path: Path) -> None:
        mock_result = {
            "elements": [
                {
                    "type": "heading",
                    "content": "Title",
                    "page number": 1,
                    "heading level": 1,
                    "bounding box": [72, 700, 540, 730],
                    "font": "Helvetica-Bold",
                },
                {
                    "type": "paragraph",
                    "content": "Hello world",
                    "page number": 1,
                },
                {
                    "type": "table",
                    "content": "| A | B |",
                    "page number": 2,
                },
                {
                    "type": "image",
                    "content": "",
                    "page number": 2,
                },
                {
                    "type": "formula",
                    "content": "E = mc^2",
                    "page number": 1,
                },
                {
                    "type": "caption",
                    "content": "Figure 1",
                    "page number": 1,
                },
            ]
        }
        mock_mod = _make_mock_opendataloader(mock_result)

        with patch.dict("sys.modules", {"opendataloader_pdf": mock_mod}):
            parser = OpenDataLoaderParser()
            pdf_path = tmp_path / "test.pdf"
            pdf_path.write_text("dummy", encoding="utf-8")
            doc = parser.parse(pdf_path)

        assert isinstance(doc, ParsedDocument)
        assert doc.title == "Title"
        assert doc.page_count == 2
        assert len(doc.elements) == 6
        assert doc.metadata["parser"] == "opendataloader"

        # Verify heading
        h = doc.elements[0]
        assert h.type == ElementType.HEADING
        assert h.content == "Title"
        assert h.metadata["heading_level"] == 1
        assert h.page == 1

        # Verify paragraph
        p = doc.elements[1]
        assert p.type == ElementType.TEXT
        assert p.content == "Hello world"

        # Verify table
        t = doc.elements[2]
        assert t.type == ElementType.TABLE
        assert "| A | B |" in t.content

        # Verify image
        img = doc.elements[3]
        assert img.type == ElementType.IMAGE
        assert img.metadata["od_type"] == "image"

        # Verify formula
        f = doc.elements[4]
        assert f.type == ElementType.FORMULA
        assert f.content == "E = mc^2"

        # Verify caption
        c = doc.elements[5]
        assert c.type == ElementType.TEXT
        assert c.content == "Figure 1"

    def test_parse_no_json_output_raises(self, tmp_path: Path) -> None:
        """Missing JSON output should raise RuntimeError."""
        mock_mod = ModuleType("opendataloader_pdf")
        mock_mod.convert = lambda *, input_path, output_dir, format: None  # noqa: A002

        with patch.dict("sys.modules", {"opendataloader_pdf": mock_mod}):
            parser = OpenDataLoaderParser()
            pdf_path = tmp_path / "empty.pdf"
            pdf_path.write_text("dummy", encoding="utf-8")
            with pytest.raises(RuntimeError, match="no JSON output"):
                parser.parse(pdf_path)


# ============================================================================
# VLMParser
# ============================================================================


class TestVLMParser:
    """Verify the VLM image parser."""

    def test_instantiation_no_client(self) -> None:
        parser = VLMParser()
        assert parser.supported_extensions == {
            FileExtension.PNG,
            FileExtension.JPG,
            FileExtension.JPEG,
            FileExtension.GIF,
            FileExtension.WEBP,
        }

    def test_instantiation_with_client(self) -> None:
        fake_client = MagicMock()
        parser = VLMParser(client=fake_client, model="ollama/glm-ocr")
        assert parser._client is fake_client
        assert parser._model == "ollama/glm-ocr"

    def test_parse_non_existent_image_returns_parsed_document(self) -> None:
        parser = VLMParser()
        fake_path = Path("/tmp/nonexistent_image_xyz.png")  # noqa: S108
        doc = parser.parse(fake_path)

        assert doc.source_path == fake_path
        assert doc.page_count == 1
        assert doc.metadata["parser"] == "vlm"

        # Should have metadata + placeholder text
        assert len(doc.elements) == 2
        assert doc.elements[0].type == ElementType.METADATA

    def test_parse_real_image_with_pil(self, tmp_path: Path) -> None:
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
        assert meta_elem.type == ElementType.METADATA
        assert isinstance(meta_elem.content, str)
        assert "test_image.png" in meta_elem.content
        assert "PNG" in meta_elem.content

        # Second element is the text (placeholder caption)
        text_elem = doc.elements[1]
        assert text_elem.type == ElementType.TEXT
        assert isinstance(text_elem.content, str)
        assert "vlm" in text_elem.content.lower() or "image" in text_elem.content.lower()

    def test_vlm_caption_with_client(self, tmp_path: Path) -> None:
        """When a client is provided, the parser calls try_complete."""
        fake_result = MagicMock()
        fake_result.content = "A red square."
        fake_client = MagicMock()
        fake_client.try_complete.return_value = fake_result

        png_path = tmp_path / "test_image.png"
        img = Image.new("RGB", (4, 4), color="red")
        img.save(png_path, format="PNG")

        parser = VLMParser(client=fake_client, model="ollama/glm-ocr")
        doc = parser.parse(png_path)

        assert doc.elements[1].content == "A red square."
        assert doc.elements[1].metadata["vlm_generated"] is True
        assert fake_client.try_complete.called

        # Verify multimodal message structure
        call_args = fake_client.try_complete.call_args
        assert call_args is not None
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert any(item.get("type") == "image_url" for item in messages[0]["content"])

    def test_vlm_caption_client_returns_none(self, tmp_path: Path) -> None:
        """When try_complete returns None, fallback to the placeholder."""
        fake_client = MagicMock()
        fake_client.try_complete.return_value = None

        png_path = tmp_path / "test_image.png"
        img = Image.new("RGB", (4, 4), color="red")
        img.save(png_path, format="PNG")

        parser = VLMParser(client=fake_client)
        doc = parser.parse(png_path)

        # Fallback: either the VLM-unavailable message or the fallback placeholder
        text = doc.elements[1].content
        assert isinstance(text, str)
        assert doc.elements[1].metadata["vlm_generated"] is False or "unavailable" in text.lower()

    def test_parse_jpg_image(self, tmp_path: Path) -> None:
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
        parser = VLMParser()
        assert parser.can_parse(Path("photo.png")) is True
        assert parser.can_parse(Path("photo.jpg")) is True
        assert parser.can_parse(Path("photo.jpeg")) is True
        assert parser.can_parse(Path("photo.gif")) is True
        assert parser.can_parse(Path("photo.webp")) is True

    def test_cannot_parse_non_image(self) -> None:
        parser = VLMParser()
        assert parser.can_parse(Path("document.pdf")) is False
        assert parser.can_parse(Path("notes.txt")) is False
        assert parser.can_parse(Path("data.csv")) is False

    def test_od_type_map_is_complete(self) -> None:
        """All mapped types resolve to non-TEXT ElementTypes."""
        assert ElementType.HEADING in _OD_TYPE_MAP.values()
        assert ElementType.TABLE in _OD_TYPE_MAP.values()
        assert ElementType.IMAGE in _OD_TYPE_MAP.values()
        assert ElementType.FORMULA in _OD_TYPE_MAP.values()


# ============================================================================
# ParserRouter registration
# ============================================================================


class TestParserRouterRegistration:
    """Verify both new parsers register correctly in ParserRouter."""

    def test_router_registers_vlm(self) -> None:
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
        router = ParserRouter()
        parser = VLMParser()
        router.register(parser)

        doc = router.parse(Path("/tmp/vlm_router_test.png"))  # noqa: S108
        assert doc.metadata["parser"] == "vlm"
        assert doc.page_count == 1

    def test_router_parse_fails_for_unregistered_type(self) -> None:
        router = ParserRouter()
        with pytest.raises(ValueError, match="No parser found"):
            router.parse(Path("unknown.jfif"))

    def test_router_opendataloader_import_error(self) -> None:
        """OpenDataLoaderParser raises ImportError when package is not installed."""
        with pytest.raises(ImportError, match="opendataloader-pdf"):
            OpenDataLoaderParser()
        assert OpenDataLoaderParser.supported_extensions == {FileExtension.PDF}
