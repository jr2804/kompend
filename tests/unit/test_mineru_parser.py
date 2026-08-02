"""Unit tests for the MinerU parser backend (knox-889y).

The real MinerU ``do_parse`` pulls in torch/ONNX/OCR model stacks (40+ second
import), so every test mocks ``do_parse`` to write a canned Markdown file +
image to the temp directory that ``do_parse`` would produce. This keeps the
tests fast (rule #276) while exercising the real Markdown→element walker.
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import patch

import pytest

from knox.core.models.enums import ElementType, MetadataField
from knox.preprocessing.parsers._registry import (
    PARSER_BACKENDS,
    ParserBuildOptions,
    get_backend,
    get_registered_parsers,
)
from knox.preprocessing.parsers.base import FileExtension
from knox.preprocessing.parsers.mineru_parser import MinerUParser

# ── Registration ────────────────────────────────────────────────────────────


class TestMinerURegistration:
    def test_backend_registered(self) -> None:
        assert "mineru" in PARSER_BACKENDS

    def test_metadata_present(self) -> None:
        names = {entry["name"] for entry in get_registered_parsers()}
        assert "mineru" in names

    def test_factory_builds_mineru_class(self) -> None:
        parser = get_backend("mineru")(ParserBuildOptions())
        assert isinstance(parser, MinerUParser)


# ── Supported extensions ────────────────────────────────────────────────────


def test_supported_extensions() -> None:
    assert MinerUParser.supported_extensions == {
        FileExtension.PDF,
        FileExtension.DOCX,
        FileExtension.PPTX,
        FileExtension.XLSX,
    }


def test_figure_enrichment_is_false() -> None:
    """MinerU has no internal captioning — VLM must enrich every figure."""
    assert MinerUParser.figure_enrichment is False


def test_can_parse_pdf(tmp_path: Path) -> None:
    parser = MinerUParser()
    assert parser.can_parse(Path("paper.pdf"))
    assert parser.can_parse(Path("doc.docx"))
    assert parser.can_parse(Path("slides.pptx"))
    assert parser.can_parse(Path("sheet.xlsx"))
    assert not parser.can_parse(Path("page.html"))
    assert not parser.can_parse(Path("notes.md"))


# ── Constructor validation ──────────────────────────────────────────────────


@pytest.mark.parametrize("backend", ["pipeline", "vlm", "hybrid"])
def test_valid_backends(backend: str) -> None:
    parser = MinerUParser(backend=backend)
    assert parser.backend == backend


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown MinerU backend"):
        MinerUParser(backend="nonsense")


@pytest.mark.parametrize("method", ["auto", "ocr", "txt"])
def test_valid_parse_methods(method: str) -> None:
    parser = MinerUParser(parse_method=method)
    assert parser.parse_method == method


def test_invalid_parse_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown MinerU parse_method"):
        MinerUParser(parse_method="nonsense")


def test_backend_case_insensitive() -> None:
    parser = MinerUParser(backend="PIPELINE")
    assert parser.backend == "pipeline"


# ── Markdown walking (mocked do_parse) ──────────────────────────────────────


def _fake_do_parse_factory(markdown: str, image_bytes: bytes | None = None) -> object:
    """Build a mock ``do_parse`` that writes the canned markdown + image.

    Mirrors MinerU's real output layout: ``{tmpdir}/{stem}/{method}/{stem}.md``
    plus ``images/`` next to it.
    """

    def _fake_do_parse(
        output_dir: str,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        p_lang_list: list[str],
        **kwargs: object,
    ) -> None:
        stem = pdf_file_names[0]
        backend = kwargs.get("backend", "pipeline")
        parse_method = kwargs.get("parse_method", "auto")
        subdir = parse_method if backend == "pipeline" else backend
        md_dir = Path(output_dir) / stem / subdir
        images_dir = md_dir / "images"
        md_dir.mkdir(parents=True, exist_ok=True)
        (md_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
        if image_bytes is not None:
            images_dir.mkdir(parents=True, exist_ok=True)
            (images_dir / "fig1.jpg").write_bytes(image_bytes)

    return _fake_do_parse


SIMPLE_MARKDOWN = """\
# Document Title

## Introduction

This is the first paragraph about transformers.

It continues on a second line.

| Col A | Col B |
| ----- | ----- |
| 1     | 2     |
| 3     | 4     |

![Figure 1 caption](images/fig1.jpg)

Final paragraph after the figure.
"""


class TestMarkdownWalking:
    def test_parse_returns_parsed_document(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        with patch("mineru.cli.common.do_parse", side_effect=_fake_do_parse_factory(SIMPLE_MARKDOWN)):
            parser = MinerUParser()
            doc = parser.parse(pdf)

        assert doc.source_file == pdf
        assert doc.title == "Document Title"
        assert doc.metadata["parser"] == "mineru"

    def test_heading_and_text_extraction(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        with patch("mineru.cli.common.do_parse", side_effect=_fake_do_parse_factory(SIMPLE_MARKDOWN)):
            doc = MinerUParser().parse(pdf)

        headings = [e for e in doc.elements if e.type == ElementType.HEADING]
        assert len(headings) == 2
        assert headings[0].content == "Document Title"
        assert headings[0].heading_level == 1
        assert headings[1].content == "Introduction"
        assert headings[1].heading_level == 2

    def test_table_extraction(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        with patch("mineru.cli.common.do_parse", side_effect=_fake_do_parse_factory(SIMPLE_MARKDOWN)):
            doc = MinerUParser().parse(pdf)

        tables = [e for e in doc.elements if e.type == ElementType.TABLE]
        assert len(tables) == 1
        grid = tables[0].content
        assert grid == [["Col A", "Col B"], ["1", "2"], ["3", "4"]]

    def test_image_extraction_file_mode(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        image = b"\xff\xd8\xff\xe0fake-jpeg-data"
        with patch("mineru.cli.common.do_parse", side_effect=_fake_do_parse_factory(SIMPLE_MARKDOWN, image)):
            doc = MinerUParser(image_mode="file").parse(pdf)

        images = [e for e in doc.elements if e.type == ElementType.IMAGE]
        assert len(images) == 1
        img = images[0]
        assert MetadataField.IMAGE_PATH in img.metadata
        out_path = Path(img.metadata[MetadataField.IMAGE_PATH])
        assert out_path.exists()
        assert out_path.read_bytes() == image
        assert img.metadata["alt_text"] == "Figure 1 caption"

    def test_image_extraction_base64_mode(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        image = b"\xff\xd8\xff\xe0fake-jpeg-data"
        with patch("mineru.cli.common.do_parse", side_effect=_fake_do_parse_factory(SIMPLE_MARKDOWN, image)):
            doc = MinerUParser(image_mode="base64").parse(pdf)

        images = [e for e in doc.elements if e.type == ElementType.IMAGE]
        assert len(images) == 1
        img = images[0]
        assert MetadataField.IMAGE_BASE64 in img.metadata
        data_uri = img.metadata[MetadataField.IMAGE_BASE64]
        assert data_uri.startswith("data:image/jpeg;base64,")
        decoded = base64.b64decode(data_uri.split(",", 1)[1])
        assert decoded == image

    def test_reading_order_is_sequential(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        with patch("mineru.cli.common.do_parse", side_effect=_fake_do_parse_factory(SIMPLE_MARKDOWN)):
            doc = MinerUParser().parse(pdf)

        orders = [e.reading_order for e in doc.elements if e.reading_order is not None]
        assert orders == sorted(orders)
        assert orders[0] == 0

    def test_paragraph_merges_consecutive_lines(self, tmp_path: Path) -> None:
        """Two text lines with no blank line between them merge into one paragraph."""
        md = "# Title\n\nLine one of paragraph.\nLine two continues directly.\n"
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        with patch("mineru.cli.common.do_parse", side_effect=_fake_do_parse_factory(md)):
            doc = MinerUParser().parse(pdf)

        texts = [e for e in doc.elements if e.type == ElementType.TEXT]
        para_contents = [t.content for t in texts if "Line one" in t.content]
        assert len(para_contents) == 1
        assert "Line one of paragraph." in para_contents[0]
        assert "Line two continues directly." in para_contents[0]


# ── Metadata propagation ────────────────────────────────────────────────────


def test_metadata_records_backend_and_method(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    with patch("mineru.cli.common.do_parse", side_effect=_fake_do_parse_factory(SIMPLE_MARKDOWN)):
        doc = MinerUParser(backend="vlm").parse(pdf)

    assert doc.metadata["backend"] == "vlm"
    assert doc.metadata["parse_method"] == "auto"


def test_title_falls_back_to_stem(tmp_path: Path) -> None:
    md_no_title = "Just plain text without any heading."
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    with patch("mineru.cli.common.do_parse", side_effect=_fake_do_parse_factory(md_no_title)):
        doc = MinerUParser().parse(pdf)

    assert doc.title == "report"


def test_empty_markdown_produces_no_elements(tmp_path: Path) -> None:
    pdf = tmp_path / "blank.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    with patch("mineru.cli.common.do_parse", side_effect=_fake_do_parse_factory("")):
        doc = MinerUParser().parse(pdf)

    assert doc.elements == []
    assert doc.title == "blank"


# ── Factory options forwarding ──────────────────────────────────────────────


def test_factory_forwards_options(tmp_path: Path) -> None:
    opts = ParserBuildOptions(images_dir=tmp_path, image_mode="base64", images_scale=100.0)
    parser = get_backend("mineru")(opts)
    assert isinstance(parser, MinerUParser)
    assert parser.images_dir == tmp_path
    assert parser.image_mode == "base64"
    assert parser.images_scale == 100.0


def test_factory_defaults() -> None:
    parser = get_backend("mineru")(ParserBuildOptions())
    assert isinstance(parser, MinerUParser)
    assert parser.backend == "pipeline"
    assert parser.parse_method == "auto"
    assert parser.image_mode == "file"
