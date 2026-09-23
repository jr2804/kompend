"""Parser capability defaults — every backend runs with full extraction on.

Two regression classes:

- **LiteParse figures**: ``img_<id>.png`` refs (liteparse >= 2.x) failed the
  ``images_by_id`` lookup because ``_ref_to_id`` only stripped the legacy
  ``image_`` prefix. Every figure was silently dropped (0/24 resolved).
- **Docling formulas/code**: the pipeline was built with picture options only,
  so Docling's own defaults (``do_formula_enrichment=False``,
  ``do_code_enrichment=False``) stripped every formula and code block.
  ``[pdf]`` now drives all three passes and defaults them on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from docling.datamodel.pipeline_options import TableFormerMode

from knox.core.config import PdfConfig, PdfOcrMode, PdfTableMode
from knox.core.config.engine._env import read_env_vars
from knox.core.models import ElementType, ParsedDocument
from knox.preprocessing.parsers import liteparse_parser as lp
from knox.preprocessing.parsers.docling_parser import DoclingParser
from knox.preprocessing.parsers.liteparse_parser import LiteParseParser


class _FakeImage:
    def __init__(self, image_id: str) -> None:
        self.id = image_id
        self.bytes = b"\x89PNG\r\n\x1a\nfake"
        self.format = "png"


class _FakePage:
    def __init__(self, page_num: int, markdown: str) -> None:
        self.page_num = page_num
        self.markdown = markdown


class _FakeResult:
    def __init__(self, pages: list[_FakePage], images: list[_FakeImage]) -> None:
        self.pages = pages
        self.images = images


class _FakeLiteParse:
    """Captures constructor kwargs and returns a fixed ParseResult."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs

    def parse(self, _source: str) -> _FakeResult:
        page = _FakePage(1, "Intro text.\n\n![fig](img_p1_2.png)\n\n")
        return _FakeResult([page], [_FakeImage("p1_2")])


# ── LiteParse figure refs ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("img_p1_2.png", "p1_2"),
        ("image_p1_0.png", "p1_0"),
        ("p1_0.png", "p1_0"),
        ("images/img_p3_1.jpeg", "p3_1"),
        ("img_p9_1", "p9_1"),
    ],
)
def test_ref_to_id_strips_known_prefixes(ref: str, expected: str) -> None:
    assert lp._ref_to_id(ref) == expected


def test_liteparse_emits_image_elements_for_img_refs(tmp_path: Path) -> None:
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4 dummy")

    with patch.object(lp, "LiteParse", _FakeLiteParse):
        doc = LiteParseParser(image_mode="base64").parse(src)

    images = [e for e in doc.elements if e.type is ElementType.IMAGE]
    assert len(images) == 1, "img_ ref must resolve to an IMAGE element"
    assert str(images[0].content).startswith("data:image/png;base64,")


def test_liteparse_requests_image_extraction(tmp_path: Path) -> None:
    """Full capability: extract_images must be requested, not left to default."""
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4 dummy")

    with patch.object(lp, "LiteParse", _FakeLiteParse):
        LiteParseParser(image_mode="base64").parse(src)

    assert _FakeLiteParse.last_kwargs.get("extract_images") is True


# ── PdfConfig enrichment flags ─────────────────────────────────────────


def test_pdf_config_enrichment_defaults() -> None:
    cfg = PdfConfig()
    assert cfg.table_structure is True
    assert cfg.formula_enrichment is True
    # Code enrichment is off by default: born-digital code is already text, and
    # the CodeFormulaV2 VLM costs ~140s per code block on CPU.
    assert cfg.code_enrichment is False


def test_pdf_config_ocr_and_table_mode_defaults() -> None:
    cfg = PdfConfig()
    assert cfg.ocr is PdfOcrMode.AUTO
    assert cfg.table_mode is PdfTableMode.ACCURATE


def test_pdf_config_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KNOX_PDF_FORMULA_ENRICHMENT", "0")
    monkeypatch.setenv("KNOX_PDF_CODE_ENRICHMENT", "false")
    result = read_env_vars()
    pdf_section = result["compile"]["pdf"]
    assert pdf_section["formula_enrichment"] is False
    assert pdf_section["code_enrichment"] is False
    assert "table_structure" not in pdf_section  # untouched flag stays default


# ── Docling pipeline wiring ────────────────────────────────────────────


def test_docling_pipeline_reflects_pdf_config() -> None:
    parser = DoclingParser(
        image_mode="base64",
        pdf_config=PdfConfig(table_structure=True, formula_enrichment=True, code_enrichment=True),
    )
    opts = parser._pipeline_options(generate_picture_images=True)  # noqa: SLF001
    assert opts.do_table_structure is True
    assert opts.do_formula_enrichment is True
    assert opts.do_code_enrichment is True
    assert opts.generate_picture_images is True


def test_docling_pipeline_can_disable_enrichment() -> None:
    parser = DoclingParser(
        image_mode="base64",
        pdf_config=PdfConfig(table_structure=False, formula_enrichment=False, code_enrichment=False),
    )
    opts = parser._pipeline_options(generate_picture_images=True)  # noqa: SLF001
    assert opts.do_table_structure is False
    assert opts.do_formula_enrichment is False
    assert opts.do_code_enrichment is False


def test_docling_enrichment_independent_of_image_mode() -> None:
    """Enrichment must not hinge on image_mode — the old else-branch dropped it."""
    parser = DoclingParser(image_mode="none", pdf_config=PdfConfig())
    opts = parser._pipeline_options(generate_picture_images=False)  # noqa: SLF001
    assert opts.do_formula_enrichment is True
    assert opts.do_code_enrichment is False
    assert opts.generate_picture_images is False


def test_docling_pipeline_threads_from_config() -> None:
    """[compile.pdf] num_threads must reach Docling's accelerator options."""
    parser = DoclingParser(pdf_config=PdfConfig(num_threads=11))
    opts = parser._pipeline_options(generate_picture_images=False)  # noqa: SLF001
    assert opts.accelerator_options.num_threads == 11


def test_docling_pipeline_do_ocr_flag() -> None:
    parser = DoclingParser(pdf_config=PdfConfig())
    off = parser._pipeline_options(generate_picture_images=False, do_ocr=False)  # noqa: SLF001
    on = parser._pipeline_options(generate_picture_images=False, do_ocr=True)  # noqa: SLF001
    assert off.do_ocr is False
    assert on.do_ocr is True


def test_docling_pipeline_table_mode_from_config() -> None:
    parser = DoclingParser(pdf_config=PdfConfig(table_mode=PdfTableMode.FAST))
    opts = parser._pipeline_options(generate_picture_images=False)  # noqa: SLF001
    assert opts.table_structure_options.mode is TableFormerMode.FAST


def test_docling_parse_returns_parsed_document() -> None:
    """Sanity: the pipeline options are accepted by Docling's constructor."""
    parser = DoclingParser(image_mode="base64", pdf_config=PdfConfig())
    converter = parser._get_converter()  # noqa: SLF001
    assert converter is not None
    assert isinstance(ParsedDocument(source_file=Path("x"), elements=[], title="t"), ParsedDocument)
