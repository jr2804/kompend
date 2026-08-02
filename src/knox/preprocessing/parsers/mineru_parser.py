"""MinerU-based parser (opendatalab/MinerU) — layout-aware PDF and Office extraction.

MinerU is a high-accuracy document parser with three backends:

* ``pipeline`` — traditional OCR + layout-detection model (default).
* ``vlm`` — vision-language model backend (needs ``mineru-vl-utils`` + GPU server).
* ``hybrid`` — pipeline for text-heavy pages, VLM for complex layouts.

This parser drives MinerU's :func:`do_parse` API, which converts a source file
to Markdown (with extracted images) in a temp directory. The resulting Markdown
is then walked into KNOX :class:`DocumentElement` values, mirroring the
:mod:`liteparse_parser` approach so the downstream chunker/resolver/VLM path is
unchanged.

MinerU has no internal figure captioning, so ``figure_enrichment`` is ``False``
and every extracted figure flows through ``VLMEnricher``.
"""

from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path
from typing import Any

from mineru.cli.common import do_parse
from mineru.utils.enum_class import MakeMode

from knox.core.models import DocumentElement, ElementType, MetadataField, ParsedDocument
from knox.logging import get_logger
from knox.preprocessing.parsers._registry import ParserBuildOptions, register_backend
from knox.preprocessing.parsers.base import (
    IMAGE_SCALE_DEFAULT,
    BaseParser,
    FileExtension,
)

logger = get_logger(__name__)

# Markdown line classifiers (shared shape with the LiteParse walker).
_IMAGE_REF_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
_TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+\|?\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

_MIME_BY_FMT: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
}

#: Backends accepted by :func:`do_parse` (normalized via ``normalize_backend``).
_VALID_BACKENDS = {"pipeline", "vlm", "hybrid"}

#: Parse methods for the pipeline backend: ``auto`` (OCR auto-detect),
#: ``ocr`` (force OCR), ``txt`` (text-layer only).
_VALID_PARSE_METHODS = {"auto", "ocr", "txt"}


class MinerUParser(BaseParser):
    """Parser using MinerU (opendatalab/MinerU) for layout-aware extraction.

    MinerU converts PDF and Office documents to Markdown with extracted images,
    via three selectable backends (``pipeline``, ``vlm``, ``hybrid``). The
    Markdown output is walked into KNOX ``DocumentElement`` values, matching the
    :class:`DoclingParser`/``LiteParseParser`` contract so downstream stages are
    unchanged.
    """

    supported_extensions: set[FileExtension] = {
        FileExtension.PDF,
        FileExtension.DOCX,
        FileExtension.PPTX,
        FileExtension.XLSX,
    }

    # No internal captioning — VLM still captions every figure.
    figure_enrichment: bool = False

    def __init__(
        self,
        *,
        backend: str = "pipeline",
        parse_method: str = "auto",
        formula_enable: bool = True,
        table_enable: bool = True,
        image_mode: str = "file",
        images_dir: Path | None = None,
        images_scale: float = IMAGE_SCALE_DEFAULT,
        **kwargs: Any,
    ) -> None:
        """Initialize the parser.

        Args:
            backend: MinerU backend — ``"pipeline"`` (OCR + layout model),
                ``"vlm"`` (vision-language model), or ``"hybrid"``.
            parse_method: Pipeline OCR strategy — ``"auto"`` (detect),
                ``"ocr"`` (force), ``"txt"`` (text-layer only).
            formula_enable: Run LaTeX formula recognition.
            table_enable: Run table structure recognition.
            image_mode: ``"file"`` (write images to ``images_dir``) or
                ``"base64"`` (inline data URIs).
            images_dir: Directory for extracted images. When ``None``, defaults
                to ``<source>_images/`` next to each source file.
            images_scale: Image resolution multiplier (unused by MinerU directly;
                kept for API symmetry with other parsers).
            **kwargs: Accepted and ignored (BaseParser protocol compliance).
        """
        backend = backend.lower()
        parse_method = parse_method.lower()
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"Unknown MinerU backend {backend!r}. Valid: {sorted(_VALID_BACKENDS)}.")
        if parse_method not in _VALID_PARSE_METHODS:
            raise ValueError(f"Unknown MinerU parse_method {parse_method!r}. Valid: {sorted(_VALID_PARSE_METHODS)}.")
        self.backend = backend
        self.parse_method = parse_method
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.image_mode = image_mode
        self.images_dir = images_dir
        self.images_scale = images_scale

    def parse(self, source_file: Path) -> ParsedDocument:
        """Parse a single source file via MinerU ``do_parse``.

        MinerU writes ``{stem}.md`` + ``images/`` under a temp directory. The
        Markdown is read back and walked into :class:`DocumentElement` values.
        """
        source_bytes = source_file.read_bytes()
        stem = source_file.stem

        with tempfile.TemporaryDirectory() as tmpdir:
            # do_parse creates {tmpdir}/{stem}/{parse_method}/{stem}.md + images/
            do_parse(
                output_dir=tmpdir,
                pdf_file_names=[stem],
                pdf_bytes_list=[source_bytes],
                p_lang_list=[""],  # auto-detect language
                backend=self.backend,
                parse_method=self.parse_method,
                formula_enable=self.formula_enable,
                table_enable=self.table_enable,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=True,
                f_dump_middle_json=False,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
                f_make_md_mode=MakeMode.MM_MD,
            )

            md_path = Path(tmpdir) / stem / self._output_subdir() / f"{stem}.md"
            images_root = md_path.parent / "images"

            markdown = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
            images_by_name = self._load_images(images_root)

        elements, title = _markdown_to_elements(
            markdown,
            source_file,
            images_by_name,
            image_mode=self.image_mode,
            images_dir=self._resolve_images_dir(source_file),
        )

        page_count = _estimate_page_count(elements)

        return ParsedDocument(
            source_file=source_file,
            elements=elements,
            title=title or stem,
            page_count=page_count,
            metadata={
                "parser": "mineru",
                "backend": self.backend,
                "parse_method": self.parse_method,
            },
        )

    def _output_subdir(self) -> str:
        """MinerU groups output by parse method: ``pipeline``/``ocr``/``office``."""
        if self.backend == "pipeline":
            return self.parse_method
        return self.backend

    def _resolve_images_dir(self, source_file: Path) -> Path:
        if self.images_dir is not None:
            return self.images_dir
        return source_file.parent / f"{source_file.stem}_images"

    @staticmethod
    def _load_images(images_root: Path) -> dict[str, bytes]:
        """Map ``{image_name}.ext → raw bytes`` for every file under ``images_root``."""
        result: dict[str, bytes] = {}
        if not images_root.is_dir():
            return result
        for path in images_root.iterdir():
            if path.is_file():
                result[path.name] = path.read_bytes()
        return result


def _markdown_to_elements(
    markdown: str,
    source_file: Path,
    images_by_name: dict[str, bytes],
    *,
    image_mode: str,
    images_dir: Path,
) -> tuple[list[DocumentElement], str]:
    """Convert MinerU Markdown into KNOX elements.

    MinerU emits page-level structure only in its middle JSON (not the Markdown
    itself), so page numbers are not available per-element — they default to
    ``None``. Reading order is derived from document order.

    Returns ``(elements, first_h1_title)``.
    """
    lines = markdown.splitlines()
    elements: list[DocumentElement] = []
    first_h1 = ""
    reading_order = 0
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2).strip()
            if level == 1 and not first_h1:
                first_h1 = text
            elements.append(
                DocumentElement(
                    type=ElementType.HEADING,
                    content=text,
                    reading_order=reading_order,
                    heading_level=level,
                    source_file=source_file,
                )
            )
            reading_order += 1
            i += 1
            continue

        img_match = _IMAGE_REF_RE.match(stripped)
        if img_match:
            element = _build_image_element(
                img_match.group(1),
                img_match.group(2),
                reading_order,
                source_file,
                images_by_name,
                image_mode=image_mode,
                images_dir=images_dir,
            )
            if element is not None:
                elements.append(element)
                reading_order += 1
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1].strip()):
            grid, consumed = _parse_table(lines, i)
            if grid:
                elements.append(
                    DocumentElement(
                        type=ElementType.TABLE,
                        content=grid,
                        reading_order=reading_order,
                        source_file=source_file,
                    )
                )
                reading_order += 1
            i += consumed
            continue

        # Default: gather a paragraph of consecutive plain-text lines.
        paragraph: list[str] = [stripped]
        j = i + 1
        while j < n:
            nxt = lines[j].strip()
            if not nxt or _HEADING_RE.match(nxt) or _IMAGE_REF_RE.match(nxt) or nxt.startswith("|"):
                break
            paragraph.append(nxt)
            j += 1
        elements.append(
            DocumentElement(
                type=ElementType.TEXT,
                content="\n".join(paragraph),
                reading_order=reading_order,
                source_file=source_file,
            )
        )
        reading_order += 1
        i = j

    return elements, first_h1


def _build_image_element(
    alt_text: str,
    ref_url: str,
    reading_order: int,
    source_file: Path,
    images_by_name: dict[str, bytes],
    *,
    image_mode: str,
    images_dir: Path,
) -> DocumentElement | None:
    """Resolve a MinerU markdown image ref (``images/xxx.jpg``) to an IMAGE element."""
    image_name = _ref_to_name(ref_url)
    raw = images_by_name.get(image_name)
    if raw is None:
        logger.debug("MinerU image ref %r has no matching file — skipped", ref_url)
        return None

    fmt = Path(image_name).suffix.lstrip(".").lower()
    mime = _MIME_BY_FMT.get(fmt, "image/png")

    if image_mode == "base64":
        data_uri = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        content = data_uri
        meta: dict[str, Any] = {MetadataField.IMAGE_BASE64: data_uri, "image_mode": "base64"}
    else:
        target_dir = images_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{source_file.stem}_img_{image_name}"
        out_path = target_dir / out_name
        out_path.write_bytes(raw)
        content = str(out_path)
        meta = {MetadataField.IMAGE_PATH: str(out_path), "image_mode": "file"}

    if alt_text:
        meta["alt_text"] = alt_text

    return DocumentElement(
        type=ElementType.IMAGE,
        content=content,
        reading_order=reading_order,
        metadata=meta,
        source_file=source_file,
    )


def _ref_to_name(ref_url: str) -> str:
    """``images/foo.jpg`` → ``foo.jpg`` (keep extension; MinerU emits ``images/<name>``)."""
    return Path(ref_url).name


def _parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    """Parse a GFM table starting at ``start``. Returns ``(grid, consumed)``."""
    block: list[str] = []
    j = start
    while j < len(lines) and lines[j].strip().startswith("|"):
        block.append(lines[j].strip())
        j += 1
    grid: list[list[str]] = []
    for row in block:
        if _TABLE_SEP_RE.match(row):
            continue
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        grid.append(cells)
    return grid, j - start


def _estimate_page_count(elements: list[DocumentElement]) -> int:
    """MinerU Markdown has no page markers; page_count is unknown (0)."""
    return 0


def _factory(options: ParserBuildOptions) -> MinerUParser:
    return MinerUParser(
        image_mode=options.image_mode,
        images_dir=options.images_dir,
        images_scale=options.images_scale,
    )


register_backend("mineru", _factory, parser_cls=MinerUParser)
