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

import importlib
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from knox.core.models import DocumentElement, ParsedDocument
from knox.logging import get_logger
from knox.preprocessing.parsers._registry import ParserBuildOptions, register_backend
from knox.preprocessing.parsers.base import (
    IMAGE_SCALE_DEFAULT,
    BaseParser,
    FileExtension,
)
from knox.preprocessing.parsers.markdown import (
    ImagePayload,
    MarkdownContext,
    markdown_to_elements,
)

# Markdown line classifiers (shared shape with the LiteParse walker).
_VALID_BACKENDS = {"pipeline", "vlm", "hybrid"}

#: Parse methods for the pipeline backend: ``auto`` (OCR auto-detect),
#: ``ocr`` (force OCR), ``txt`` (text-layer only).
_VALID_PARSE_METHODS = {"auto", "ocr", "txt"}

# Optional backend: mineru is not a locked dependency — installed manually
# (see pyproject optional-dependencies comment). Importing raises ImportError
# when absent; the parser registry relies on that to skip this backend.
# ``importlib`` keeps ty from flagging a module missing from this env.
do_parse = importlib.import_module("mineru.cli.common").do_parse
MakeMode = importlib.import_module("mineru.utils.enum_class").MakeMode

logger = get_logger(__name__)


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

            md_file = Path(tmpdir) / stem / self._output_subdir() / f"{stem}.md"
            images_root = md_file.parent / "images"

            markdown = md_file.read_text(encoding="utf-8") if md_file.exists() else ""
            images_by_name = self._load_images(images_root)

        elements, _next_order, title = markdown_to_elements(
            markdown,
            MarkdownContext(
                source_file=source_file,
                image_mode=self.image_mode,
                images_dir=self._resolve_images_dir(source_file),
                resolve_image=_mineru_image_resolver(images_by_name),
            ),
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


def _mineru_image_resolver(images_by_name: dict[str, bytes]) -> Callable[[str, str], ImagePayload | None]:
    """Look up figure bytes in MinerU's extracted ``images/`` directory."""

    def resolve(_alt_text: str, ref_url: str) -> ImagePayload | None:
        image_name = Path(ref_url).name
        raw = images_by_name.get(image_name)
        if raw is None:
            logger.debug("MinerU image ref %r has no matching file — skipped", ref_url)
            return None
        fmt = Path(image_name).suffix.lstrip(".").lower() or "png"
        return ImagePayload(data=raw, format=fmt, name=Path(image_name).stem)

    return resolve


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
