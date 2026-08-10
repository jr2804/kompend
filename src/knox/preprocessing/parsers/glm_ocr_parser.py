"""GLM-OCR parser (zai-org/GLM-OCR) — multimodal OCR for images and PDFs.

GLM-OCR is a lightweight (0.9B) multimodal OCR model from Zhipu AI. It accepts
raster images (PNG/JPG/TIFF/BMP/GIF/WEBP) and PDFs (rendered to images
internally) and returns structured elements labelled ``text``, ``title``,
``table``, ``formula``, and ``figure`` — which map directly onto KNOX
:class:`ElementType` values.

Two execution modes:

* **Cloud** (default): calls Zhipu MaaS. Requires ``ZHIPU_API_KEY`` env var
  (or ``api_key`` constructor arg). No local GPU needed.
* **Self-hosted**: set ``mode="selfhosted"`` and point ``api_url`` at a vLLM /
  SGLang / Ollama endpoint serving the GLM-OCR model.

The ``glmocr`` package is an optional dependency — install with::

    uv sync --group glm-ocr

GLM-OCR does no internal captioning, so ``figure_enrichment`` is ``False`` and
every extracted figure still flows through ``VLMEnricher``.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

from glmocr import PipelineResult
from glmocr import parse as glm_parse
from PIL import Image

from knox.core.models import (
    DocumentElement,
    ElementType,
    MetadataField,
    ParsedDocument,
)
from knox.logging import get_logger
from knox.preprocessing.parsers._registry import ParserBuildOptions, register_backend
from knox.preprocessing.parsers.base import (
    IMAGE_SCALE_DEFAULT,
    BaseParser,
    FileExtension,
)

# GLM-OCR element labels → KNOX ElementType.
_LABEL_TO_TYPE: dict[str, ElementType] = {
    "text": ElementType.TEXT,
    "title": ElementType.HEADING,
    "table": ElementType.TABLE,
    "formula": ElementType.FORMULA,
    "figure": ElementType.IMAGE,
    "image": ElementType.IMAGE,
    "caption": ElementType.TEXT,
    "list": ElementType.LIST_ITEM,
    "code": ElementType.CODE,
}

_MIME_BY_FMT: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
}

logger = get_logger(__name__)


class GlmOcrParser(BaseParser):
    """Parser using GLM-OCR (zai-org/GLM-OCR) for image-based OCR.

    Accepts raster images and PDFs. Uses the ``glmocr`` SDK's ``parse()``
    convenience function, which returns a :class:`PipelineResult` carrying
    structured JSON elements (per-page, with labels and bounding boxes),
    assembled Markdown, and any cropped figure regions.

    Cloud mode (default) needs ``ZHIPU_API_KEY``; self-hosted mode needs a
    running vLLM/SGLang/Ollama endpoint. Neither runs a local model inside the
    KNOX process, so CPU-only hosts are fine.
    """

    supported_extensions: set[FileExtension] = {
        FileExtension.PNG,
        FileExtension.JPG,
        FileExtension.JPEG,
        FileExtension.TIFF,
        FileExtension.TIF,
        FileExtension.BMP,
        FileExtension.GIF,
        FileExtension.WEBP,
        FileExtension.PDF,
    }

    # No internal captioning — VLM still captions every figure.
    figure_enrichment: bool = False

    def __init__(
        self,
        *,
        image_mode: str = "file",
        images_dir: Path | None = None,
        images_scale: float = IMAGE_SCALE_DEFAULT,
        api_key: str | None = None,
        api_url: str | None = None,
        mode: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the parser.

        Args:
            image_mode: ``"file"`` (write images to ``images_dir``) or
                ``"base64"`` (inline data URIs).
            images_dir: Directory for extracted figures. When ``None``, defaults
                to ``<source>_images/`` next to each source file.
            images_scale: Image resolution multiplier (``1.0`` = 72 DPI).
                Forwarded to GLM-OCR for PDF rasterisation.
            api_key: Zhipu MaaS API key. Defaults to ``ZHIPU_API_KEY`` env var.
            api_url: Endpoint URL (self-hosted mode) or MaaS override.
            mode: ``"selfhosted"`` for a local endpoint, else cloud (MaaS).
            model: Model identifier override.
            timeout: Request timeout in seconds.
            **kwargs: Accepted and ignored (BaseParser protocol compliance).
        """
        self.image_mode = image_mode
        self.images_dir = images_dir
        self.images_scale = images_scale
        self.api_key = api_key
        self.api_url = api_url
        self.mode = mode
        self.model = model
        self.timeout = timeout

    def parse(self, source_file: Path) -> ParsedDocument:
        """Parse *source_file* via GLM-OCR and return a :class:`ParsedDocument`."""
        results = glm_parse(
            str(source_file),
            api_key=self.api_key,
            api_url=self.api_url,
            mode=self.mode,
            model=self.model,
            timeout=self.timeout,
            save_layout_visualization=False,
        )
        # ``parse`` returns a single PipelineResult for a single input.
        result = (results[0] if results else None) if isinstance(results, list) else results
        if result is None:
            logger.warning("GLM-OCR returned no result for %s", source_file)
            return _empty_document(source_file)

        images_dir = self._resolve_images_dir(source_file)
        elements, title = _result_to_elements(
            result,
            source_file,
            self.image_mode,
            images_dir,
        )
        page_count = _count_pages(result)

        return ParsedDocument(
            source_file=source_file,
            elements=elements,
            title=title or source_file.stem,
            page_count=page_count,
            metadata={"parser": "glm-ocr"},
        )

    def _resolve_images_dir(self, source_file: Path) -> Path:
        if self.images_dir is not None:
            return self.images_dir
        return source_file.parent / f"{source_file.stem}_images"


def _result_to_elements(
    result: PipelineResult,
    source_file: Path,
    image_mode: str,
    images_dir: Path,
) -> tuple[list[DocumentElement], str]:
    """Walk a GLM-OCR ``PipelineResult`` into KNOX elements.

    Returns ``(elements, first_title)``. Prefers the structured ``json_result``
    (rich labels: text/title/table/formula/figure); falls back to the
    ``markdown_result`` when JSON is empty or malformed.
    """
    json_result = getattr(result, "json_result", None)
    elements: list[DocumentElement] = []
    reading_order = 0
    first_title = ""

    if json_result:
        try:
            pages = _coerce_pages(json_result)
        except (ValueError, TypeError):
            pages = []
        for page_idx, page_elements in enumerate(pages):
            page_no = page_idx + 1
            for raw in page_elements:
                if not isinstance(raw, dict):
                    continue
                label = str(raw.get("label", "text")).lower()
                content = raw.get("content")
                if content is None:
                    continue
                etype = _LABEL_TO_TYPE.get(label, ElementType.TEXT)
                if etype == ElementType.HEADING and not first_title:
                    first_title = str(content).strip().splitlines()[0] if content else ""
                elements.append(
                    DocumentElement(
                        type=etype,
                        content=_coerce_content(etype, content),
                        page=page_no,
                        reading_order=reading_order,
                        metadata=_element_metadata(label, raw),
                        source_file=source_file,
                    ),
                )
                reading_order += 1

    # Save cropped figure regions (image_files: filename → PIL.Image).
    image_files = getattr(result, "image_files", None) or {}
    for img_name, pil_img in image_files.items():
        elements.append(
            _build_image_element_from_pil(
                pil_img,
                img_name=img_name,
                source_file=source_file,
                image_mode=image_mode,
                images_dir=images_dir,
                reading_order=reading_order,
            ),
        )
        reading_order += 1

    # Fallback: if JSON produced nothing, parse the markdown.
    if not elements:
        md = getattr(result, "markdown_result", None) or ""
        if md:
            elements, reading_order, first_title = _markdown_to_elements(
                md,
                source_file,
                image_mode,
                images_dir,
                reading_order,
            )

    return elements, first_title


def _coerce_content(etype: ElementType, content: str | list) -> str | list[list[str]]:
    """Coerce a raw GLM-OCR content value into the KNOX element content shape."""
    if etype == ElementType.TABLE:
        # Tables may arrive as markdown (str) or as a 2-D list.
        if isinstance(content, list):
            return [list(row) for row in content]
        return _markdown_table_to_grid(str(content))
    return str(content)


def _markdown_table_to_grid(md: str) -> list[list[str]]:
    """Best-effort parse of a GFM markdown table into a 2-D string grid."""
    rows: list[list[str]] = []
    for line in str(md).strip().splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        # Skip separator rows (| --- | :---: |).
        stripped.strip("|").strip()
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(set(c) <= {"-", ":"} for c in cells):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        rows.append(cells)
    return rows


def _element_metadata(label: str, raw: dict) -> dict[str, Any]:
    """Build KNOX metadata for one GLM-OCR element."""
    meta: dict[str, Any] = {"label": label}
    bbox = raw.get("bbox_2d")
    if bbox:
        meta["bbox"] = list(bbox) if not isinstance(bbox, list) else bbox
    idx = raw.get("index")
    if idx is not None:
        meta["index"] = idx
    return meta


def _build_image_element_from_pil(
    pil_img: Image.Image,
    *,
    img_name: str,
    source_file: Path,
    image_mode: str,
    images_dir: Path,
    reading_order: int,
) -> DocumentElement:
    """Save a cropped PIL figure (from GLM-OCR ``image_files``) as a KNOX image."""
    fmt = (pil_img.format or "PNG").lower().replace("jpeg", "jpg")
    mime = _MIME_BY_FMT.get(fmt, "image/png")
    buf = io.BytesIO()
    save_fmt = "JPEG" if fmt in {"jpg", "jpeg"} else (pil_img.format or "PNG")
    pil_img.save(buf, format=save_fmt)
    raw = buf.getvalue()

    page_no = 1
    if image_mode == "base64":
        data_uri = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
        content: str | bytes = data_uri
        meta: dict[str, Any] = {
            MetadataField.IMAGE_BASE64: data_uri,
            "image_mode": "base64",
        }
    else:
        images_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(img_name).stem or f"img_{reading_order}"
        out_path = images_dir / f"{source_file.stem}_{safe_name}.{fmt}"
        out_path.write_bytes(raw)
        content = str(out_path)
        meta = {
            MetadataField.IMAGE_PATH: str(out_path),
            "image_mode": "file",
        }

    return DocumentElement(
        type=ElementType.IMAGE,
        content=content,
        page=page_no,
        reading_order=reading_order,
        metadata=meta,
        source_file=source_file,
    )


def _markdown_to_elements(
    markdown: str,
    source_file: Path,
    image_mode: str,
    images_dir: Path,
    reading_order: int,
) -> tuple[list[DocumentElement], int, str]:
    """Fallback: split markdown into text paragraphs (one element per block)."""
    elements: list[DocumentElement] = []
    first_title = ""
    for block in markdown.split("\n\n"):
        text = block.strip()
        if not text:
            continue
        if text.startswith("# ") and not first_title:
            first_title = text.lstrip("# ").strip()
        etype = ElementType.HEADING if text.startswith("#") else ElementType.TEXT
        level = 0
        if etype == ElementType.HEADING:
            level = len(text) - len(text.lstrip("#"))
        elements.append(
            DocumentElement(
                type=etype,
                content=text.lstrip("# ").strip() if etype == ElementType.HEADING else text,
                page=1,
                reading_order=reading_order,
                heading_level=level or None,
                source_file=source_file,
            ),
        )
        reading_order += 1
    return elements, reading_order, first_title


def _count_pages(result: PipelineResult) -> int:
    """Best-effort page count from a PipelineResult."""
    json_result = getattr(result, "json_result", None)
    if json_result:
        try:
            pages = _coerce_pages(json_result)
            if pages:
                return len(pages)
        except (ValueError, TypeError):
            pass
    original_images = getattr(result, "original_images", None) or []
    return len(original_images) or 1


def _coerce_pages(json_result: PipelineResult) -> list[list[dict]]:
    """Normalise json_result into a list-of-pages (each a list of element dicts).

    GLM-OCR emits ``[[{…}, {…}], [{…}]]`` (pages → elements) but some modes
    return a flat ``[{…}]`` or a single dict.
    """
    if isinstance(json_result, str):
        json_result = json.loads(json_result)
    if isinstance(json_result, dict):
        # Single element, not paginated.
        return [[json_result]]
    if isinstance(json_result, list):
        if not json_result:
            return []
        if isinstance(json_result[0], dict):
            # Flat list of elements — treat as one page.
            return [json_result]
        if isinstance(json_result[0], list):
            return json_result  # Already pages → elements.
    return []


def _empty_document(source_file: Path) -> ParsedDocument:
    return ParsedDocument(
        source_file=source_file,
        elements=[],
        title=source_file.stem,
        page_count=0,
        metadata={"parser": "glm-ocr", "error": "no result"},
    )


def _factory(options: ParserBuildOptions) -> GlmOcrParser:
    return GlmOcrParser(
        image_mode=options.image_mode,
        images_dir=options.images_dir,
        images_scale=options.images_scale,
    )


register_backend("glm-ocr", _factory, parser_cls=GlmOcrParser)
