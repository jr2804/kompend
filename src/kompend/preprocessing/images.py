"""Figure images from the parse stage — IR asset in, cached file or inline payload out.

parsecraft hands figure crops over as IR assets. ``PageResult.assets`` carries
an ``ExtractedAsset`` whose ``uri`` is a ``data:image/png;base64,…`` payload;
the chunk that owns the crop points at its asset through
``metadata["asset_id"]``. mineru builds both today
(``parsecraft/backends/mineru/_impl.py``); the docling adapter maps pictures
the same way once parsecraft ships it.

Two storage modes, one per consumer:

- :attr:`ImageMode.FILE` writes the bytes into the vault's image cache
  (``VaultPaths.images_dir``) and hands the VLM stage a path to read.
- :attr:`ImageMode.BASE64` keeps the payload inline in the element metadata,
  for callers that pass bytes to a model instead of a file.

The file name is derived from the source stem (for human readability) and
a digest of the bytes, so it is content-addressed: the same figure across
re-parses keeps the same file, and a changed figure gets a new one instead
of overwriting the old crop.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Final
from xml.etree import ElementTree

import resvg_py
from parsecraft.ir.models import AssetKind, DocumentResult, ExtractedAsset, StructuredChunk
from PIL import Image

from kompend.core.config import ImageMode
from kompend.core.models import MetadataField
from kompend.logging import get_logger

logger = get_logger(__name__)


class ImageMediaType(StrEnum):
    """The media types an extracted figure may arrive as."""

    PNG = "image/png"
    JPEG = "image/jpeg"
    GIF = "image/gif"
    WEBP = "image/webp"
    BMP = "image/bmp"
    TIFF = "image/tiff"
    SVG = "image/svg+xml"


class VectorRasterError(RuntimeError):
    """A vector figure could not be rendered to pixels."""


#: Media type → file suffix. Only the raster types the VLM and the embedders
#: accept are listed; anything else keeps the media type's own subtype as a
#: suffix so the file stays openable.
_SUFFIX_BY_MEDIA_TYPE: Final[dict[ImageMediaType, str]] = {
    ImageMediaType.PNG: ".png",
    ImageMediaType.JPEG: ".jpg",
    ImageMediaType.GIF: ".gif",
    ImageMediaType.WEBP: ".webp",
    ImageMediaType.BMP: ".bmp",
    ImageMediaType.TIFF: ".tiff",
    ImageMediaType.SVG: ".svg",
}

#: Characters that may not appear in a file name on any supported platform.
_UNSAFE_NAME_CHARS: str = '<>:"/\\|?*'

#: Suffixes carrying pixels. Anything outside this set and :data:`_VECTOR_SUFFIXES`
#: is passed through untouched, so a caller's own renderer still sees it.
_RASTER_SUFFIXES: Final[frozenset[str]] = frozenset(suffix for media_type, suffix in _SUFFIX_BY_MEDIA_TYPE.items() if media_type is not ImageMediaType.SVG)

#: Suffixes holding vector geometry rather than pixels.
_VECTOR_SUFFIXES: Final[frozenset[str]] = frozenset({".svg", ".svgz"})

#: Long edge of a rasterized vector figure, in pixels. Vision models work well
#: around 1.5k, and a vector table can be arbitrarily large — capping the long
#: edge is what keeps a render from becoming a canvas of many thousands of rows.
MAX_RASTER_LONG_EDGE: Final[int] = 1600

#: Ceiling on the rendered canvas: one pathological SVG must not allocate gigabytes.
MAX_RASTER_PIXELS: Final[int] = 4_000_000

#: How often the scalar is stepped down for a figure that declares no usable
#: canvas. Four steps take 1600 px to 6 px, which no document can overshoot.
_GEOMETRYLESS_ATTEMPTS: Final[int] = 4

#: Chunk metadata key a backend uses to name the asset a chunk owns.
_ASSET_ID_KEY: str = "asset_id"

logger = get_logger(__name__)


class ImageStore:
    """Stores extracted assets and hands back the metadata that points at them.

    One store per parser run: it holds the cache directory and the mode, so
    the mapping code does not have to thread them through every call.
    """

    def __init__(self, images_dir: Path | None, mode: ImageMode = ImageMode.FILE) -> None:
        self._images_dir = images_dir
        self._mode = mode

    @property
    def enabled(self) -> bool:
        """False when the run extracts no images at all.

        A run without a cache directory cannot store a crop, so it extracts
        none: the figure's own text stays a text element.
        """
        return self._mode is not ImageMode.NONE and self._images_dir is not None

    @property
    def mode(self) -> ImageMode:
        """The configured storage mode."""
        return self._mode

    def store(self, asset: ExtractedAsset, *, source_stem: str) -> dict[str, str]:
        """The element metadata for one asset — empty when it cannot be stored."""
        return store_asset(asset, source_stem=source_stem, images_dir=self._images_dir, mode=self._mode)


def assets_by_id(document: DocumentResult) -> dict[str, ExtractedAsset]:
    """Every image asset of a document, keyed by its id.

    Assets are page-scoped in the IR but ids are unique across the document,
    so one flat map serves the whole mapping pass.
    """
    return {asset.id: asset for page in document.pages for asset in page.assets if asset.kind == AssetKind.IMAGE}


def asset_for_chunk(chunk: StructuredChunk, assets: Mapping[str, ExtractedAsset]) -> ExtractedAsset | None:
    """The image asset a chunk owns, or ``None`` when it owns none.

    A backend names its asset in the chunk metadata; a chunk without that
    name owns no image.
    """
    asset_id = chunk.metadata.get(_ASSET_ID_KEY)
    if not asset_id:
        return None
    return assets.get(asset_id)


def store_asset(
    asset: ExtractedAsset,
    *,
    source_stem: str,
    images_dir: Path | None,
    mode: ImageMode,
) -> dict[str, str]:
    """Store one asset and return the element metadata that points at it.

    Returns an empty mapping when the asset cannot be stored (no cache
    directory, an unreadable payload) — the caller keeps the text it has and
    the run continues; the reason is logged once per asset.
    """
    if mode is ImageMode.NONE:
        return {}
    payload = _payload_of(asset)
    if payload is None:
        logger.warning("Image asset '%s' carries no readable payload; skipping it", asset.id)
        return {}
    data, media_type = payload
    data_uri = _as_data_uri(data, media_type)
    if mode is ImageMode.BASE64:
        return {MetadataField.IMAGE_BASE64: data_uri}
    if images_dir is None:
        logger.warning("No image cache directory configured; asset '%s' is not stored", asset.id)
        return {}
    path = _write_file(asset, data=data, media_type=media_type, source_stem=source_stem, images_dir=images_dir)
    if path is None:
        return {}
    return {MetadataField.IMAGE_FILE: str(path)}


def _payload_of(asset: ExtractedAsset) -> tuple[bytes, str] | None:
    """The asset's bytes and media type, whatever shape its URI has."""
    uri = asset.uri
    if uri.startswith("data:"):
        return _decode_data_uri(uri)
    path = Path(uri)
    if not path.is_file():
        return None
    try:
        return path.read_bytes(), asset.media_type
    except OSError:
        return None


def _decode_data_uri(uri: str) -> tuple[bytes, str] | None:
    """Decode a ``data:<media type>;base64,<payload>`` URI."""
    header, _, payload = uri.partition(",")
    if not _:
        return None
    media_type = header[5:].split(";", 1)[0] or ImageMediaType.PNG.value
    try:
        return base64.b64decode(payload, validate=False), media_type
    except (binascii.Error, ValueError):
        return None


def _as_data_uri(data: bytes, media_type: str) -> str:
    """The inline representation of the bytes."""
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def _write_file(
    asset: ExtractedAsset,
    *,
    data: bytes,
    media_type: str,
    source_stem: str,
    images_dir: Path,
) -> Path | None:
    """Write the bytes into the cache and return the path.

    The file name is content-addressed (sha256 of the bytes) so the same
    figure across re-parses keeps the same file and a changed figure gets a
    new one. The source stem is kept for human readability only.
    """
    suffix = _suffix_for(media_type)
    digest = hashlib.sha256(data).hexdigest()[:16]
    name = f"{_safe_name(source_stem)}__{digest}{suffix}"
    target = images_dir / name
    try:
        images_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    except OSError as exc:
        logger.warning("Could not write image '%s' to the cache: %s", target, exc)
        return None
    return target


def rasterise_bytes(image_file: str | Path) -> bytes | None:
    """Raster bytes for a figure, or ``None`` when it cannot be produced."""
    try:
        return rasterise_image(image_file).read_bytes()
    except (VectorRasterError, OSError):
        return None


def rasterise_image(image_file: str | Path) -> Path:
    """A raster copy of *image_file* that a vision model or embedder can read.

    A raster file is returned as it is. A vector figure (``.svg``/``.svgz`` —
    what ``extract_vector_graphics`` writes for tables and diagrams, and what
    modern ``.docx`` files embed) is rendered to a PNG beside it on first use
    and reused afterwards while the source is unchanged; the vector original
    stays the artifact the vault publishes.

    Raises:
        VectorRasterError: the figure could not be rendered — the caller
            records it as a failed unit instead of silently dropping a figure.
    """
    source = Path(image_file)
    if is_raster_image(source):
        return source
    if source.suffix.lower() not in _VECTOR_SUFFIXES:
        msg = f"{source.suffix or 'no extension'} is neither a raster nor a vector image: {source}"
        raise VectorRasterError(msg)
    rendered = source.with_suffix(".png")
    if rendered.is_file() and rendered.stat().st_mtime >= source.stat().st_mtime:
        return rendered
    try:
        pixels = _render_vector(source)
    except Exception as exc:
        msg = f"could not render {source.name}: {exc}"
        raise VectorRasterError(msg) from exc
    try:
        rendered.write_bytes(pixels)
    except OSError as exc:
        msg = f"could not write the rendered figure {rendered.name}: {exc}"
        raise VectorRasterError(msg) from exc
    logger.debug("Rendered %s to %s (%d bytes)", source.name, rendered.name, len(pixels))
    return rendered


def is_raster_image(image_file: str | Path) -> bool:
    """Whether *image_file* already holds pixels."""
    return Path(image_file).suffix.lower() in _RASTER_SUFFIXES


def _render_vector(source: Path) -> bytes:
    """Render a vector figure to PNG bytes, bounded by :data:`MAX_RASTER_LONG_EDGE`.

    The scale comes from the **long** edge of the declared canvas, never from
    the width alone: a PDF-to-SVG export routinely states a sub-pixel width
    (``viewBox="0 0 0.13 79.4"`` for a hairline rule), so scaling that canvas to
    1600 px wide asks for a 610:1 strip of 202 Mpx - past PIL's decode ceiling.
    """
    width, height = _vector_canvas(source)
    long_edge = max(width, height)
    if long_edge <= 0:
        # No usable geometry: the renderer picks the aspect, so the scalar
        # alone decides how many pixels come back.
        return _render_geometryless(source)
    scale = min(MAX_RASTER_LONG_EDGE / long_edge, _pixel_budget_scale(max(width, 1), max(height, 1)))
    if height > width:
        pixels = bytes(resvg_py.svg_to_bytes(svg_path=str(source), height=max(1, round(height * scale))))
    else:
        pixels = bytes(resvg_py.svg_to_bytes(svg_path=str(source), width=max(1, round(width * scale))))
    shrunk = _shrink(pixels)
    if shrunk is not None:
        return shrunk
    # The renderer ignored the requested edge; fall back to the scalar ladder.
    return _render_geometryless(source)


def _render_geometryless(source: Path) -> bytes:
    """Render a figure that declares no usable canvas, stepping the scalar down.

    Each attempt quarters the width until the pixels decode inside the budget;
    a figure that still overshoots at 6 px wide is refused as a failed render
    rather than left to allocate past PIL's decompression-bomb ceiling.
    """
    width = MAX_RASTER_LONG_EDGE
    for _ in range(_GEOMETRYLESS_ATTEMPTS):
        pixels = _shrink(bytes(resvg_py.svg_to_bytes(svg_path=str(source), width=width)))
        if pixels is not None:
            return pixels
        width //= 4
    msg = f"the render stayed above PIL's decode ceiling even at {width} px wide"
    raise VectorRasterError(msg)


def _pixel_budget_scale(width: int, height: int) -> float:
    """Scale that keeps ``width * height`` under :data:`MAX_RASTER_PIXELS`."""
    area = width * height
    if area <= MAX_RASTER_PIXELS:
        return float("inf")
    return (MAX_RASTER_PIXELS / area) ** 0.5


def _shrink(pixels: bytes | bytearray) -> bytes | None:
    """Downscale rendered pixels that came back over the canvas budget.

    ``None`` means the render arrived larger than PIL will decode at all
    (:class:`PIL.Image.DecompressionBombError`); the caller re-renders smaller
    instead of handing a decompression bomb to a vision model.
    """
    data = bytes(pixels)
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height <= MAX_RASTER_PIXELS:
                return data
            resized = image.copy()
    except Image.DecompressionBombError:
        return None
    resized.thumbnail((MAX_RASTER_LONG_EDGE, MAX_RASTER_LONG_EDGE))
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG")
    return buffer.getvalue()


def _vector_canvas(source: Path) -> tuple[int, int]:
    """The figure's own pixel geometry, or ``(0, 0)`` when it does not state one."""
    try:
        root = ElementTree.parse(source).getroot()  # noqa: S314 — the parser stage produced this file
    except (ElementTree.ParseError, OSError):
        return 0, 0
    view_box = root.get("viewBox")
    if view_box:
        parts = view_box.replace(",", " ").split()
        if len(parts) == 4:
            try:
                return round(float(parts[2])), round(float(parts[3]))
            except ValueError:
                pass
    return _length(root.get("width")), _length(root.get("height"))


def _length(raw: str | None) -> int:
    """A CSS length in whole pixels, ignoring the unit (an SVG renders in user units)."""
    if not raw:
        return 0
    digits = "".join(char for char in raw if char.isdigit() or char == ".")
    try:
        return round(float(digits))
    except ValueError:
        return 0


def _suffix_for(media_type: str) -> str:
    """The file suffix of a media type.

    A known type keeps its canonical suffix; an unknown one keeps its own
    subtype, so the file stays openable.
    """
    try:
        known = ImageMediaType(media_type)
    except ValueError:
        subtype = media_type.rsplit("/", 1)[-1]
        return f".{subtype}" if subtype else ".bin"
    return _SUFFIX_BY_MEDIA_TYPE[known]


def _safe_name(value: str) -> str:
    """A file-name-safe form of *value* (path separators and reserved characters go)."""
    cleaned = "".join("_" if char in _UNSAFE_NAME_CHARS else char for char in value).strip(". ")
    return cleaned or "image"
