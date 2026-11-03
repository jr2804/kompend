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
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Final

from parsecraft.ir.models import AssetKind, DocumentResult, ExtractedAsset, StructuredChunk

from kompend.core.config import ImageMode
from kompend.core.models import MetadataField
from kompend.logging import get_logger


class ImageMediaType(StrEnum):
    """The media types an extracted figure may arrive as."""

    PNG = "image/png"
    JPEG = "image/jpeg"
    GIF = "image/gif"
    WEBP = "image/webp"
    BMP = "image/bmp"
    TIFF = "image/tiff"
    SVG = "image/svg+xml"


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
