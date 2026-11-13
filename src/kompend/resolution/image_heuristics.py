"""Bounded, model-free pre-filter for figures headed to the VLM.

The VLM path already screens figures by byte size and minimum dimension.
Those floors miss the large-but-empty classes: a solid fill, a near-blank
page, or a separator spanning a wide canvas all clear the floors while
carrying no information for the caption model.

This module adds two cheap stages ahead of the caption call:

- **Structure** reads only the image header: an extreme aspect ratio with a
  thin short edge is a line, rule, or border.
- **Pixels** decodes a downscaled sample (bounded by a pixel and long-edge
  cap): a near-uniform fill or a frame with almost no edges and no variance
  is decorative.

Every verdict is conservative. A confident ``DROP`` needs a specific reason;
anything ambiguous is ``UNSURE`` and flows on to the existing gates. A
transparent canvas is a hint, never a drop on its own: diagrams routinely
have transparent backgrounds. Unreadable inputs are ``UNSURE`` too, so a
malformed header can never suppress a figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from PIL import Image, ImageFilter, ImageStat

from kompend.logging import get_logger

logger = get_logger(__name__)

#: Grayscale value above which an edge-filter pixel counts as an edge.
_EDGE_THRESHOLD: Final[int] = 32

#: Alpha values at or below this are treated as fully transparent.
_TRANSPARENT_ALPHA: Final[int] = 8


class ImageRelevance(StrEnum):
    """The typed decision of the pre-filter."""

    KEEP = "keep"
    DROP = "drop"
    UNSURE = "unsure"


@dataclass(frozen=True, slots=True)
class HeuristicVerdict:
    """One pre-filter decision with its reason and confidence.

    ``confidence`` is only meaningful for :attr:`ImageRelevance.DROP`; the
    enricher drops a figure only when it clears the configured minimum.
    """

    decision: ImageRelevance
    reason: str
    confidence: float
    stage: str


@dataclass(frozen=True, slots=True)
class HeuristicConfig:
    """Thresholds for the pre-filter, supplied by the compile config.

    No defaults live here: ``CompileConfig`` owns the defaults, and the
    builder in the resolve stage copies them in. That keeps one source of
    truth for every knob and lets tests construct an explicit config.
    """

    enabled: bool
    min_aspect_ratio: float
    max_thin_edge: int
    max_analysis_edge: int
    max_analysis_pixels: int
    uniform_stddev: float
    near_blank_stddev: float
    near_blank_edge_density: float
    transparency_ratio: float
    min_confidence: float


def assess_image(path: str | Path, config: HeuristicConfig) -> HeuristicVerdict:
    """Classify one image as ``KEEP``, ``DROP``, or ``UNSURE``.

    Never raises: an unreadable or pathological image is ``UNSURE`` so the
    caller can fall through to the existing gates and the VLM.
    """
    image_path = Path(path)
    try:
        width, height = _header_size(image_path)
    except Exception as exc:
        logger.debug("Image heuristics could not read %s: %s — unsure", image_path, exc)
        return HeuristicVerdict(ImageRelevance.UNSURE, f"unreadable header: {exc}", 0.0, "structure")
    if width <= 0 or height <= 0:
        return HeuristicVerdict(ImageRelevance.UNSURE, f"empty dimensions ({width}×{height})", 0.0, "structure")

    structural = _structural_verdict(width, height, config)
    if structural is not None:
        return structural

    if width * height > config.max_analysis_pixels:
        return HeuristicVerdict(
            ImageRelevance.UNSURE,
            f"analysis skipped: {width}×{height} exceeds {config.max_analysis_pixels}px",
            0.0,
            "structure",
        )
    return _pixel_verdict(image_path, config)


def _header_size(path: Path) -> tuple[int, int]:
    """Read the declared size without decoding pixels."""
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _structural_verdict(width: int, height: int, config: HeuristicConfig) -> HeuristicVerdict | None:
    """Stage A: a thin, extremely elongated canvas is a line or border."""
    short_edge = min(width, height)
    long_edge = max(width, height)
    aspect = long_edge / short_edge
    if aspect >= config.min_aspect_ratio and short_edge <= config.max_thin_edge:
        return HeuristicVerdict(
            ImageRelevance.DROP,
            f"line-like ({width}×{height}, aspect {aspect:.0f})",
            0.95,
            "structure",
        )
    return None


def _pixel_verdict(path: Path, config: HeuristicConfig) -> HeuristicVerdict:
    """Stage B: bounded statistics on a downscaled sample.

    The image is decoded once at no more than ``max_analysis_pixels``;
    JPEG sources use ``draft`` so the decoder can reduce before allocating
    the full frame.
    """
    try:
        with Image.open(path) as opened:
            if opened.format == "JPEG":
                opened.draft("RGB", (config.max_analysis_edge, config.max_analysis_edge))
            working = opened.convert("RGBA")
        working.thumbnail((config.max_analysis_edge, config.max_analysis_edge), Image.Resampling.BILINEAR)
        flattened = _flatten(working)
        transparent_ratio = _transparent_ratio(working)
    except Exception as exc:
        logger.debug("Image heuristics could not analyse %s: %s — unsure", path, exc)
        return HeuristicVerdict(ImageRelevance.UNSURE, f"unreadable raster: {exc}", 0.0, "pixels")

    stddev = float(ImageStat.Stat(flattened).stddev[0])
    if stddev <= config.uniform_stddev:
        return HeuristicVerdict(
            ImageRelevance.DROP,
            f"near-uniform fill (stddev {stddev:.2f})",
            0.97,
            "pixels",
        )

    edge_density = _edge_density(flattened)
    if stddev <= config.near_blank_stddev and edge_density <= config.near_blank_edge_density:
        return HeuristicVerdict(
            ImageRelevance.DROP,
            f"near-blank frame (stddev {stddev:.2f}, edge density {edge_density:.4f})",
            0.9,
            "pixels",
        )

    if transparent_ratio >= config.transparency_ratio:
        return HeuristicVerdict(
            ImageRelevance.UNSURE,
            f"transparency-dominant canvas ({transparent_ratio:.1%}) — possible diagram",
            0.0,
            "pixels",
        )
    return HeuristicVerdict(ImageRelevance.UNSURE, f"no decorative signature (stddev {stddev:.2f})", 0.0, "pixels")


def _flatten(image: Image.Image) -> Image.Image:
    """Composite alpha onto white, then convert to grayscale."""
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("L")


def _transparent_ratio(image: Image.Image) -> float:
    """Fraction of pixels that are (near-)fully transparent."""
    alpha = image.getchannel("A")
    histogram = alpha.histogram()
    total = image.width * image.height
    return sum(histogram[: _TRANSPARENT_ALPHA + 1]) / total if total else 0.0


def _edge_density(gray: Image.Image) -> float:
    """Fraction of pixels the edge filter marks, on the downscaled sample."""
    edges = gray.filter(ImageFilter.FIND_EDGES)
    histogram = edges.histogram()
    total = edges.width * edges.height
    return sum(histogram[_EDGE_THRESHOLD + 1 :]) / total if total else 0.0
