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
from typing import Final, Protocol

from PIL import Image, ImageFilter, ImageStat

from kompend.logging import get_logger

#: Grayscale value above which an edge-filter pixel counts as an edge.
_EDGE_THRESHOLD: Final[int] = 32

#: Alpha values at or below this are treated as fully transparent.
_TRANSPARENT_ALPHA: Final[int] = 8

logger = get_logger(__name__)


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


class HeuristicConfigSource(Protocol):
    """The compile config fields the pre-filter reads.

    A protocol, not an import of ``PipelineConfig``: the resolution tier
    should not depend on the pipeline tier's concrete config class.
    """

    vlm_heuristic_filter_enabled: bool
    vlm_heuristic_min_aspect_ratio: float
    vlm_heuristic_max_thin_edge: int
    vlm_heuristic_max_analysis_edge: int
    vlm_heuristic_max_analysis_pixels: int
    vlm_heuristic_uniform_stddev: float
    vlm_heuristic_near_blank_stddev: float
    vlm_heuristic_near_blank_edge_density: float
    vlm_heuristic_transparency_ratio: float
    vlm_heuristic_min_confidence: float


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

    @classmethod
    def from_pipeline(cls, config: HeuristicConfigSource) -> HeuristicConfig:
        """Copy the pipeline config's pre-filter knobs into one config."""
        return cls(
            enabled=config.vlm_heuristic_filter_enabled,
            min_aspect_ratio=config.vlm_heuristic_min_aspect_ratio,
            max_thin_edge=config.vlm_heuristic_max_thin_edge,
            max_analysis_edge=config.vlm_heuristic_max_analysis_edge,
            max_analysis_pixels=config.vlm_heuristic_max_analysis_pixels,
            uniform_stddev=config.vlm_heuristic_uniform_stddev,
            near_blank_stddev=config.vlm_heuristic_near_blank_stddev,
            near_blank_edge_density=config.vlm_heuristic_near_blank_edge_density,
            transparency_ratio=config.vlm_heuristic_transparency_ratio,
            min_confidence=config.vlm_heuristic_min_confidence,
        )


@dataclass(frozen=True, slots=True)
class ImageObservations:
    """Measured facts about one image, shared by the filter and the prompts.

    ``sampled`` says whether the bounded pixel pass ran. ``note`` explains
    why it did not when it did not. The fields are deliberately raw: the
    prompt renderer decides how to phrase them, the filter decides whether
    they cross a threshold.
    """

    readable: bool
    width: int = 0
    height: int = 0
    aspect: float = 0.0
    sampled: bool = False
    line_like: bool = False
    grayscale_stddev: float | None = None
    edge_density: float | None = None
    transparency_ratio: float | None = None
    note: str = ""


def assess_image(path: str | Path, config: HeuristicConfig) -> HeuristicVerdict:
    """Classify one image as ``KEEP``, ``DROP``, or ``UNSURE``.

    Never raises: an unreadable or pathological image is ``UNSURE`` so the
    caller can fall through to the existing gates and the VLM.
    """
    return verdict_from_observations(observe_image(path, config), config)


def observe_image(path: str | Path, config: HeuristicConfig) -> ImageObservations:
    """Measure one image once, within the same bounded budget as the filter.

    Returns raw facts only. A caller that needs a verdict calls
    :func:`verdict_from_observations`; a caller that needs prompt text calls
    :func:`render_visual_hints`. Sharing the observation is what keeps the
    resolve stage's hints and the enricher's decision from decoding twice.
    """
    image_path = Path(path)
    try:
        width, height = _header_size(image_path)
    except Exception as exc:
        logger.debug("Image heuristics could not read %s: %s — unsure", image_path, exc)
        return ImageObservations(readable=False, note=f"unreadable header: {exc}")
    if width <= 0 or height <= 0:
        return ImageObservations(readable=False, note=f"empty dimensions ({width}×{height})")

    short_edge = min(width, height)
    long_edge = max(width, height)
    aspect = long_edge / short_edge if short_edge else 0.0
    if aspect >= config.min_aspect_ratio and short_edge <= config.max_thin_edge:
        return ImageObservations(
            readable=True,
            width=width,
            height=height,
            aspect=aspect,
            line_like=True,
            note="line-like proportions",
        )
    if width * height > config.max_analysis_pixels:
        return ImageObservations(
            readable=True,
            width=width,
            height=height,
            aspect=aspect,
            note=f"analysis skipped: {width}×{height} exceeds {config.max_analysis_pixels}px",
        )
    try:
        stddev, edge_density, transparency_ratio = _pixel_stats(image_path, config)
    except Exception as exc:
        logger.debug("Image heuristics could not analyse %s: %s — unsure", image_path, exc)
        return ImageObservations(
            readable=True,
            width=width,
            height=height,
            aspect=aspect,
            note=f"unreadable raster: {exc}",
        )
    return ImageObservations(
        readable=True,
        width=width,
        height=height,
        aspect=aspect,
        sampled=True,
        grayscale_stddev=stddev,
        edge_density=edge_density,
        transparency_ratio=transparency_ratio,
    )


def verdict_from_observations(observations: ImageObservations, config: HeuristicConfig) -> HeuristicVerdict:
    """Turn measured facts into the conservative KEEP/DROP/UNSURE verdict."""
    if not observations.readable:
        return HeuristicVerdict(ImageRelevance.UNSURE, observations.note or "unreadable", 0.0, "structure")

    structural = _structural_verdict(observations.width, observations.height, config)
    if structural is not None:
        return structural
    if not observations.sampled:
        stage = "pixels" if observations.note.startswith("unreadable raster") else "structure"
        return HeuristicVerdict(ImageRelevance.UNSURE, observations.note or "analysis skipped", 0.0, stage)
    return _pixel_verdict(observations, config)


def render_visual_hints(observations: ImageObservations, config: HeuristicConfig) -> str:
    """Render measured facts as one compact prompt clause.

    Facts only, never instructions: the resolve stage puts this text into
    the surroundings block the decision model and the caption model read.
    """
    if not observations.readable:
        return "header unreadable."
    facts = [f"{observations.width}×{observations.height} px", f"aspect {observations.aspect:.2f}:1"]
    if observations.line_like:
        facts.append("line-like proportions")
    elif observations.sampled:
        stddev = observations.grayscale_stddev or 0.0
        edge_density = observations.edge_density or 0.0
        transparency = observations.transparency_ratio or 0.0
        facts.append(f"downscaled sample stddev {stddev:.2f}")
        facts.append(f"edge density {edge_density:.4f}")
        if transparency >= 0.01:
            facts.append(f"transparency {transparency:.1%}")
        if stddev <= config.uniform_stddev:
            facts.append("near-uniform fill")
        elif stddev <= config.near_blank_stddev and edge_density <= config.near_blank_edge_density:
            facts.append("near-blank frame")
    elif observations.note:
        facts.append(observations.note)
    return "; ".join(facts) + "."


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


def _pixel_stats(path: Path, config: HeuristicConfig) -> tuple[float, float, float]:
    """Decode one bounded sample and return (stddev, edge density, transparency).

    JPEG sources use ``draft`` so the decoder can reduce before allocating
    the full frame; every source is thumbnailed to ``max_analysis_edge``.
    """
    with Image.open(path) as opened:
        if opened.format == "JPEG":
            opened.draft("RGB", (config.max_analysis_edge, config.max_analysis_edge))
        working = opened.convert("RGBA")
    working.thumbnail((config.max_analysis_edge, config.max_analysis_edge), Image.Resampling.BILINEAR)
    flattened = _flatten(working)
    stddev = float(ImageStat.Stat(flattened).stddev[0])
    return stddev, _edge_density(flattened), _transparent_ratio(working)


def _pixel_verdict(observations: ImageObservations, config: HeuristicConfig) -> HeuristicVerdict:
    """Stage B: bounded statistics on the downscaled sample."""
    stddev = observations.grayscale_stddev or 0.0
    edge_density = observations.edge_density or 0.0
    transparency_ratio = observations.transparency_ratio or 0.0

    if stddev <= config.uniform_stddev:
        return HeuristicVerdict(
            ImageRelevance.DROP,
            f"near-uniform fill (stddev {stddev:.2f})",
            0.97,
            "pixels",
        )

    if stddev <= config.near_blank_stddev and edge_density <= config.near_blank_edge_density:
        return HeuristicVerdict(
            ImageRelevance.DROP,
            f"near-blank frame (stddev {stddev:.2f}, edge density {edge_density:.4f})",
            0.9,
            "pixels",
        )

    if transparency_ratio >= config.transparency_ratio:
        return HeuristicVerdict(
            ImageRelevance.UNSURE,
            f"transparency-dominant canvas ({transparency_ratio:.1%}) — possible diagram",
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
