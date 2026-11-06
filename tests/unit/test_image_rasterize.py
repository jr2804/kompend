"""Tests for rasterizing vector figures at the model boundary."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from kompend.preprocessing.images import (
    MAX_RASTER_LONG_EDGE,
    MAX_RASTER_PIXELS,
    VectorRasterError,
    is_raster_image,
    rasterise_bytes,
    rasterise_image,
)

_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="{width}" height="{height}" fill="#336699"/></svg>'


class TestIsRasterImage:
    """Which suffixes already carry pixels."""

    def test_png_is_raster(self, tmp_path: Path) -> None:
        assert is_raster_image(tmp_path / "figure.png") is True

    def test_svg_is_not(self, tmp_path: Path) -> None:
        assert is_raster_image(tmp_path / "figure.svg") is False

    def test_svgz_is_not(self, tmp_path: Path) -> None:
        assert is_raster_image(tmp_path / "figure.svgz") is False


class TestRasteriseImage:
    """Rendering a vector figure to a PNG beside it."""

    def test_a_raster_file_is_returned_unchanged(self, tmp_path: Path) -> None:
        png = tmp_path / "figure.png"
        Image.new("RGB", (40, 30), "#336699").save(png)
        assert rasterise_image(png) == png

    def test_a_vector_figure_renders_a_png_beside_it(self, tmp_path: Path) -> None:
        svg = _svg(tmp_path / "table.svg")
        rendered = rasterise_image(svg)
        assert rendered == tmp_path / "table.png"
        assert rendered.is_file()
        with Image.open(rendered) as image:
            assert image.format == "PNG"
            # A vector figure is re-rendered at a size a vision model can read,
            # keeping the aspect its geometry declares.
            assert max(image.size) == MAX_RASTER_LONG_EDGE
            assert abs(image.width / image.height - 200 / 120) < 0.01

    def test_the_render_is_reused_while_the_vector_is_unchanged(self, tmp_path: Path) -> None:
        svg = _svg(tmp_path / "table.svg")
        first = rasterise_image(svg)
        stamp = first.stat().st_mtime_ns
        assert rasterise_image(svg) == first
        assert first.stat().st_mtime_ns == stamp

    def test_a_render_older_than_its_vector_is_repeated(self, tmp_path: Path) -> None:
        svg = _svg(tmp_path / "table.svg")
        rendered = rasterise_image(svg)
        os.utime(rendered, (0, 0))
        rasterise_image(svg)
        assert rendered.stat().st_mtime_ns > 0

    def test_a_non_image_is_a_failure_not_a_silent_skip(self, tmp_path: Path) -> None:
        text = tmp_path / "notes.txt"
        text.write_text("nothing to see", encoding="utf-8")
        with pytest.raises(VectorRasterError):
            rasterise_image(text)

    def test_unparseable_vector_geometry_still_renders(self, tmp_path: Path) -> None:
        """A figure without width/height or viewBox is rendered, not skipped."""
        target = tmp_path / "sparse.svg"
        target.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><rect width="180" height="90" fill="#336699"/></svg>',
            encoding="utf-8",
        )
        rendered = rasterise_image(target)
        assert rendered.is_file()
        with Image.open(rendered) as image:
            assert max(image.size) <= MAX_RASTER_LONG_EDGE

    def test_a_pathological_canvas_stays_inside_the_pixel_budget(self, tmp_path: Path) -> None:
        svg = _svg(tmp_path / "strip.svg", width=1400, height=110600)
        rendered = rasterise_image(svg)
        with Image.open(rendered) as image:
            assert image.width * image.height <= MAX_RASTER_PIXELS
            assert max(image.size) <= MAX_RASTER_LONG_EDGE


class TestRasteriseBytes:
    """The byte-level entry point used by the media embedder."""

    def test_bytes_of_a_vector_figure_are_png(self, tmp_path: Path) -> None:
        svg = _svg(tmp_path / "table.svg")
        data = rasterise_bytes(svg)
        assert data is not None
        assert data.startswith(b"\x89PNG")

    def test_a_missing_figure_is_none(self, tmp_path: Path) -> None:
        assert rasterise_bytes(tmp_path / "gone.png") is None


def _svg(path: Path, width: int = 200, height: int = 120) -> Path:
    """Write a solid-rectangle SVG and return its path."""
    path.write_text(_SVG.format(width=width, height=height), encoding="utf-8")
    return path
