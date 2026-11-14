"""Tests for the bounded image pre-filter ahead of the VLM path."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from PIL import Image, ImageDraw

from kompend.core.models import Chunk, ElementType, MetadataField, ResolvedChunk
from kompend.resolution.image_heuristics import (
    HeuristicConfig,
    ImageRelevance,
    assess_image,
    observe_image,
    render_visual_hints,
)
from kompend.resolution.vlm_enricher import VLMEnricher


class TestAssessImage:
    def test_thin_line_is_dropped_by_structure(self, tmp_path: Path) -> None:
        verdict = assess_image(_line(tmp_path / "rule.png"), _config())
        assert verdict.decision is ImageRelevance.DROP
        assert verdict.stage == "structure"
        assert "line-like" in verdict.reason
        assert verdict.confidence >= 0.9

    def test_solid_fill_is_dropped_by_pixels(self, tmp_path: Path) -> None:
        verdict = assess_image(_solid(tmp_path / "solid.png"), _config())
        assert verdict.decision is ImageRelevance.DROP
        assert verdict.stage == "pixels"
        assert "near-uniform" in verdict.reason

    def test_blank_page_is_dropped_by_pixels(self, tmp_path: Path) -> None:
        verdict = assess_image(_solid(tmp_path / "blank.png", color="white"), _config())
        assert verdict.decision is ImageRelevance.DROP
        assert "near-uniform" in verdict.reason

    def test_chart_survives(self, tmp_path: Path) -> None:
        verdict = assess_image(_chart(tmp_path / "chart.png"), _config())
        assert verdict.decision is not ImageRelevance.DROP

    def test_transparent_canvas_is_unsure_not_dropped(self, tmp_path: Path) -> None:
        verdict = assess_image(_transparent_with_mark(tmp_path / "transparent.png"), _config())
        assert verdict.decision is ImageRelevance.UNSURE
        assert "transparency-dominant" in verdict.reason

    def test_oversized_analysis_is_skipped_safely(self, tmp_path: Path) -> None:
        verdict = assess_image(_solid(tmp_path / "large.png", size=(200, 200)), _config(max_analysis_pixels=100))
        assert verdict.decision is ImageRelevance.UNSURE
        assert "analysis skipped" in verdict.reason

    def test_unreadable_input_is_unsure(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not an image")
        verdict = assess_image(broken, _config())
        assert verdict.decision is ImageRelevance.UNSURE
        assert "unreadable header" in verdict.reason


class TestVLMEnricherHeuristics:
    def test_solid_figure_is_filtered_before_the_vlm(self, tmp_path: Path) -> None:
        path = _solid(tmp_path / "solid.png")
        agent = _FakeAgent()
        enricher = VLMEnricher(
            agent=agent,  # ty: ignore[invalid-argument-type]
            min_image_bytes=0,
            min_image_dimension=0,
            heuristics=_config(),
        )
        resolved = enricher.enrich(ResolvedChunk(chunk=_image_chunk(str(path))))
        agent.assert_not_called()
        statuses = [entity for entity in resolved.entities if entity.get("type") == "image_caption_status"]
        assert len(statuses) == 1
        assert statuses[0]["name"] == "vlm_filtered"
        assert "near-uniform" in statuses[0]["context"]
        assert resolved.metadata["vlm_heuristic"] == "drop"
        assert "near-uniform" in resolved.metadata["vlm_heuristic_reason"]
        assert not enricher.failures.failures

    def test_line_separator_is_filtered_before_the_vlm(self, tmp_path: Path) -> None:
        path = _line(tmp_path / "rule.png")
        agent = _FakeAgent()
        enricher = VLMEnricher(
            agent=agent,  # ty: ignore[invalid-argument-type]
            min_image_bytes=0,
            min_image_dimension=0,
            heuristics=_config(),
        )
        resolved = enricher.enrich(ResolvedChunk(chunk=_image_chunk(str(path))))
        agent.assert_not_called()
        statuses = [entity for entity in resolved.entities if entity.get("type") == "image_caption_status"]
        assert "line-like" in statuses[0]["context"]

    def test_informative_figure_reaches_the_vlm(self, tmp_path: Path) -> None:
        path = _chart(tmp_path / "chart.png")
        agent = _FakeAgent(caption="A bar chart.")
        enricher = VLMEnricher(
            agent=agent,  # ty: ignore[invalid-argument-type]
            min_image_bytes=0,
            min_image_dimension=0,
            heuristics=_config(),
        )
        resolved = enricher.enrich(ResolvedChunk(chunk=_image_chunk(str(path))))
        agent.assert_called_once()
        captions = [entity for entity in resolved.entities if entity.get("type") == "image_description"]
        assert captions[0]["context"] == "A bar chart."
        assert resolved.metadata.get("vlm_heuristic") is None

    def test_disabled_gate_reproduces_the_old_behaviour(self, tmp_path: Path) -> None:
        path = _solid(tmp_path / "solid.png")
        agent = _FakeAgent(caption="A grey square.")
        enricher = VLMEnricher(
            agent=agent,  # ty: ignore[invalid-argument-type]
            min_image_bytes=0,
            min_image_dimension=0,
            heuristics=_config(enabled=False),
        )
        resolved = enricher.enrich(ResolvedChunk(chunk=_image_chunk(str(path))))
        agent.assert_called_once()
        captions = [entity for entity in resolved.entities if entity.get("type") == "image_description"]
        assert captions[0]["context"] == "A grey square."

    def test_sub_threshold_confidence_falls_through(self, tmp_path: Path) -> None:
        path = _line(tmp_path / "rule.png")
        agent = _FakeAgent(caption="A horizontal rule.")
        enricher = VLMEnricher(
            agent=agent,  # ty: ignore[invalid-argument-type]
            min_image_bytes=0,
            min_image_dimension=0,
            heuristics=_config(min_confidence=0.99),
        )
        resolved = enricher.enrich(ResolvedChunk(chunk=_image_chunk(str(path))))
        agent.assert_called_once()
        captions = [entity for entity in resolved.entities if entity.get("type") == "image_description"]
        assert captions[0]["context"] == "A horizontal rule."


class TestImageObservations:
    def test_a_sampled_image_reports_its_measured_facts(self, tmp_path: Path) -> None:
        observations = observe_image(_solid(tmp_path / "solid.png"), _config())

        assert observations.readable
        assert observations.sampled
        assert (observations.width, observations.height) == (400, 300)
        assert observations.grayscale_stddev == 0.0
        assert observations.transparency_ratio == 0.0

    def test_a_line_is_observed_without_a_pixel_sample(self, tmp_path: Path) -> None:
        observations = observe_image(_line(tmp_path / "rule.png"), _config())

        assert observations.line_like
        assert not observations.sampled
        assert "line-like proportions" in render_visual_hints(observations, _config())

    def test_an_oversized_image_is_not_sampled(self, tmp_path: Path) -> None:
        observations = observe_image(
            _solid(tmp_path / "large.png", size=(200, 200)),
            _config(max_analysis_pixels=100),
        )

        assert not observations.sampled
        assert "analysis skipped" in observations.note

    def test_hints_are_facts_not_a_verdict(self, tmp_path: Path) -> None:
        hints = render_visual_hints(observe_image(_chart(tmp_path / "chart.png"), _config()), _config())

        assert "400×300 px" in hints
        assert "edge density" in hints
        assert "near-uniform" not in hints
        assert "decorative" not in hints


class TestPrecomputedHeuristicVerdict:
    def test_the_enricher_reuses_the_resolve_stage_measurement(self, tmp_path: Path) -> None:
        path = _solid(tmp_path / "solid.png")
        agent = _FakeAgent()
        enricher = VLMEnricher(
            agent=agent,  # ty: ignore[invalid-argument-type]
            min_image_bytes=0,
            min_image_dimension=0,
            heuristics=_config(),
        )
        chunk = _image_chunk(str(path))
        chunk.metadata[MetadataField.VLM_CONTEXT] = {
            "visual_hints": "400×300 px; aspect 1.33:1; near-uniform fill.",
            "heuristic_decision": "drop",
            "heuristic_reason": "near-uniform fill (stddev 0.00)",
            "heuristic_confidence": "0.97",
            "heuristic_stage": "pixels",
        }
        with patch("kompend.resolution.vlm_enricher.assess_image") as assess:
            resolved = enricher.enrich(ResolvedChunk(chunk=chunk))

        assess.assert_not_called()
        agent.assert_not_called()
        assert resolved.metadata["vlm_heuristic"] == "drop"
        assert "near-uniform" in resolved.metadata["vlm_heuristic_reason"]


class _FakeAgent:
    """Minimal stand-in for ``VlmAgent`` — records calls, returns a caption."""

    def __init__(self, caption: str = "A caption.") -> None:
        self._caption = caption
        self.calls = 0

    def run(self, deps: object) -> str:
        self.calls += 1
        return self._caption

    def assert_called_once(self) -> None:
        assert self.calls == 1, f"expected one VLM call, got {self.calls}"

    def assert_not_called(self) -> None:
        assert self.calls == 0, f"expected no VLM call, got {self.calls}"


def _config(**overrides: Any) -> HeuristicConfig:
    base = HeuristicConfig(
        enabled=True,
        min_aspect_ratio=20.0,
        max_thin_edge=96,
        max_analysis_edge=320,
        max_analysis_pixels=2_000_000,
        uniform_stddev=1.0,
        near_blank_stddev=4.0,
        near_blank_edge_density=0.001,
        transparency_ratio=0.995,
        min_confidence=0.9,
    )
    return replace(base, **overrides)


def _solid(path: Path, *, size: tuple[int, int] = (400, 300), color: str = "#808080") -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def _line(path: Path, *, size: tuple[int, int] = (2000, 80)) -> Path:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.line((0, size[1] // 2, size[0], size[1] // 2), fill="black", width=1)
    image.save(path)
    return path


def _chart(path: Path, *, size: tuple[int, int] = (400, 300)) -> Path:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.line((40, 20, 40, 260), fill="black", width=2)
    draw.line((40, 260, 380, 260), fill="black", width=2)
    bars = [("#3366cc", 60), ("#dc3912", 110), ("#ff9900", 80), ("#109618", 150), ("#990099", 95)]
    for index, (color, height) in enumerate(bars):
        left = 60 + index * 60
        draw.rectangle((left, 260 - height, left + 40, 260), fill=color, outline="black")
    image.save(path)
    return path


def _transparent_with_mark(path: Path, *, size: tuple[int, int] = (400, 300)) -> Path:
    image = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 40, 40), fill=(0, 0, 0, 255))
    image.save(path)
    return path


def _image_chunk(image_file: str) -> Chunk:
    return Chunk(
        id="chunk_img_001",
        text="Figure showing architecture.",
        element_types=[ElementType.IMAGE],
        source_elements=[2],
        metadata={"image_file": image_file},
    )
