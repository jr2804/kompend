"""Tests for KNOX Tier 3 resolvers and enrichers.

VLMEnricher now uses VlmAgent instead of LLMClient.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from knox.agents.vlm import VlmAgent
from knox.core.models import Chunk, ElementType, ResolvedChunk
from knox.resolution import VLMEnricher


def _text_chunk(text: str = "Apple Inc. was founded by Steve Jobs in 1976.") -> Chunk:
    return Chunk(id="chunk_001", text=text, element_types=[ElementType.TEXT], source_elements=[0])


def _image_chunk(image_path: str | None = "/tmp/test.png") -> Chunk:  # noqa: S108
    return Chunk(
        id="chunk_img_001",
        text="Figure showing architecture.",
        element_types=[ElementType.IMAGE],
        source_elements=[2],
        metadata={"image_path": image_path} if image_path else {},
    )


def _make_agent() -> VlmAgent:
    """Build a VlmAgent with a test model."""
    return VlmAgent("ollama/test", timeout=120)


class TestVLMEnricher:
    def test_non_image_chunk_skipped(self) -> None:
        """Text chunks bypass the VLM call entirely."""
        agent = _make_agent()
        enricher = VLMEnricher(agent=agent)
        resolved = enricher.enrich(ResolvedChunk(chunk=_text_chunk("just text")))
        for entity in resolved.entities:
            assert entity.get("type") != "image_description"

    def test_image_chunk_placeholder_when_no_path(self) -> None:
        """Image chunk without a file path produces a placeholder entity."""
        agent = _make_agent()
        enricher = VLMEnricher(agent=agent)
        chunk = _image_chunk(image_path=None)
        resolved = enricher.enrich(ResolvedChunk(chunk=chunk))
        descs = [e for e in resolved.entities if e.get("type") == "image_description"]
        assert len(descs) == 1
        assert "no image path" in descs[0]["context"].lower()

    def test_image_chunk_missing_file_records_error(self, tmp_path: Path) -> None:
        """Image path pointing to a non-existent file → filtered entity, no LLM call."""
        agent = _make_agent()
        enricher = VLMEnricher(agent=agent, min_image_bytes=0, min_image_dimension=0)
        chunk = _image_chunk(image_path=str(tmp_path / "missing.png"))
        resolved = enricher.enrich(ResolvedChunk(chunk=chunk))
        descs = [e for e in resolved.entities if e.get("type") == "image_description"]
        assert len(descs) == 1
        assert "missing" in descs[0]["context"].lower()

    def test_caption_generated_when_agent_succeeds(self, tmp_path: Path) -> None:
        png_path = tmp_path / "test.png"
        png_path.write_bytes(b"PNG mock data")

        agent = _make_agent()
        enricher = VLMEnricher(agent=agent, min_image_bytes=0, min_image_dimension=0)
        with patch.object(agent, "run", return_value="A test image caption"):
            resolved = enricher.enrich(ResolvedChunk(chunk=_image_chunk(image_path=str(png_path))))
        descs = [e for e in resolved.entities if e.get("type") == "image_description"]
        assert len(descs) == 1
        assert descs[0]["context"] == "A test image caption"
        assert resolved.metadata.get("vlm_enriched") is True

    def test_caption_returns_none_records_error(self, tmp_path: Path) -> None:
        png_path = tmp_path / "test.png"
        png_path.write_bytes(b"PNG mock data")

        agent = _make_agent()
        enricher = VLMEnricher(agent=agent, min_image_bytes=0, min_image_dimension=0)
        with patch.object(agent, "run", side_effect=Exception("VLM failed")):
            resolved = enricher.enrich(ResolvedChunk(chunk=_image_chunk(image_path=str(png_path))))
        descs = [e for e in resolved.entities if e.get("type") == "image_description"]
        assert len(descs) == 1
        assert "failed" in descs[0]["context"].lower()

    def test_pre_existing_description_bypasses_vlm(self) -> None:
        """A figure already carrying a parser/backend description skips the VLM call."""
        agent = _make_agent()
        enricher = VLMEnricher(agent=agent)
        chunk = _image_chunk()
        chunk.metadata["description"] = "A diagram of the pipeline."
        resolved = enricher.enrich(ResolvedChunk(chunk=chunk))
        descs = [e for e in resolved.entities if e.get("type") == "image_description"]
        assert len(descs) == 1
        assert descs[0]["context"] == "A diagram of the pipeline."
        assert descs[0]["name"] == "parser_caption_chunk_img_001"
        assert resolved.metadata.get("figure_enriched") is True

    def test_pre_filter_skips_tiny_image_no_vlm_call(self, tmp_path: Path) -> None:
        """Image below byte threshold is filtered before any VLM call."""
        tiny = tmp_path / "tiny.png"
        Image.new("RGB", (8, 8)).save(tiny)
        agent = _make_agent()
        enricher = VLMEnricher(agent=agent)  # default thresholds
        resolved = enricher.enrich(ResolvedChunk(chunk=_image_chunk(image_path=str(tiny))))
        descs = [e for e in resolved.entities if e.get("type") == "image_description"]
        assert len(descs) == 1
        assert "filtered" in descs[0]["context"].lower()
        assert resolved.metadata.get("vlm_filtered") is True

    def test_pre_filter_allows_large_image(self, tmp_path: Path) -> None:
        """Image above thresholds reaches the VLM."""
        big = tmp_path / "big.png"
        img = Image.new("RGB", (200, 200))
        img.putdata([(x ^ y, (x * y) % 255, (x + y) % 255) for y in range(200) for x in range(200)])
        img.save(big)
        assert big.stat().st_size >= 2048, f"test image too small: {big.stat().st_size}"
        agent = _make_agent()
        enricher = VLMEnricher(agent=agent)  # default thresholds
        with patch.object(agent, "run", return_value="A red square."):
            resolved = enricher.enrich(ResolvedChunk(chunk=_image_chunk(image_path=str(big))))
        descs = [e for e in resolved.entities if e.get("type") == "image_description"]
        assert len(descs) == 1
        assert descs[0]["context"] == "A red square."
        assert resolved.metadata.get("vlm_filtered") is None


class TestVLMEnricherIntegration:
    """Sanity checks against the unified pydantic-ai agent path (mocked)."""

    def test_agent_run_sync_invoked_for_image(self, tmp_path: Path) -> None:
        """End-to-end: VlmAgent routes through the unified pydantic-ai agent
        path (build_model + KnoxModel), not a direct litellm.completion call.
        """
        png_path = tmp_path / "img.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)

        agent = _make_agent()
        enricher = VLMEnricher(agent=agent, min_image_bytes=0, min_image_dimension=0)

        mock_result = MagicMock()
        mock_result.output = "caption text"
        mock_ai_agent = MagicMock()
        mock_ai_agent.run_sync.return_value = mock_result

        with patch.object(VlmAgent, "_build_agent", return_value=mock_ai_agent):
            resolved = enricher.enrich(
                ResolvedChunk(chunk=_image_chunk(image_path=str(png_path)))
            )

        mock_ai_agent.run_sync.assert_called_once()
        # The user prompt passed to run_sync is a sequence: [text, BinaryContent].
        call = mock_ai_agent.run_sync.call_args
        user_prompt = call.args[0] if call.args else call.kwargs["user_prompt"]
        assert isinstance(user_prompt, list)
        assert len(user_prompt) == 2  # instruction text + image part
        descs = [e for e in resolved.entities if e.get("type") == "image_description"]
        assert descs[0]["context"] == "caption text"
