"""Tests for KNOX Tier 3 resolvers and enrichers.

VLMEnricher routes through the shared :class:`knox.llm.LLMClient`.
These tests inject a client whose ``try_complete`` is patched so the
tests stay fast and deterministic.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from knox.core.models import Chunk, ElementType, ResolvedChunk
from knox.llm.client import LLMClient, LLMResult
from knox.llm.failover import CallDiagnostics
from knox.resolution import VLMEnricher


def _text_chunk(text: str = "Apple Inc. was founded by Steve Jobs in 1976.") -> Chunk:
    return Chunk(id="chunk_001", text=text, element_types=[ElementType.TEXT], source_elements=[0])


def _atomic_chunk(elem_type: ElementType = ElementType.TABLE) -> Chunk:
    return Chunk(id="chunk_atomic", text="", element_types=[elem_type], source_elements=[1])


def _image_chunk(image_path: str | None = "/tmp/test.png") -> Chunk:  # noqa: S108
    return Chunk(
        id="chunk_img_001",
        text="Figure showing architecture.",
        element_types=[ElementType.IMAGE],
        source_elements=[2],
        metadata={"image_path": image_path} if image_path else {},
    )


def _make_client(try_complete_return: LLMResult | None = None) -> LLMClient:
    """Build an LLMClient with try_complete patched to return a fixed value."""
    client = LLMClient(default_model="ollama/test", max_retries_per_model=0)
    client.try_complete = MagicMock(return_value=try_complete_return)  # type: ignore[method-assign]
    return client


def _llm_result(data: dict, model: str = "ollama/test") -> LLMResult:
    return LLMResult(
        content="",
        json_data=data,
        diagnostics=CallDiagnostics(model_used=model, attempts=1, total_latency_s=0.01),
    )


class TestVLMEnricher:
    def test_non_image_chunk_skipped(self) -> None:
        """Text chunks bypass the VLM call entirely."""
        client = _make_client()
        enricher = VLMEnricher(client=client)
        resolved = enricher.enrich(ResolvedChunk(chunk=_text_chunk("just text")))
        for entity in resolved.entities:
            assert entity.get("type") != "image_description"
        client.try_complete.assert_not_called()  # type: ignore[attr-defined]

    def test_image_chunk_placeholder_when_no_path(self) -> None:
        """Image chunk without a file path produces a placeholder entity."""
        client = _make_client()
        enricher = VLMEnricher(client=client)
        chunk = _image_chunk(image_path=None)
        resolved = enricher.enrich(ResolvedChunk(chunk=chunk))
        descs = [e for e in resolved.entities if e.get("type") == "image_description"]
        assert len(descs) == 1
        assert "no image path" in descs[0]["context"].lower()

    def test_image_chunk_missing_file_records_error(self, tmp_path: Path) -> None:
        """Image path pointing to a non-existent file → error entity, no LLM call."""
        client = _make_client()
        enricher = VLMEnricher(client=client)
        chunk = _image_chunk(image_path=str(tmp_path / "missing.png"))
        resolved = enricher.enrich(ResolvedChunk(chunk=chunk))
        descs = [e for e in resolved.entities if e.get("type") == "image_description"]
        assert len(descs) == 1
        assert "not found" in descs[0]["context"].lower()
        client.try_complete.assert_not_called()  # type: ignore[attr-defined]

    def test_caption_generated_when_client_succeeds(self, tmp_path: Path) -> None:
        png_path = tmp_path / "test.png"
        png_path.write_bytes(b"PNG mock data")

        result = LLMResult(
            content="A test image caption",
            json_data=None,
            diagnostics=CallDiagnostics(model_used="ollama/test", attempts=1, total_latency_s=0.01),
        )
        client = _make_client(try_complete_return=result)
        enricher = VLMEnricher(client=client)
        resolved = enricher.enrich(ResolvedChunk(chunk=_image_chunk(image_path=str(png_path))))
        descs = [e for e in resolved.entities if e.get("type") == "image_description"]
        assert len(descs) == 1
        assert descs[0]["context"] == "A test image caption"
        assert resolved.metadata.get("vlm_enriched") is True

    def test_caption_returns_none_records_error(self, tmp_path: Path) -> None:
        png_path = tmp_path / "test.png"
        png_path.write_bytes(b"PNG mock data")

        client = _make_client(try_complete_return=None)
        enricher = VLMEnricher(client=client)
        resolved = enricher.enrich(ResolvedChunk(chunk=_image_chunk(image_path=str(png_path))))
        descs = [e for e in resolved.entities if e.get("type") == "image_description"]
        assert len(descs) == 1
        assert "failed" in descs[0]["context"].lower()

    def test_role_forwarded_to_client(self, tmp_path: Path) -> None:
        png_path = tmp_path / "test.png"
        png_path.write_bytes(b"PNG mock data")

        client = _make_client(try_complete_return=None)
        enricher = VLMEnricher(client=client, role="custom.vlm")
        enricher.enrich(ResolvedChunk(chunk=_image_chunk(image_path=str(png_path))))
        call_kwargs = client.try_complete.call_args.kwargs  # type: ignore[attr-defined]
        assert call_kwargs["role"] == "custom.vlm"


class TestVLMEnricherIntegration:
    """Sanity checks against the real LiteLLM call path (mocked)."""

    def test_litellm_completion_invoked_for_image(self, tmp_path: Path) -> None:
        """End-to-end: real LLMClient + patched litellm.completion."""
        png_path = tmp_path / "img.png"
        png_path.write_bytes(b"PNG")

        client = LLMClient(default_model="ollama/test", max_retries_per_model=0)
        enricher = VLMEnricher(client=client)

        mock_response = MagicMock()
        mock_message = MagicMock(content="caption text")
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.delta = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=2)
        # Streaming support.
        event = MagicMock()
        event.choices = [mock_choice]
        event.usage = MagicMock(prompt_tokens=5, completion_tokens=2)
        mock_response.__iter__ = lambda self: iter([event])
        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            resolved = enricher.enrich(ResolvedChunk(chunk=_image_chunk(image_path=str(png_path))))

        mock_completion.assert_called_once()
        descs = [e for e in resolved.entities if e.get("type") == "image_description"]
        assert descs[0]["context"] == "caption text"
