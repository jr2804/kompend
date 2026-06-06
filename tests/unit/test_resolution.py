"""Tests for KNOX Tier 3 resolvers and enrichers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knox.core.models import Chunk, ElementType, ResolvedChunk
from knox.resolution import LLMResolver, VLMEnricher


# ===========================================================================
#  Helpers
# ===========================================================================

def _text_chunk(text: str = "Apple Inc. was founded by Steve Jobs in 1976.") -> Chunk:
    return Chunk(
        id="chunk_001",
        text=text,
        element_types=[ElementType.TEXT],
        source_elements=[0],
    )


def _atomic_chunk(elem_type: ElementType = ElementType.TABLE) -> Chunk:
    return Chunk(
        id="chunk_atomic",
        text="",
        element_types=[elem_type],
        source_elements=[1],
    )


def _image_chunk(image_path: str | None = "/tmp/test.png") -> Chunk:
    return Chunk(
        id="chunk_img_001",
        text="Figure showing architecture.",
        element_types=[ElementType.IMAGE],
        source_elements=[2],
        metadata={"image_path": image_path} if image_path else {},
    )


# ===========================================================================
#  LLMResolver
# ===========================================================================

class TestLLMResolver:
    """LLMResolver behaviour."""

    def test_text_chunk_returns_resolved_chunk(self) -> None:
        """A text chunk always comes back as a ResolvedChunk (no-op fallback)."""
        resolver = LLMResolver()
        chunk = _text_chunk()
        result = resolver.resolve(chunk)
        assert isinstance(result, ResolvedChunk)
        assert result.chunk is chunk

    def test_text_chunk_no_litellm(self) -> None:
        """When litellm is absent the resolver returns a passthrough ResolvedChunk."""
        with patch.object(LLMResolver, "__init__", return_value=None):
            resolver = LLMResolver.__new__(LLMResolver)
            resolver.model = "llama3.2"
            resolver.provider = "ollama"
            resolver._system_prompt = ""
            resolver._litellm = None
        chunk = _text_chunk()
        result = resolver.resolve(chunk)
        assert isinstance(result, ResolvedChunk)
        assert result.chunk is chunk
        assert result.entities == []

    def test_atomic_table_skipped(self) -> None:
        """Atomic chunks (TABLE) skip LLM and return as-is."""
        resolver = LLMResolver()
        chunk = _atomic_chunk(ElementType.TABLE)
        result = resolver.resolve(chunk)
        assert isinstance(result, ResolvedChunk)
        assert result.chunk is chunk
        assert result.entities == []

    def test_atomic_image_skipped(self) -> None:
        """Atomic chunks (IMAGE) skip LLM and return as-is."""
        resolver = LLMResolver()
        chunk = _atomic_chunk(ElementType.IMAGE)
        result = resolver.resolve(chunk)
        assert isinstance(result, ResolvedChunk)
        assert result.chunk is chunk
        assert result.entities == []

    def test_atomic_formula_skipped(self) -> None:
        """Atomic chunks (FORMULA) skip LLM and return as-is."""
        resolver = LLMResolver()
        chunk = _atomic_chunk(ElementType.FORMULA)
        result = resolver.resolve(chunk)
        assert isinstance(result, ResolvedChunk)
        assert result.entities == []

    def test_atomic_code_skipped(self) -> None:
        """Atomic chunks (CODE) skip LLM and return as-is."""
        resolver = LLMResolver()
        chunk = _atomic_chunk(ElementType.CODE)
        result = resolver.resolve(chunk)
        assert isinstance(result, ResolvedChunk)
        assert result.entities == []

    def test_resolver_never_raises_on_litellm_failure(self) -> None:
        """If litellm *is* available but the call fails, fall back gracefully."""
        resolver = LLMResolver()
        # Force _litellm to a mock that raises
        mock_litellm = MagicMock()
        mock_litellm.completion.side_effect = RuntimeError("LLM down")
        resolver._litellm = mock_litellm

        chunk = _text_chunk()
        result = resolver.resolve(chunk)
        assert isinstance(result, ResolvedChunk)
        assert result.chunk is chunk
        assert result.entities == []
        # Ensure the mock was actually called
        mock_litellm.completion.assert_called_once()

    def test_resolver_with_mocked_litellm_success(self) -> None:
        """With a working litellm mock, entities are extracted."""
        resolver = LLMResolver()
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value.choices = [
            MagicMock(message=MagicMock(content='{"entities": [{"name": "Apple Inc.", "type": "ORG", "context": "company"}], "relations": [], "resolved_references": []}'))
        ]
        resolver._litellm = mock_litellm

        chunk = _text_chunk()
        result = resolver.resolve(chunk)
        assert isinstance(result, ResolvedChunk)
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "Apple Inc."
        assert result.metadata.get("resolved_by") == "ollama/llama3.2"

    def test_resolver_mocked_litellm_bad_json(self) -> None:
        """If the LLM returns unparseable JSON, fall back gracefully."""
        resolver = LLMResolver()
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value.choices = [
            MagicMock(message=MagicMock(content="not valid json"))
        ]
        resolver._litellm = mock_litellm

        chunk = _text_chunk()
        result = resolver.resolve(chunk)
        assert isinstance(result, ResolvedChunk)
        assert result.entities == []

    def test_resolver_mocked_litellm_empty_response(self) -> None:
        """If the LLM returns an empty response body, fall back gracefully."""
        resolver = LLMResolver()
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value.choices = [
            MagicMock(message=MagicMock(content=""))
        ]
        resolver._litellm = mock_litellm

        chunk = _text_chunk()
        result = resolver.resolve(chunk)
        assert isinstance(result, ResolvedChunk)
        assert result.entities == []


# ===========================================================================
#  VLMEnricher
# ===========================================================================

class TestVLMEnricher:
    """VLMEnricher behaviour."""

    def test_no_backend_noop(self) -> None:
        """When no VLM backend is available, enrich is a no-op."""
        enricher = VLMEnricher()
        chunk = _text_chunk()
        resolved = ResolvedChunk(chunk=chunk)
        result = enricher.enrich(resolved)
        assert result is resolved  # same object, unchanged
        assert result.entities == []

    def test_non_image_chunk_skipped(self) -> None:
        """Non-IMAGE chunks are passed through unchanged."""
        enricher = VLMEnricher()
        # Even if backend available, non-image chunk passes through
        enricher._available = True
        enricher._backend = "litellm"
        chunk = _text_chunk()
        resolved = ResolvedChunk(chunk=chunk)
        result = enricher.enrich(resolved)
        assert result.entities == []

    def test_image_chunk_placeholder_when_no_path(self) -> None:
        """Image chunks without an image_path get a placeholder entity."""
        enricher = VLMEnricher()
        enricher._available = True
        enricher._backend = "litellm"
        chunk = _image_chunk(image_path=None)
        resolved = ResolvedChunk(chunk=chunk)
        result = enricher.enrich(resolved)
        assert len(result.entities) == 1
        assert result.entities[0]["type"] == "image_description"

    def test_enrich_with_mock(self) -> None:
        """With a mocked caption generator, IMAGE chunks get an entity."""
        enricher = VLMEnricher()
        enricher._available = True
        enricher._backend = "litellm"

        chunk = _image_chunk(image_path="/tmp/test.png")
        resolved = ResolvedChunk(chunk=chunk)

        # Patch the private caption method
        with patch.object(enricher, "_generate_caption", return_value="A diagram of the system architecture."):
            result = enricher.enrich(resolved)

        assert len(result.entities) == 1
        entity = result.entities[0]
        assert entity["type"] == "image_description"
        assert "system architecture" in entity["context"]
        assert result.metadata.get("vlm_enriched") is True

    def test_enrich_handles_exception_gracefully(self) -> None:
        """If _generate_caption raises, add an error entity instead of crashing."""
        enricher = VLMEnricher()
        enricher._available = True
        enricher._backend = "litellm"

        chunk = _image_chunk(image_path="/tmp/missing.png")
        resolved = ResolvedChunk(chunk=chunk)

        with patch.object(enricher, "_generate_caption", side_effect=FileNotFoundError("not found")):
            result = enricher.enrich(resolved)

        assert len(result.entities) == 1
        assert result.entities[0]["type"] == "image_description"
        assert "failed" in result.entities[0]["context"].lower()

    def test_caption_via_litellm_mocked(self) -> None:
        """_caption_via_litellm works with a mocked litellm completion."""
        enricher = VLMEnricher()
        enricher._available = True
        enricher._backend = "litellm"

        # Create a tiny valid PNG to pass through base64 encoding
        png_path = Path("/tmp/__test_vlm.png")
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fakedata")

        try:
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value.choices = [
                    MagicMock(message=MagicMock(content="A small test image."))
                ]
                caption = enricher._caption_via_litellm(png_path)
                assert caption == "A small test image."
                mock_completion.assert_called_once()
        finally:
            png_path.unlink(missing_ok=True)