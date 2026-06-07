"""Tests for KNOX Tier 3 resolvers and enrichers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import knox.resolution.llm as llm_mod
from knox.core.models import Chunk, ElementType, ResolvedChunk
from knox.resolution import LLMResolver, VLMEnricher


def _text_chunk(text: str = "Apple Inc. was founded by Steve Jobs in 1976.") -> Chunk:
    return Chunk(id="chunk_001", text=text, element_types=[ElementType.TEXT], source_elements=[0])


def _atomic_chunk(elem_type: ElementType = ElementType.TABLE) -> Chunk:
    return Chunk(id="chunk_atomic", text="", element_types=[elem_type], source_elements=[1])


def _image_chunk(image_path: str | None = "/tmp/test.png") -> Chunk:
    return Chunk(
        id="chunk_img_001", text="Figure showing architecture.",
        element_types=[ElementType.IMAGE], source_elements=[2],
        metadata={"image_path": image_path} if image_path else {},
    )


class TestLLMResolver:

    def test_text_chunk_returns_resolved_chunk(self) -> None:
        resolver = LLMResolver()
        chunk = _text_chunk()
        result = resolver.resolve(chunk)
        assert isinstance(result, ResolvedChunk)
        assert result.chunk is chunk

    def test_text_chunk_no_litellm(self) -> None:
        """When litellm is absent the resolver returns a passthrough."""
        with patch.object(llm_mod, "litellm", None):
            resolver = LLMResolver.__new__(LLMResolver)
            resolver.model = "llama3.2"
            resolver.provider = "ollama"
            resolver._system_prompt = ""
        chunk = _text_chunk()
        result = resolver.resolve(chunk)
        assert isinstance(result, ResolvedChunk)
        assert result.chunk is chunk

    def test_atomic_table_skipped(self) -> None:
        resolver = LLMResolver()
        chunk = _atomic_chunk(ElementType.TABLE)
        result = resolver.resolve(chunk)
        assert result.chunk is chunk
        assert result.entities == []

    def test_atomic_image_skipped(self) -> None:
        resolver = LLMResolver()
        chunk = _atomic_chunk(ElementType.IMAGE)
        result = resolver.resolve(chunk)
        assert result.chunk is chunk
        assert result.entities == []

    def test_atomic_formula_skipped(self) -> None:
        resolver = LLMResolver()
        chunk = _atomic_chunk(ElementType.FORMULA)
        result = resolver.resolve(chunk)
        assert result.chunk is chunk

    def test_atomic_code_skipped(self) -> None:
        resolver = LLMResolver()
        chunk = _atomic_chunk(ElementType.CODE)
        result = resolver.resolve(chunk)
        assert result.chunk is chunk

    def test_resolver_never_raises_on_litellm_failure(self) -> None:
        """If litellm call fails, fall back gracefully."""
        resolver = LLMResolver()
        with patch("litellm.completion", side_effect=RuntimeError("LLM down")) as mock_completion:
            chunk = _text_chunk()
            result = resolver.resolve(chunk)
            assert isinstance(result, ResolvedChunk)
            assert result.entities == []
            mock_completion.assert_called_once()

    def test_resolver_with_mocked_litellm_success(self) -> None:
        """With a working litellm mock, entities are extracted."""
        resolver = LLMResolver()
        mock_choices = [MagicMock(message=MagicMock(
            content='{"entities": [{"name": "Apple Inc.", "type": "ORG", "context": "company"}], "relations": [], "resolved_references": []}'
        ))]
        with patch("litellm.completion", return_value=MagicMock(choices=mock_choices)):
            chunk = _text_chunk()
            result = resolver.resolve(chunk)
            assert len(result.entities) == 1
            assert result.entities[0]["name"] == "Apple Inc."

    def test_resolver_mocked_litellm_bad_json(self) -> None:
        """If LLM returns unparseable JSON, fall back."""
        resolver = LLMResolver()
        mock_choices = [MagicMock(message=MagicMock(content="not valid json"))]
        with patch("litellm.completion", return_value=MagicMock(choices=mock_choices)):
            chunk = _text_chunk()
            result = resolver.resolve(chunk)
            assert result.entities == []

    def test_resolver_mocked_litellm_empty_response(self) -> None:
        """If LLM returns empty content, fall back."""
        resolver = LLMResolver()
        mock_choices = [MagicMock(message=MagicMock(content=""))]
        with patch("litellm.completion", return_value=MagicMock(choices=mock_choices)):
            chunk = _text_chunk()
            result = resolver.resolve(chunk)
            assert result.entities == []


class TestVLMEnricher:

    def test_no_backend_noop(self) -> None:
        """No VLM backend available → no-op."""
        enricher = VLMEnricher()
        resolved = enricher.enrich(ResolvedChunk(chunk=_text_chunk()))
        assert resolved.entities == []

    def test_non_image_chunk_skipped(self) -> None:
        enricher = VLMEnricher()
        enricher._backend = "litellm"
        chunk = _text_chunk("just text")
        resolved = enricher.enrich(ResolvedChunk(chunk=chunk))
        for e in resolved.entities:
            assert e.get("type") != "image_description"

    def test_image_chunk_placeholder_when_no_path(self) -> None:
        """Image chunk without file → placeholder entity."""
        enricher = VLMEnricher()
        enricher._backend = "litellm"
        chunk = _image_chunk(image_path=None)
        resolved = enricher.enrich(ResolvedChunk(chunk=chunk))
        descs = [e for e in resolved.entities if e.get("type") == "image_description"]
        assert len(descs) >= 1

    def test_enrich_with_mock(self) -> None:
        enricher = VLMEnricher()
        enricher._backend = "litellm"
        with patch.object(enricher, "_generate_caption", return_value="A test image"):
            chunk = _image_chunk()
            resolved = enricher.enrich(ResolvedChunk(chunk=chunk))
        descs = [e for e in resolved.entities if e.get("type") == "image_description"]
        assert any("test" in d.get("context", "") for d in descs)

    def test_enrich_handles_exception_gracefully(self) -> None:
        enricher = VLMEnricher()
        enricher._backend = "litellm"
        with patch.object(enricher, "_generate_caption", side_effect=RuntimeError("VLM crash")):
            chunk = _image_chunk()
            resolved = enricher.enrich(ResolvedChunk(chunk=chunk))
        assert isinstance(resolved, ResolvedChunk)

    def test_caption_via_litellm_mocked(self) -> None:
        enricher = VLMEnricher()
        enricher._backend = "litellm"
        png_path = Path("/tmp/test_vlm.png")
        try:
            png_path.write_bytes(b"PNG mock data")
            with patch("litellm.completion") as mock_completion:
                mock_completion.return_value.choices = [
                    MagicMock(message=MagicMock(content="A test image caption"))
                ]
                caption = enricher._caption_via_litellm(png_path)
                assert caption == "A test image caption"
                mock_completion.assert_called_once()
        finally:
            if png_path.exists():
                png_path.unlink()