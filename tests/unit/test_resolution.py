"""Tests for KNOX Tier 3 resolvers and enrichers.

VLMEnricher routes through the shared :class:`knox.llm.LLMClient`.
These tests inject a client whose ``try_complete`` is patched so the
tests stay fast and deterministic.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knox.core.models import Chunk, ElementType, ResolvedChunk
from knox.llm.client import LLMClient, LLMResult
from knox.llm.failover import CallDiagnostics
from knox.resolution import VLMEnricher
from knox.resolution.chain import ResolverChain
from knox.resolution.llm_resolver import LLMResolver


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


class TestLLMResolver:
    """LLMResolver extracts entities and typed relations."""

    def test_empty_text_returns_resolved_unchanged(self) -> None:
        client = _make_client()
        resolver = LLMResolver(client=client)
        chunk = _text_chunk("")
        resolved = resolver.resolve(chunk)
        assert resolved.chunk == chunk
        assert resolved.entities == []
        assert resolved.relations == []
        client.try_complete.assert_not_called()  # type: ignore[attr-defined]

    def test_entities_and_relations_parsed(self) -> None:
        result = _llm_result(
            {
                "entities": [
                    {"name": "KNOX", "type": "project", "context": "knowledge base"},
                    {"name": "WiCER", "type": "method", "context": "compiler"},
                ],
                "relations": [
                    {"source": "KNOX", "target": "WiCER", "type": "uses", "confidence": 0.9},
                ],
            }
        )
        client = _make_client(try_complete_return=result)
        resolver = LLMResolver(client=client, confidence_threshold=0.0)
        resolved = resolver.resolve(_text_chunk("KNOX uses WiCER."))
        assert len(resolved.entities) == 2
        assert resolved.relations[0]["type"] == "uses"
        assert resolved.relations[0]["confidence"] == pytest.approx(0.9)

    def test_relations_filtered_by_confidence(self) -> None:
        result = _llm_result(
            {
                "entities": [{"name": "A", "type": "x"}, {"name": "B", "type": "y"}],
                "relations": [
                    {"source": "A", "target": "B", "type": "uses", "confidence": 0.9},
                    {"source": "A", "target": "B", "type": "related", "confidence": 0.3},
                ],
            }
        )
        client = _make_client(try_complete_return=result)
        resolver = LLMResolver(client=client, confidence_threshold=0.5)
        resolved = resolver.resolve(_text_chunk("A uses B."))
        assert len(resolved.relations) == 1
        assert resolved.relations[0]["type"] == "uses"

    def test_unknown_relation_types_dropped(self) -> None:
        result = _llm_result(
            {
                "entities": [{"name": "A", "type": "x"}, {"name": "B", "type": "y"}],
                "relations": [
                    {"source": "A", "target": "B", "type": "invented", "confidence": 0.9},
                ],
            }
        )
        client = _make_client(try_complete_return=result)
        resolver = LLMResolver(client=client)
        resolved = resolver.resolve(_text_chunk("A invented B."))
        assert resolved.relations == []

    def test_resolver_chain_accumulates_results(self) -> None:
        """A resolver chain preserves LLMResolver output when VLM runs after it."""
        text_chunk = _text_chunk("KNOX uses WiCER.")

        relation_result = _llm_result(
            {
                "entities": [
                    {"name": "KNOX", "type": "project"},
                    {"name": "WiCER", "type": "method"},
                ],
                "relations": [
                    {"source": "KNOX", "target": "WiCER", "type": "uses", "confidence": 0.8},
                ],
            }
        )
        llm_client = _make_client(try_complete_return=relation_result)
        vlm_client = _make_client()

        chain = ResolverChain(
            [
                LLMResolver(client=llm_client, confidence_threshold=0.0),
                VLMEnricher(client=vlm_client),
            ]
        )
        resolved = chain.resolve(text_chunk)
        # Relations from LLMResolver survive the subsequent VLMEnricher pass.
        assert len(resolved.relations) == 1
        assert resolved.relations[0]["type"] == "uses"
