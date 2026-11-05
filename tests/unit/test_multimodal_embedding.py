"""Multimodal embedding: text + figure into one vector.

Covers the two halves: the Ollama media endpoint (``kompend.rag.multimodal``)
and the store's use of it (``kompend.rag.vector_store``).
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from kompend.core.models import CompiledWiki, WikiNode
from kompend.core.models.enums import MetadataField, NodeType
from kompend.core.utils.exceptions import LLMOutputError
from kompend.rag.embedding_base import EmbeddingProvider
from kompend.rag.embedding_pydantic import PydanticAIEmbeddingProvider
from kompend.rag.multimodal import MediaEmbeddingUnsupported, OllamaMediaEmbedder
from kompend.rag.sqlite_vec_store import SqliteVecWikiStore
from kompend.rag.vector_store import content_hash

PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082")


class _Response:
    """Minimal stand-in for ``httpx.Response``."""

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text
        self.is_error = status_code >= 400

    def json(self) -> Any:
        return json.loads(self.text)


class TestOllamaMediaEmbedder:
    def test_a_media_capable_model_returns_the_vector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_post(monkeypatch, _Response(200, '{"embeddings": [[0.25, 0.75]]}'))

        vector = OllamaMediaEmbedder("embed-model").embed("a figure", PNG)

        assert vector == [0.25, 0.75]

    def test_the_image_is_sent_as_raw_base64(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[dict] = []
        _patch_post(monkeypatch, _Response(200, '{"embeddings": [[0.1]]}'), captured)

        OllamaMediaEmbedder("embed-model").embed("a figure", PNG)

        payload = captured[0]["payload"]
        assert payload["model"] == "embed-model"
        assert payload["input"]["text"] == "a figure"
        assert payload["input"]["image"] == base64.b64encode(PNG).decode()
        assert not payload["input"]["image"].startswith("data:")

    def test_a_model_without_an_image_encoder_is_reported_as_unsupported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_post(monkeypatch, _Response(501, '{"error":"model does not support media embeddings"}'))

        with pytest.raises(MediaEmbeddingUnsupported):
            OllamaMediaEmbedder("embed-model").embed("a figure", PNG)

    def test_a_server_error_raises_an_llm_output_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_post(monkeypatch, _Response(500, "internal error"))

        with pytest.raises(LLMOutputError):
            OllamaMediaEmbedder("embed-model").embed("a figure", PNG)

    def test_a_response_without_a_vector_raises_an_llm_output_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_post(monkeypatch, _Response(200, '{"embeddings": []}'))

        with pytest.raises(LLMOutputError):
            OllamaMediaEmbedder("embed-model").embed("a figure", PNG)

    def test_an_empty_image_is_refused_before_the_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = False

        def fake_post(url: str, *, json: dict, timeout: float) -> _Response:
            nonlocal called
            called = True
            return _Response(200, '{"embeddings": [[0.1]]}')

        monkeypatch.setattr("kompend.rag.multimodal.httpx.post", fake_post)

        with pytest.raises(LLMOutputError):
            OllamaMediaEmbedder("embed-model").embed("a figure", b"")

        assert called is False


class TestProviderMediaModel:
    """The provider addresses Ollama's media endpoint with the bare model name."""

    def test_the_provider_prefix_is_stripped_for_the_media_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def _fake_post(url: str, **kwargs: Any) -> _Response:
            seen["url"] = url
            seen["body"] = kwargs["json"]
            return _Response(200, json.dumps({"embeddings": [[0.1, 0.2]]}))

        monkeypatch.setattr("kompend.rag.embedding_pydantic.build_embedder", lambda _model_string: object())
        monkeypatch.setattr("kompend.rag.multimodal.httpx.post", _fake_post)
        provider = PydanticAIEmbeddingProvider("ollama:embeddinggemma-2:440m")

        provider.embed_media("a figure", PNG)

        assert seen["body"]["model"] == "embeddinggemma-2:440m"
        assert seen["url"].endswith("/api/embed")
        assert "/v1/" not in seen["url"]


class TestMediaEndpoint:
    """Ollama's native API lives at the server root, not under ``/v1``."""

    def test_the_openai_compatible_base_is_stripped(self) -> None:
        embedder = OllamaMediaEmbedder("embed-model", base_url="http://127.0.0.1:11434/v1")

        assert embedder.endpoint == "http://127.0.0.1:11434/api/embed"

    def test_a_server_root_base_is_kept(self) -> None:
        embedder = OllamaMediaEmbedder("embed-model", base_url="http://127.0.0.1:11434/")

        assert embedder.endpoint == "http://127.0.0.1:11434/api/embed"


class _FigureProvider(EmbeddingProvider):
    """Provider that can embed a figure and reports which path it took."""

    def __init__(self, dim: int = 4, *, supports_media: bool = True) -> None:
        self._dim = dim
        self._supports_media = supports_media
        self.media_calls: list[tuple[str, int]] = []

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        return [0.5] * self._dim

    def embed_media(self, text: str, image: bytes) -> list[float]:
        if not self._supports_media:
            raise MediaEmbeddingUnsupported("Embedding model 'stub' does not support media embeddings")
        self.media_calls.append((text, len(image)))
        return [0.9] * self._dim


class TestStoreUsesTheFigure:
    def test_a_node_with_a_figure_is_embedded_with_the_image(self, tmp_path: Path) -> None:
        provider = _FigureProvider()
        store = SqliteVecWikiStore(db_file=tmp_path / "vectors", embedding_provider=provider)

        assert store.build_index(_wiki_with_figure(_figure(tmp_path))) == 1
        store.close()

        assert len(provider.media_calls) == 1
        assert provider.media_calls[0][1] == len(PNG)

    def test_a_model_without_an_image_encoder_indexes_the_text_and_reports_it(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        provider = _FigureProvider(supports_media=False)
        store = SqliteVecWikiStore(db_file=tmp_path / "vectors", embedding_provider=provider)

        with caplog.at_level(logging.WARNING):
            assert store.build_index(_wiki_with_figure(_figure(tmp_path))) == 1
        store.close()

        assert provider.media_calls == []
        assert any("does not support media embeddings" in record.getMessage() for record in caplog.records)

    def test_a_missing_figure_is_reported_and_does_not_stop_the_build(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        provider = _FigureProvider()
        store = SqliteVecWikiStore(db_file=tmp_path / "vectors", embedding_provider=provider)
        wiki = _wiki_with_figure(tmp_path / "gone.png")

        with caplog.at_level(logging.WARNING):
            assert store.build_index(wiki) == 1
        store.close()

        assert provider.media_calls == []
        assert any("figure file is missing" in record.getMessage() for record in caplog.records)


class TestStalenessCoversTheFigure:
    def test_the_fingerprint_changes_when_the_figure_changes(self, tmp_path: Path) -> None:
        figure = _figure(tmp_path)
        node = _wiki_with_figure(figure).nodes[0]

        before = content_hash(node)
        figure.write_bytes(PNG + b"changed")
        after = content_hash(node)

        assert before != after

    def test_the_fingerprint_survives_a_figure_that_cannot_be_read(self, tmp_path: Path) -> None:
        figure = _figure(tmp_path)
        node = _wiki_with_figure(figure).nodes[0]

        before = content_hash(node)
        figure.unlink()
        after = content_hash(node)

        assert before != after


def _patch_post(monkeypatch: pytest.MonkeyPatch, response: _Response, captured: list[dict] | None = None) -> None:
    def fake_post(url: str, *, json: dict, timeout: float) -> _Response:
        if captured is not None:
            captured.append({"url": url, "payload": json})
        return response

    monkeypatch.setattr("kompend.rag.multimodal.httpx.post", fake_post)


def _figure(tmp_path: Path) -> Path:
    figure = tmp_path / "figure.png"
    figure.write_bytes(PNG)
    return figure


def _wiki_with_figure(figure: Path) -> CompiledWiki:
    return CompiledWiki(
        nodes=[
            WikiNode(
                id="n1",
                title="t",
                content="c",
                node_type=NodeType.CONCEPT,
                metadata={MetadataField.IMAGES: [{"path": str(figure), "caption": "a figure"}]},
            ),
        ],
        edges=[],
        toc=[],
    )
