"""Unit tests for wiki cache write/load round-trip (N1.5)."""

from __future__ import annotations

import json
from typing import Any

from knox.cli.pipeline_cmds import _write_wiki_cache
from knox.core.models import CompiledWiki, WikiNode, load_cached_wiki


def _make_wiki() -> CompiledWiki:
    return CompiledWiki(
        nodes=[
            WikiNode(
                id="intro",
                title="Introduction",
                content="Hello world",
                node_type="section",
                source_chunks=["chunk-1"],
            ),
        ],
        edges=[{"source": "intro", "target": "intro", "type": "self"}],
        toc=[{"id": "intro", "title": "Introduction"}],
        quality_score=0.85,
        iteration=2,
        metadata={"key": "value"},
    )


class TestWriteWikiCache:
    def test_writes_cache_file(self, tmp_path: Any) -> None:
        wiki = _make_wiki()
        _write_wiki_cache(wiki, tmp_path)

        cache_file = tmp_path / ".knox" / "last_wiki.json"
        assert cache_file.exists()

        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["id"] == "intro"
        assert data["quality_score"] == 0.85
        assert data["iteration"] == 2

    def test_creates_knox_dir(self, tmp_path: Any) -> None:
        wiki = _make_wiki()
        assert not (tmp_path / ".knox").exists()
        _write_wiki_cache(wiki, tmp_path)
        assert (tmp_path / ".knox").exists()

    def test_overwrites_existing(self, tmp_path: Any) -> None:
        wiki1 = _make_wiki()
        _write_wiki_cache(wiki1, tmp_path)

        wiki2 = CompiledWiki(nodes=[], edges=[], toc=[], quality_score=0.5)
        _write_wiki_cache(wiki2, tmp_path)

        loaded = load_cached_wiki(tmp_path)
        assert loaded is not None
        assert len(loaded.nodes) == 0
        assert loaded.quality_score == 0.5


class TestLoadCachedWiki:
    def test_returns_none_when_no_cache(self, tmp_path: Any) -> None:
        result = load_cached_wiki(tmp_path)
        assert result is None

    def test_round_trip(self, tmp_path: Any) -> None:
        wiki = _make_wiki()
        _write_wiki_cache(wiki, tmp_path)

        loaded = load_cached_wiki(tmp_path)
        assert loaded is not None
        assert len(loaded.nodes) == 1
        assert loaded.nodes[0].title == "Introduction"
        assert loaded.nodes[0].content == "Hello world"
        assert len(loaded.edges) == 1
        assert loaded.quality_score == 0.85
        assert loaded.iteration == 2

    def test_returns_none_on_corrupt_json(self, tmp_path: Any) -> None:
        cache_dir = tmp_path / ".knox"
        cache_dir.mkdir(parents=True)
        (cache_dir / "last_wiki.json").write_text("{invalid json", encoding="utf-8")
        result = load_cached_wiki(tmp_path)
        assert result is None
