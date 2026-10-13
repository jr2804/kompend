"""Unit tests for wiki cache write/load round-trip (N1.5)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from kompend.cli._pipeline_helpers import write_wiki_cache
from kompend.core.models import CompiledWiki, WikiNode, load_cached_wiki
from kompend.core.paths import KOMP_DIR
from kompend.core.store import open_store, vault_db_path


class TestWriteWikiCache:
    def test_writes_snapshot(self, tmp_path: Any) -> None:
        wiki = _make_wiki()
        write_wiki_cache(wiki, tmp_path)

        assert vault_db_path(tmp_path).exists()
        with open_store(vault_db_path(tmp_path)) as conn:
            row = conn.execute("SELECT nodes, quality_score, iteration FROM wiki_snapshot WHERE kind = 'last'").fetchone()
        assert row is not None
        assert len(json.loads(row["nodes"])) == 1
        assert json.loads(row["nodes"])[0]["id"] == "intro"
        assert row["quality_score"] == 0.85
        assert row["iteration"] == 2

    def test_creates_kompend_dir(self, tmp_path: Any) -> None:
        wiki = _make_wiki()
        assert not (tmp_path / KOMP_DIR).exists()
        write_wiki_cache(wiki, tmp_path)
        assert (tmp_path / KOMP_DIR).exists()

    def test_overwrites_existing(self, tmp_path: Any) -> None:
        wiki1 = _make_wiki()
        write_wiki_cache(wiki1, tmp_path)

        wiki2 = CompiledWiki(nodes=[], edges=[], toc=[], quality_score=0.5)
        write_wiki_cache(wiki2, tmp_path)

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
        write_wiki_cache(wiki, tmp_path)

        loaded = load_cached_wiki(tmp_path)
        assert loaded is not None
        assert len(loaded.nodes) == 1
        assert loaded.nodes[0].title == "Introduction"
        assert loaded.nodes[0].content == "Hello world"
        assert len(loaded.edges) == 1
        assert loaded.quality_score == 0.85
        assert loaded.iteration == 2

    def test_corrupt_payload_raises(self, tmp_path: Any) -> None:
        # Per src/kompend/AGENTS.md: a corrupt wiki snapshot is a bug, not a silent miss.
        # Missing -> None (caller prompts to compile); corrupt -> raises so the
        # real cause surfaces instead of a misleading "run compile first".
        db_path = vault_db_path(tmp_path)
        with open_store(db_path) as conn:
            conn.execute("INSERT INTO wiki_snapshot(kind, nodes) VALUES ('last', '{invalid json')")
        with pytest.raises(json.JSONDecodeError):
            load_cached_wiki(tmp_path)


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
        toc=[{"id": "intro", "title": "Introduction", "level": 1}],
        quality_score=0.85,
        iteration=2,
        metadata={"key": "value"},
    )
