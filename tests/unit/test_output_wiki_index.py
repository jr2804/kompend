"""Tests for wiki index/log/overview generation (N11)."""

from __future__ import annotations

from pathlib import Path

from knox.core.models import CompiledWiki, WikiNode
from knox.output.wiki_index import (
    generate_index,
    generate_log_entry,
    prepend_log,
    write_index_and_log,
)


def _make_wiki(**overrides: object) -> CompiledWiki:
    """Create a minimal CompiledWiki for testing."""
    defaults: dict = {
        "nodes": [
            WikiNode(id="attention", title="Attention", content="Attention is all you need.", node_type="concept"),
            WikiNode(id="transformer", title="Transformer", content="Uses attention.", node_type="concept"),
        ],
        "edges": [],
        "toc": [{"id": "attention", "title": "Attention"}, {"id": "transformer", "title": "Transformer"}],
        "metadata": {"source_count": 5, "source_names": ["a.pdf", "b.pdf", "c.pdf"]},
    }
    defaults.update(overrides)
    return CompiledWiki(**defaults)


class TestGenerateIndex:
    def test_basic(self) -> None:
        wiki = _make_wiki()
        result = generate_index(wiki)
        assert "Wiki Index" in result
        assert "**Sources:**" in result
        assert "5" in result
        assert "**Articles:**" in result
        assert "2" in result

    def test_empty_wiki(self) -> None:
        wiki = _make_wiki(nodes=[], metadata={})
        result = generate_index(wiki)
        assert "**Sources:**" in result
        assert "0" in result
        assert "**Articles:**" in result


class TestGenerateLogEntry:
    def test_basic(self) -> None:
        wiki = _make_wiki()
        entry = generate_log_entry(wiki)
        assert "[20" in entry  # ISO date
        assert "2 page(s)" in entry
        assert "concept" in entry  # node_type breakdown

    def test_source_names(self) -> None:
        wiki = _make_wiki()
        entry = generate_log_entry(wiki)
        assert "a.pdf" in entry
        assert "b.pdf" in entry


class TestPrependLog:
    def test_creates_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.md"
        prepend_log(log_path, "## [2026-01-01] test\n")
        assert log_path.exists()
        assert "test" in log_path.read_text(encoding="utf-8")

    def test_newest_on_top(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log.md"
        prepend_log(log_path, "## [2026-01-01] first\n")
        prepend_log(log_path, "## [2026-01-02] second\n")
        text = log_path.read_text(encoding="utf-8")
        assert "first" in text
        assert "second" in text
        # newest entry must appear BEFORE the older entry
        assert text.index("second") < text.index("first")


class TestWriteIndexAndLog:
    def test_writes_both_files(self, tmp_path: Path) -> None:
        wiki = _make_wiki()
        result = write_index_and_log(wiki, tmp_path)
        assert "index" in result
        assert "log" in result
        assert result["index"].exists()
        assert result["log"].exists()

    def test_index_content(self, tmp_path: Path) -> None:
        wiki = _make_wiki()
        write_index_and_log(wiki, tmp_path)
        index_text = (tmp_path / "index.md").read_text(encoding="utf-8")
        assert "**Sources:**" in index_text
        assert "**Articles:**" in index_text
        assert "5" in index_text
        assert "2" in index_text
