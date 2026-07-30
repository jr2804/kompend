"""Tests for wiki index/log/overview generation (N11)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from knox.core.models import CompiledWiki, WikiNode
from knox.output.wiki_index import (
    generate_index,
    generate_log_entry,
    generate_overview,
    prepend_log,
    write_index,
)


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

    def test_extraction_stats_table(self) -> None:
        wiki = _make_wiki(
            metadata={
                "source_count": 2,
                "source_names": ["a.pdf", "b.pdf"],
                "extraction_stats": [
                    {
                        "source": "a.pdf",
                        "summary": "Paper on attention mechanisms.",
                        "duration_s": 12.3,
                        "summary_words": 340,
                        "entity_count": 12,
                        "concept_count": 5,
                        "tag_count": 3,
                        "batches": 2,
                    },
                    {
                        "source": "b.pdf",
                        "summary": "",
                        "duration_s": 8.1,
                        "summary_words": 0,
                        "entity_count": 4,
                        "concept_count": 2,
                        "tag_count": 0,
                        "batches": 1,
                    },
                ],
            }
        )
        entry = generate_log_entry(wiki)
        assert "**Extraction:**" in entry
        assert "a.pdf" in entry
        assert "12.3s" in entry
        assert "340w" in entry  # summary words rendered
        assert "b.pdf" in entry
        assert "8.1s" in entry
        assert "-" in entry  # zero summary renders as dash
        # total row present when >1 source
        assert "**total**" in entry
        assert "20.4s" in entry

    def test_source_names_short_paths(self) -> None:
        """Source table strips full paths to just filenames."""
        wiki = _make_wiki(
            metadata={
                "source_count": 2,
                "source_names": [
                    "/home/user/project/sources/paper_a.pdf",
                    "/home/user/project/sources/paper_b.pdf",
                ],
            }
        )
        entry = generate_log_entry(wiki)
        assert "paper_a.pdf" in entry
        assert "paper_b.pdf" in entry
        # full paths must NOT appear
        assert "/home/user" not in entry
        assert "sources/" not in entry

    def test_extraction_title_column(self) -> None:
        """Extraction stats table shows summary text as title."""
        wiki = _make_wiki(
            metadata={
                "source_count": 1,
                "source_names": ["x.pdf"],
                "extraction_stats": [
                    {
                        "source": "x.pdf",
                        "summary": "Deep learning for NLP tasks.",
                        "duration_s": 5.0,
                        "summary_words": 10,
                        "entity_count": 1,
                        "concept_count": 1,
                        "tag_count": 1,
                        "batches": 1,
                    },
                ],
            }
        )
        entry = generate_log_entry(wiki)
        assert "Deep learning for NLP tasks." in entry

    def test_source_list_title_from_extraction(self) -> None:
        """Source list table picks up titles from extraction_stats."""
        wiki = _make_wiki(
            metadata={
                "source_count": 3,
                "source_names": ["a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf", "f.pdf"],
                "extraction_stats": [
                    {
                        "source": "a.pdf",
                        "summary": "Alpha paper.",
                        "duration_s": 1.0,
                        "summary_words": 5,
                        "entity_count": 1,
                        "concept_count": 1,
                        "tag_count": 1,
                        "batches": 1,
                    },
                ],
            }
        )
        entry = generate_log_entry(wiki)
        # >5 sources triggers table format with Title column
        assert "| Title |" in entry
        assert "Alpha paper." in entry


class TestPrependLog:
    def test_creates_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "log.md"
        prepend_log(log_file, "## [2026-01-01] test\n")
        assert log_file.exists()
        assert "test" in log_file.read_text(encoding="utf-8")

    def test_newest_on_top(self, tmp_path: Path) -> None:
        log_file = tmp_path / "log.md"
        prepend_log(log_file, "## [2026-01-01] first\n")
        prepend_log(log_file, "## [2026-01-02] second\n")
        text = log_file.read_text(encoding="utf-8")
        assert "first" in text
        assert "second" in text
        # newest entry must appear BEFORE the older entry
        assert text.index("second") < text.index("first")


class TestWriteIndex:
    def test_writes_index_file(self, tmp_path: Path) -> None:
        wiki = _make_wiki()
        result = write_index(wiki, tmp_path)
        assert "index" in result
        assert result["index"].exists()

    def test_index_content(self, tmp_path: Path) -> None:
        wiki = _make_wiki()
        write_index(wiki, tmp_path)
        index_text = (tmp_path / "index.md").read_text(encoding="utf-8")
        assert "**Sources:**" in index_text
        assert "**Articles:**" in index_text
        assert "5" in index_text
        assert "2" in index_text


class TestGenerateOverviewProgress:
    """Tests for progress_callback in generate_overview."""

    def test_single_batch(self) -> None:
        """Single batch: callback fires (0,1) then (1,1)."""
        calls: list[tuple[int, int]] = []

        mock_agent = MagicMock()
        mock_agent.run.return_value = {"content": "Overview text"}

        generate_overview(
            ["short text"],
            agent=mock_agent,
            overview_budget=10000,
            progress_callback=lambda c, t: calls.append((c, t)),
        )

        assert calls[0] == (0, 1)
        assert calls[-1] == (1, 1)

    def test_multi_batch_with_merge(self) -> None:
        """Multiple batches: total includes merge step."""
        calls: list[tuple[int, int]] = []

        mock_agent = MagicMock()
        mock_agent.run.return_value = {"content": "Partial overview"}

        generate_overview(
            ["aaa", "bbb", "ccc"],
            agent=mock_agent,
            overview_budget=5,
            progress_callback=lambda c, t: calls.append((c, t)),
        )

        total = calls[0][1]
        assert total > 1
        assert calls[0] == (0, total)
        assert calls[-1] == (total, total)

    def test_no_callback(self) -> None:
        """Works fine without callback."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = {"content": "Overview text"}

        result = generate_overview(["short text"], agent=mock_agent, overview_budget=10000)

        assert result is not None


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
