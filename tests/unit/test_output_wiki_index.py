"""Tests for wiki index/log/overview generation (N11)."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

from kompend.agents.types import OverviewDeps, OverviewPage
from kompend.agents.wiki_index import AgenticOverviewAgent
from kompend.core.models import CompiledWiki, WikiNode
from kompend.output.pages.index import _full_catalog, exported_sources_section, generate_index, sync_exported_sources_section, write_index
from kompend.output.pages.overview import generate_overview_agentic
from kompend.output.wiki_index import (
    LogStage,
    append_log_entry,
    generate_log_entry,
    generate_overview,
)

_ISO_DATE_HEADING_RE = re.compile(r"## \d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?")


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
        entry = generate_log_entry(wiki, stage=LogStage.COMPILE)
        assert "[20" in entry  # ISO date
        assert "2 page(s)" in entry
        assert "concept" in entry  # node_type breakdown

    def test_source_names(self) -> None:
        wiki = _make_wiki()
        entry = generate_log_entry(wiki, stage=LogStage.COMPILE)
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
        entry = generate_log_entry(wiki, stage=LogStage.COMPILE)
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
        entry = generate_log_entry(wiki, stage=LogStage.COMPILE)
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
        entry = generate_log_entry(wiki, stage=LogStage.COMPILE)
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
        entry = generate_log_entry(wiki, stage=LogStage.COMPILE)
        # >5 sources triggers table format with Title column
        assert "| Title |" in entry
        assert "Alpha paper." in entry


class TestAppendLogEntry:
    def test_creates_file_with_header(self, tmp_path: Path) -> None:
        log_file = tmp_path / "log.md"
        append_log_entry(log_file, action="Ingest", detail="1 document(s)")
        text = log_file.read_text(encoding="utf-8")
        assert text.startswith("# Bundle Update Log")
        assert "**Ingest**: 1 document(s)" in text

    def test_newest_section_first(self, tmp_path: Path) -> None:
        log_file = tmp_path / "log.md"
        append_log_entry(log_file, action="Creation", detail="first")
        append_log_entry(log_file, action="Query", detail="second")
        text = log_file.read_text(encoding="utf-8")
        assert "first" in text
        assert "second" in text
        # newest entry must appear BEFORE the older entry
        assert text.index("second") < text.index("first")

    def test_each_entry_gets_timestamped_heading(self, tmp_path: Path) -> None:
        log_file = tmp_path / "log.md"
        append_log_entry(log_file, action="Creation", detail="one")
        append_log_entry(log_file, action="Lint", detail="two")
        text = log_file.read_text(encoding="utf-8")
        # one date-grouped heading, both entries as timestamped list items
        assert len(_ISO_DATE_HEADING_RE.findall(text)) == 1
        assert len(re.findall(r"^- \[\d{2}:\d{2}:\d{2}\] \*\*\w+\*\*", text, re.M)) == 2
        # newest entry first within the day
        assert text.index("**Lint**") < text.index("**Creation**")
        # MD022/MD032: blank line between heading and list
        assert re.search(r"## \d{4}-\d{2}-\d{2}\n\n- \[", text)


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


class TestAgenticOverview:
    """Agentic overview synthesis: one multi-turn session for all pages."""

    def test_one_session_reads_pages_and_saves(self) -> None:
        """Agent runs once, reads every page, submits the overview via save_overview()."""
        agent = _mock_agentic_overview("Thematic overview text")
        pages = [
            OverviewPage(slug="a", title="Alpha", content="Alpha content"),
            OverviewPage(slug="b", title="Beta", content="Beta content"),
        ]
        result = generate_overview_agentic(pages, agent=agent)
        assert result == "Thematic overview text"
        assert agent.run.call_count == 1

    def test_failure_returns_none(self) -> None:
        agent = MagicMock(spec=AgenticOverviewAgent)
        agent.run.side_effect = RuntimeError("LLM failed")

        result = generate_overview_agentic([OverviewPage(slug="a", title="Alpha", content="x")], agent=agent)
        assert result is None

    def test_empty_pages_returns_none(self) -> None:
        agent = _mock_agentic_overview()
        assert generate_overview_agentic([], agent=agent) is None


# ── generate_index (portable catalog) ────────────────────────────────────


class TestGenerateIndexPortable:
    """Root index: unindented list, relative links, summaries, bundle link."""

    def test_entries_unindented_with_relative_links(self) -> None:

        catalog = _full_catalog(self._wiki(), Path("wiki"), "wiki")
        assert "\n  1." not in catalog  # no indented list (MD005)
        assert "\n1. [Concept A](wiki/concepts/concept-a.md)" in catalog
        assert "\n2. [Introduction](wiki/intro.md)" in catalog

    def test_entries_carry_one_line_summaries(self) -> None:

        catalog = _full_catalog(self._wiki(), Path("wiki"), "wiki")
        assert "Concept A" in catalog
        assert "First line of concept A." in catalog

    def test_links_to_bundle_index(self, tmp_path: Path) -> None:
        """Nested layout: root index is a thin hub pointing at the catalog."""
        write_index(self._wiki(), tmp_path / "wiki", base_dir=tmp_path)
        index = (tmp_path / "index.md").read_text(encoding="utf-8")
        assert "[wiki/index.md](wiki/index.md)" in index
        # De-duplication: the root hub must NOT carry the full TOC.
        assert "## Table of Contents" not in index

    def test_no_obsidian_shortlinks(self, tmp_path: Path) -> None:

        write_index(self._wiki(), tmp_path / "wiki", base_dir=tmp_path)
        assert "[[" not in (tmp_path / "index.md").read_text(encoding="utf-8")

    def _wiki(self) -> CompiledWiki:
        return CompiledWiki(
            nodes=[
                WikiNode(id="concept-a", title="Concept A", content="First line of concept A.", node_type="concept"),
                WikiNode(id="intro", title="Introduction", content="The introduction section.", node_type="section"),
            ],
            edges=[],
            toc=[],
        )


# ── exported sources section ──────────────────────────────────────────────


class TestExportedSourcesSection:
    """``## Exported sources`` listing in the vault-root index."""

    def test_renders_links_from_exported_dir(self, tmp_path: Path) -> None:
        (tmp_path / "exported").mkdir()
        (tmp_path / "exported" / "a.pdf.md").write_text("# A", encoding="utf-8")
        (tmp_path / "exported" / "b.pdf.md").write_text("# B", encoding="utf-8")
        section = exported_sources_section(tmp_path)
        assert "## Exported sources" in section
        assert "- [a.pdf.md](exported/a.pdf.md)" in section
        assert "- [b.pdf.md](exported/b.pdf.md)" in section

    def test_empty_when_no_exported_dir(self, tmp_path: Path) -> None:
        assert exported_sources_section(tmp_path) == ""

    def test_empty_when_dir_holds_no_markdown(self, tmp_path: Path) -> None:
        (tmp_path / "exported").mkdir()
        (tmp_path / "exported" / "keep.bin").write_bytes(b"x")
        assert exported_sources_section(tmp_path) == ""

    def test_hub_includes_section(self, tmp_path: Path) -> None:
        (tmp_path / "exported").mkdir()
        (tmp_path / "exported" / "a.pdf.md").write_text("# A", encoding="utf-8")
        index = generate_index(_make_wiki(), output_dir=tmp_path / "wiki", base_dir=tmp_path)
        assert "## Exported sources" in index
        assert "[a.pdf.md](exported/a.pdf.md)" in index

    def test_full_catalog_includes_section(self, tmp_path: Path) -> None:
        (tmp_path / "exported").mkdir()
        (tmp_path / "exported" / "a.pdf.md").write_text("# A", encoding="utf-8")
        catalog = _full_catalog(self._wiki(), tmp_path / "wiki", "wiki", base_dir=tmp_path)
        assert "## Exported sources" in catalog
        assert "[a.pdf.md](exported/a.pdf.md)" in catalog

    def _wiki(self) -> CompiledWiki:
        return CompiledWiki(
            nodes=[
                WikiNode(id="concept-a", title="Concept A", content="First line of concept A.", node_type="concept"),
            ],
            edges=[],
            toc=[],
        )


class TestSyncExportedSourcesSection:
    """Splice: insert / refresh / remove the section in an existing index."""

    def test_inserts_when_missing(self, tmp_path: Path) -> None:
        index_file = tmp_path / "index.md"
        index_file.write_text("# Wiki Index\n\n**Sources:** 5\n", encoding="utf-8")
        (tmp_path / "exported").mkdir()
        (tmp_path / "exported" / "a.pdf.md").write_text("# A", encoding="utf-8")
        sync_exported_sources_section(index_file, tmp_path)
        text = index_file.read_text(encoding="utf-8")
        assert "## Exported sources" in text
        assert "- [a.pdf.md](exported/a.pdf.md)" in text
        assert "**Sources:** 5" in text

    def test_refresh_is_idempotent(self, tmp_path: Path) -> None:
        index_file = tmp_path / "index.md"
        index_file.write_text("# Wiki Index\n\n## Exported sources\n\n- [stale.md](exported/stale.md)\n", encoding="utf-8")
        (tmp_path / "exported").mkdir()
        (tmp_path / "exported" / "a.pdf.md").write_text("# A", encoding="utf-8")
        sync_exported_sources_section(index_file, tmp_path)
        first = index_file.read_text(encoding="utf-8")
        sync_exported_sources_section(index_file, tmp_path)
        assert index_file.read_text(encoding="utf-8") == first
        assert "stale.md" not in first
        assert "a.pdf.md" in first

    def test_removes_when_dir_empty(self, tmp_path: Path) -> None:
        index_file = tmp_path / "index.md"
        index_file.write_text("# Wiki Index\n\n## Exported sources\n\n- [a.md](exported/a.md)\n", encoding="utf-8")
        sync_exported_sources_section(index_file, tmp_path)
        text = index_file.read_text(encoding="utf-8")
        assert "Exported sources" not in text
        assert "# Wiki Index" in text

    def test_noop_when_index_missing(self, tmp_path: Path) -> None:
        sync_exported_sources_section(tmp_path / "index.md", tmp_path)
        assert not (tmp_path / "index.md").exists()


# ── Agentic overview ────────────────────────────────────────────────────


def _mock_agentic_overview(content: str = "Thematic overview.") -> MagicMock:
    agent = MagicMock(spec=AgenticOverviewAgent)

    def _run(deps: OverviewDeps) -> None:
        deps.overview = content

    agent.run.side_effect = _run
    return agent


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
