"""Tests for the capture and distill modules."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

from knox.capture.capture import (
    _merge_tags,
    capture,
)
from knox.capture.distill import (
    CapturedNote,
    _group_by_tag,
    _parse_frontmatter,
    distill,
)
from knox.cli.cmds.capture import _build_references_section
from knox.core.ingest import _extract_text_from_cached
from knox.output.markdown import MarkdownWriter
from knox.output.models import Document


class TestParseFrontmatter:
    def test_with_frontmatter(self) -> None:
        text = "---\ntitle: My Note\ntags:\n  - python\n  - testing\n---\n\nSome content here."
        meta, body = _parse_frontmatter(text)
        assert meta["title"] == "My Note"
        assert meta["tags"] == ["python", "testing"]
        assert body.strip() == "Some content here."

    def test_without_frontmatter(self) -> None:
        text = "Just plain markdown."
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == "Just plain markdown."

    def test_malformed_frontmatter(self) -> None:
        text = "---\nnot valid yaml: [unclosed\n---\nbody"
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text


class TestGroupByTag:
    def test_groups_by_first_tag(self) -> None:
        notes = [
            CapturedNote(title="a", content="x", tags=["python", "test"]),
            CapturedNote(title="b", content="y", tags=["python"]),
            CapturedNote(title="c", content="z", tags=["rust"]),
        ]
        groups = _group_by_tag(notes)
        assert set(groups.keys()) == {"python", "rust"}
        assert len(groups["python"]) == 2
        assert len(groups["rust"]) == 1

    def test_untagged_goes_to_general(self) -> None:
        notes = [CapturedNote(title="a", content="x", tags=[])]
        groups = _group_by_tag(notes)
        assert "general" in groups
        assert len(groups["general"]) == 1


class TestDistill:
    def test_missing_captured_dir_returns_empty(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()
        result = distill(sources_dir)
        assert result == []

    def test_empty_captured_dir_returns_empty(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        (sources_dir / "_captured").mkdir(parents=True)
        result = distill(sources_dir)
        assert result == []

    def test_fallback_without_llm(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        captured_dir = sources_dir / "_captured"
        captured_dir.mkdir(parents=True)
        (captured_dir / "note-1.md").write_text(
            "---\ntitle: Decision A\ntags:\n  - arch\n---\n\nUse QKV projections.\n",
            encoding="utf-8",
        )
        (captured_dir / "note-2.md").write_text(
            "---\ntitle: Discovery B\ntags:\n  - arch\n---\n\nRoPE works well.\n",
            encoding="utf-8",
        )

        paths = distill(sources_dir)
        assert len(paths) == 1
        output = paths[0].read_text(encoding="utf-8")
        assert "type: distilled" in output
        assert "Decision A" in output
        assert "Discovery B" in output
        assert "note-1.md" in output
        assert "note-2.md" in output

    def test_with_mock_llm(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        captured_dir = sources_dir / "_captured"
        captured_dir.mkdir(parents=True)
        (captured_dir / "note-1.md").write_text(
            "---\ntitle: Note A\ntags:\n  - ml\n---\n\nSome content.\n",
            encoding="utf-8",
        )

        mock_result = MagicMock()
        mock_result.json_data = {
            "title": "Synthesized ML Page",
            "content": "## Overview\n\nIntegrated content.",
            "tags": ["ml", "synthesized"],
        }
        mock_client = MagicMock()
        mock_client.try_complete.return_value = mock_result

        paths = distill(sources_dir, llm_client=mock_client)
        assert len(paths) == 1
        output = paths[0].read_text(encoding="utf-8")
        assert "Synthesized ML Page" in output
        assert "Integrated content" in output
        assert "synthesized" in output
        assert mock_client.try_complete.called

    def test_llm_failure_falls_back(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        captured_dir = sources_dir / "_captured"
        captured_dir.mkdir(parents=True)
        (captured_dir / "note-1.md").write_text(
            "---\ntitle: Note A\ntags:\n  - test\n---\n\nContent A.\n",
            encoding="utf-8",
        )

        mock_result = MagicMock()
        mock_result.json_data = None
        mock_client = MagicMock()
        mock_client.try_complete.return_value = mock_result

        paths = distill(sources_dir, llm_client=mock_client)
        assert len(paths) == 1
        output = paths[0].read_text(encoding="utf-8")
        assert "Content A" in output

    def test_multiple_tags_produce_multiple_pages(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        captured_dir = sources_dir / "_captured"
        captured_dir.mkdir(parents=True)
        for tag in ("alpha", "beta"):
            (captured_dir / f"note-{tag}.md").write_text(
                f"---\ntitle: Note {tag}\ntags:\n  - {tag}\n---\n\nContent {tag}.\n",
                encoding="utf-8",
            )

        paths = distill(sources_dir)
        assert len(paths) == 2


class TestCapture:
    def test_no_llm_applies_user_tags(self, tmp_path: Path) -> None:
        paths = capture(
            "The transformer uses d_model=512",
            sources_dir=tmp_path,
            llm_client=None,
            tags=["architecture", "decoder"],
        )
        assert len(paths) == 1
        content = paths[0].read_text(encoding="utf-8")
        assert "architecture" in content
        assert "decoder" in content

    def test_merge_tags_user_first_dedup(self) -> None:
        merged = _merge_tags(["alpha", "Beta"], ["beta", "gamma"])
        assert merged == ["alpha", "Beta", "gamma"]

    def test_merge_tags_empty(self) -> None:
        assert _merge_tags([], []) == []


class TestWriteCmd:
    """knox-zkfd: write resolves sources through cache, writes to write/ subfolder."""

    def test_write_reads_cache_not_raw_bytes(self, tmp_path: Path) -> None:
        """When a cache JSON exists, write_cmd extracts clean text, not raw PDF bytes."""
        # Create a minimal parse-cache JSON for a fake PDF.
        cache_dir = tmp_path / ".knox" / "cache"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / "paper.json"
        cache_file.write_text(
            '{"elements": [{"content": "This is clean extracted text from the PDF."}]}',
            encoding="utf-8",
        )
        text = _extract_text_from_cached(cache_file)
        assert "clean extracted text" in text
        assert "%PDF" not in text

    def test_write_default_frontmatter(self, tmp_path: Path) -> None:
        """Knox-rxod+zkfd: the write page carries kind, compiled_at, sources."""
        compiled_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        fm = {
            "title": "My Page",
            "kind": "page",
            "compiled_at": compiled_at,
            "sources": ["paper"],
        }
        doc = Document(title="My Page", body="Content.", frontmatter=fm, node_type="page")
        writer = MarkdownWriter()
        rendered = writer.renderer.render(doc)
        assert "kind: page" in rendered
        assert "compiled_at:" in rendered
        assert "sources:" in rendered

    def test_references_section_built(self) -> None:
        """_build_references_section lists sources and related wikilinks."""
        section = _build_references_section(
            [("paper_a", "text"), ("paper_b", "text")],
            related_titles=["Listening Effort", "Better Ear"],
        )
        assert "paper_a" in section
        assert "paper_b" in section
        assert "[[Listening Effort]]" in section
        assert "[[Better Ear]]" in section
