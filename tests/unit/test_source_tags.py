"""Unit tests for source-subdirectory tags (knox-ol0r)."""

from __future__ import annotations

from pathlib import Path

from knox.core.pipeline.orchestrator import WikiPipeline


class TestExtractSourceTags:
    """Tests for WikiPipeline._extract_source_tags."""

    def test_deep_nesting(self, tmp_path: Path) -> None:
        source = tmp_path / "sources" / "2024" / "conferences" / "iclr" / "paper.pdf"
        source_dir = tmp_path / "sources"
        assert WikiPipeline._extract_source_tags(source, source_dir) == ["2024", "conferences", "iclr"]

    def test_single_subdir(self, tmp_path: Path) -> None:
        source = tmp_path / "sources" / "journals" / "nature.pdf"
        source_dir = tmp_path / "sources"
        assert WikiPipeline._extract_source_tags(source, source_dir) == ["journals"]

    def test_flat_in_sources(self, tmp_path: Path) -> None:
        source = tmp_path / "sources" / "paper.pdf"
        source_dir = tmp_path / "sources"
        assert WikiPipeline._extract_source_tags(source, source_dir) == []

    def test_file_outside_sources(self, tmp_path: Path) -> None:
        source = tmp_path / "external" / "paper.pdf"
        source_dir = tmp_path / "sources"
        assert WikiPipeline._extract_source_tags(source, source_dir) == []

    def test_relative_path_with_dotdot(self, tmp_path: Path) -> None:
        source = tmp_path / "sources" / "sub" / ".." / "paper.pdf"
        source_dir = tmp_path / "sources"
        resolved = source.resolve()
        source_dir_resolved = source_dir.resolve()
        result = WikiPipeline._extract_source_tags(resolved, source_dir_resolved)
        assert result == []

    def test_empty_source_dir_name(self, tmp_path: Path) -> None:
        source = tmp_path / "sources" / "a" / "b" / "c.pdf"
        source_dir = tmp_path / "sources"
        tags = WikiPipeline._extract_source_tags(source, source_dir)
        assert tags == ["a", "b"]
