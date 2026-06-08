"""Tests for CLI source file resolution (_sources.py)."""

from __future__ import annotations

from pathlib import Path

from knox.cli._sources import (
    SUPPORTED_EXTENSIONS,
    _unique_path,
    ensure_in_sources,
    resolve_files,
)

# ── resolve_files ───────────────────────────────────────────────────────


class TestResolveFiles:
    def test_single_file(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("pdf")
        result = resolve_files(str(pdf_file))
        assert result == [pdf_file]

    def test_multiple_files(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "a.pdf"
        docx_file = tmp_path / "b.docx"
        pdf_file.write_text("a")
        docx_file.write_text("b")
        result = resolve_files(str(pdf_file), str(docx_file))
        assert len(result) == 2
        assert pdf_file in result
        assert docx_file in result

    def test_glob_pattern(self, tmp_path: Path) -> None:
        for name in ("doc1.pdf", "doc2.pdf", "readme.txt"):
            (tmp_path / name).write_text(name)
        result = resolve_files(str(tmp_path / "*.pdf"))
        assert len(result) == 2
        names = {p.name for p in result}
        assert names == {"doc1.pdf", "doc2.pdf"}

    def test_glob_recursive(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()
        (tmp_path / "top.pdf").write_text("top")
        (sub_dir / "nested.pdf").write_text("nested")
        # non-recursive: only top-level
        result = resolve_files(str(tmp_path / "*.pdf"), recursive=False)
        assert len(result) == 1
        # recursive: both
        result = resolve_files(str(tmp_path / "*.pdf"), recursive=True)
        assert len(result) == 2

    def test_directory_scanning(self, tmp_path: Path) -> None:
        for name in ("a.pdf", "b.docx", "c.md"):
            (tmp_path / name).write_text(name)
        (tmp_path / "ignored.exe").write_text("nope")
        result = resolve_files(str(tmp_path))
        assert len(result) == 3
        assert all(p.suffix in SUPPORTED_EXTENSIONS for p in result)

    def test_directory_recursive(self, tmp_path: Path) -> None:
        deep_dir = tmp_path / "deep"
        deep_dir.mkdir()
        (tmp_path / "top.pdf").write_text("top")
        (deep_dir / "nested.docx").write_text("nested")
        result = resolve_files(str(tmp_path), recursive=True)
        assert len(result) == 2

    def test_unsupported_extension_excluded(self, tmp_path: Path) -> None:
        png_file = tmp_path / "image.png"
        png_file.write_text("png")
        result = resolve_files(str(png_file))
        assert result == []

    def test_deduplication(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_text("pdf")
        result = resolve_files(str(pdf_file), str(pdf_file))
        assert result == [pdf_file]

    def test_empty_input(self) -> None:
        assert resolve_files() == []

    def test_nonexistent_path_returns_empty(self, tmp_path: Path) -> None:
        result = resolve_files(str(tmp_path / "nonexistent.xyz"))
        assert result == []

    def test_mixed_files_and_globs(self, tmp_path: Path) -> None:
        single_file = tmp_path / "single.pdf"
        single_file.write_text("single")
        (tmp_path / "glob1.docx").write_text("g1")
        (tmp_path / "glob2.docx").write_text("g2")
        result = resolve_files(str(single_file), str(tmp_path / "*.docx"))
        assert len(result) == 3


# ── ensure_in_sources ───────────────────────────────────────────────────


class TestEnsureInSources:
    def test_files_already_in_sources(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()
        pdf_file = sources_dir / "doc.pdf"
        pdf_file.write_text("pdf")
        result = ensure_in_sources([pdf_file], sources_dir)
        assert len(result) == 1
        assert result[0] == pdf_file.resolve()

    def test_external_files_copied_in(self, tmp_path: Path) -> None:
        external_dir = tmp_path / "external"
        external_dir.mkdir()
        source_file = external_dir / "paper.pdf"
        source_file.write_text("paper")

        sources_dir = tmp_path / "sources"
        result = ensure_in_sources([source_file], sources_dir)

        assert len(result) == 1
        assert result[0].name == "paper.pdf"
        assert result[0].exists()
        # Original untouched
        assert source_file.exists()

    def test_collision_gets_unique_name(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()
        (sources_dir / "doc.pdf").write_text("existing")

        external_dir = tmp_path / "ext"
        external_dir.mkdir()
        new_file = external_dir / "doc.pdf"
        new_file.write_text("new")

        result = ensure_in_sources([new_file], sources_dir)
        assert result[0].name == "doc_1.pdf"

    def test_sources_dir_created_if_missing(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "not_yet_existing"
        ext_dir = tmp_path / "ext"
        ext_dir.mkdir()
        input_file = ext_dir / "doc.pdf"
        input_file.write_text("x")

        result = ensure_in_sources([input_file], sources_dir)
        assert sources_dir.exists()
        assert result[0].exists()


# ── _unique_path ────────────────────────────────────────────────────────


class TestUniquePath:
    def test_no_conflict(self, tmp_path: Path) -> None:
        target_file = tmp_path / "doc.pdf"
        assert _unique_path(target_file) == target_file

    def test_one_conflict(self, tmp_path: Path) -> None:
        target_file = tmp_path / "doc.pdf"
        target_file.write_text("first")
        result = _unique_path(target_file)
        assert result.name == "doc_1.pdf"

    def test_multiple_conflicts(self, tmp_path: Path) -> None:
        target_file = tmp_path / "doc.pdf"
        target_file.write_text("0")
        (tmp_path / "doc_1.pdf").write_text("1")
        (tmp_path / "doc_2.pdf").write_text("2")
        result = _unique_path(target_file)
        assert result.name == "doc_3.pdf"
