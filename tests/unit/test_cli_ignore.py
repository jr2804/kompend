"""Tests for CLI ignore pattern matching (_ignore.py)."""

from __future__ import annotations

from pathlib import Path

from kompend.cli._ignore import compile_patterns, filter_files, load_gitignore, matches_any


class TestCompilePatterns:
    def test_empty(self) -> None:
        assert compile_patterns() == []

    def test_single_list(self) -> None:
        result = compile_patterns(["*.pdf", "*.tmp"])
        assert result == ["*.pdf", "*.tmp"]

    def test_multiple_lists(self) -> None:
        result = compile_patterns(["*.pdf"], ["*.tmp"], ["*.pdf"])
        assert result == ["*.pdf", "*.tmp"]  # deduplicated

    def test_none_lists(self) -> None:
        result = compile_patterns(None, ["*.tmp"], None)
        assert result == ["*.tmp"]


class TestMatchesAny:
    def test_empty_patterns(self) -> None:
        assert matches_any(Path("test.pdf"), []) is False

    def test_match_filename(self, tmp_path: Path) -> None:
        f = tmp_path / "test.pdf"
        assert matches_any(f, ["*.pdf"]) is True

    def test_no_match_different_ext(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        assert matches_any(f, ["*.pdf"]) is False

    def test_match_nested(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub" / "doc.pdf"
        nested.parent.mkdir()
        nested.write_text("x")
        assert matches_any(nested, ["*.pdf"], base_dir=tmp_path) is True

    def test_match_glob_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "drafts" / "doc.pdf"
        nested.parent.mkdir()
        nested.write_text("x")
        assert matches_any(nested, ["drafts/*"], base_dir=tmp_path) is True

    def test_no_match_when_outside_base(self, tmp_path: Path) -> None:
        """Path outside base_dir should still match by name."""
        outside = Path("/tmp/outside.pdf")  # noqa: S108
        assert matches_any(outside, ["*.pdf"]) is True


class TestFilterFiles:
    def test_no_patterns(self) -> None:
        files = [Path("a.pdf"), Path("b.txt")]
        result = filter_files(files)
        assert result == files

    def test_filter_by_extension(self) -> None:
        files = [Path("a.pdf"), Path("b.txt"), Path("c.pdf")]
        result = filter_files(files, ignore_patterns=["*.pdf"])
        assert result == [Path("b.txt")]

    def test_filter_deduplicates(self) -> None:
        files = [Path("a.pdf"), Path("b.pdf"), Path("a.pdf")]
        result = filter_files(files, ignore_patterns=["a.pdf"])
        assert result == [Path("b.pdf")]

    def test_filter_with_gitignore(self, tmp_path: Path) -> None:
        """filter_files with respect_gitignore loads patterns from .gitignore."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.log\n")
        source = tmp_path / "app.log"
        source.write_text("log")
        normal = tmp_path / "app.py"
        normal.write_text("code")

        result = filter_files(
            [source, normal],
            respect_gitignore=True,
            base_dir=tmp_path,
        )
        assert normal in result
        assert source not in result


class TestLoadGitignore:
    def test_loads_from_source_dir(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.log\nbuild/\n")
        patterns = load_gitignore(tmp_path)
        assert "*.log" in patterns
        assert "build/" in patterns

    def test_no_gitignore(self, tmp_path: Path) -> None:
        patterns = load_gitignore(tmp_path)
        # Don't assert empty — parent dirs may have .gitignore files.
        # Just assert it returns a list and doesn't crash.
        assert isinstance(patterns, list)

    def test_ignores_comments(self, tmp_path: Path) -> None:
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("# comment\n*.tmp\n")
        patterns = load_gitignore(tmp_path)
        assert "# comment" not in patterns
        assert "*.tmp" in patterns
