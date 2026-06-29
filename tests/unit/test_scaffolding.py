"""Unit tests for the knox.scaffolding package."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from knox.core.paths import CACHE_SUBDIR, KNOX_DIR
from knox.scaffolding import (
    AUTHORED_TEMPLATES,
    PROJECT_MARKERS,
    ScaffoldError,
    find_existing_project,
    scaffold_project,
)


class TestScaffoldProject:
    def test_creates_standard_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        result = scaffold_project(target)
        assert (target / "sources").is_dir()
        assert (target / "wiki").is_dir()
        assert (target / KNOX_DIR).is_dir()
        assert (target / KNOX_DIR / CACHE_SUBDIR).is_dir()
        assert len(result.created_dirs) == 4

    def test_creates_config_file(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        scaffold_project(target)
        assert (target / KNOX_DIR / "config.toml").is_file()
        content = (target / KNOX_DIR / "config.toml").read_text(encoding="utf-8")
        assert "[compile]" in content
        assert "compile.chunk_size" not in content  # dotted-form not used

    def test_creates_gitignore(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        scaffold_project(target)
        gitignore = target / ".gitignore"
        assert gitignore.is_file()
        content = gitignore.read_text(encoding="utf-8")
        assert "wiki/" in content
        assert ".knox/cache/" in content

    def test_renders_templates_with_custom_paths(self, tmp_path: Path) -> None:
        """Templates reflect config paths.sources_dir / output_dir, not defaults."""
        target = tmp_path / "wiki"
        config_file = tmp_path / "custom.toml"
        config_file.write_text('[paths]\nsources_dir = "my_docs"\noutput_dir = "my_output"\n', encoding="utf-8")
        result = scaffold_project(target, force=True, config_file=config_file)

        # Directories created from config
        assert (target / "my_docs").is_dir()
        assert (target / "my_output").is_dir()
        assert result.sources_dir == "my_docs"
        assert result.output_dir == "my_output"

        # Rendered templates use config paths
        gitignore = (target / ".gitignore").read_text(encoding="utf-8")
        assert "my_output/" in gitignore
        assert "wiki/" not in gitignore

        readme = (target / "README.md").read_text(encoding="utf-8")
        assert "my_docs/" in readme
        assert "my_output/" in readme
        assert "sources/" not in readme

    def test_creates_authored_templates_by_default(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        result = scaffold_project(target)
        assert (target / "NOTES.md").is_file()
        assert (target / "README.md").is_file()
        assert result.skipped_files == []
        assert len(result.copied_files) == 4  # config, gitignore, notes, readme

    def test_bare_skips_authored_templates(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        result = scaffold_project(target, bare=True)
        assert not (target / "NOTES.md").exists()
        assert not (target / "README.md").exists()
        assert (target / KNOX_DIR / "config.toml").is_file()
        assert (target / ".gitignore").is_file()
        assert set(result.skipped_files) == set(AUTHORED_TEMPLATES)
        assert len(result.copied_files) == 2  # config + gitignore

    def test_creates_nested_target(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "wiki"
        result = scaffold_project(target)
        assert target.is_dir()
        assert (target / "sources").is_dir()
        assert len(result.created_dirs) == 4

    def test_existing_project_knox_toml_rejected_without_force(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / "knox.toml").write_text("# existing", encoding="utf-8")
        with pytest.raises(ScaffoldError, match="Existing project"):
            scaffold_project(target)

    def test_existing_project_config_toml_rejected_without_force(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KNOX_DIR).mkdir()
        (target / KNOX_DIR / "config.toml").write_text("# existing", encoding="utf-8")
        with pytest.raises(ScaffoldError, match="Existing project"):
            scaffold_project(target)

    def test_force_overwrites(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KNOX_DIR).mkdir()
        old_config = target / KNOX_DIR / "config.toml"
        old_config.write_text("# old", encoding="utf-8")
        scaffold_project(target, force=True)
        assert old_config.read_text(encoding="utf-8") != "# old"

    def test_target_path_is_file_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "not_a_dir"
        target.write_text("file", encoding="utf-8")
        with pytest.raises(ScaffoldError, match="not a directory"):
            scaffold_project(target)

    def test_default_target_is_cwd_relative(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = scaffold_project(".")
        assert result.target_dir == tmp_path.resolve()
        assert (result.target_dir / "sources").is_dir()

    def test_git_init_runs_when_requested(self, tmp_path: Path) -> None:
        if not _git_available():
            pytest.skip("git not on PATH")
        target = tmp_path / "wiki"
        result = scaffold_project(target, git=True)
        assert result.git_initialised is True
        assert (target / ".git").is_dir()

    def test_git_init_skipped_by_default(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        result = scaffold_project(target)
        assert result.git_initialised is False
        assert not (target / ".git").exists()

    def test_git_init_propagates_error(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        with (
            patch("knox.scaffolding._core.subprocess.run", side_effect=subprocess.CalledProcessError(1, "git", stderr=b"boom")),
            pytest.raises(ScaffoldError, match="git init failed"),
        ):
            scaffold_project(target, git=True)

    def test_keep_cache_preserves_cache_contents(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KNOX_DIR).mkdir()
        (target / KNOX_DIR / "config.toml").write_text("# old", encoding="utf-8")
        cache_dir = target / KNOX_DIR / CACHE_SUBDIR
        cache_dir.mkdir()
        (cache_dir / "doc1.json").write_text("{}", encoding="utf-8")
        (cache_dir / "doc2.json").write_text("{}", encoding="utf-8")

        scaffold_project(target, force=True, keep_cache=True)

        assert (cache_dir / "doc1.json").is_file()
        assert (cache_dir / "doc2.json").is_file()

    def test_keep_cache_removes_other_files(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KNOX_DIR).mkdir()
        (target / KNOX_DIR / "config.toml").write_text("# old", encoding="utf-8")
        (target / "stale_output.md").write_text("old wiki", encoding="utf-8")
        cache_dir = target / KNOX_DIR / CACHE_SUBDIR
        cache_dir.mkdir()
        (cache_dir / "doc1.json").write_text("{}", encoding="utf-8")

        scaffold_project(target, force=True, keep_cache=True)

        assert not (target / "stale_output.md").exists()
        assert (cache_dir / "doc1.json").is_file()

    def test_keep_cache_noop_without_force(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KNOX_DIR).mkdir()
        (target / KNOX_DIR / "config.toml").write_text("# existing", encoding="utf-8")

        with pytest.raises(ScaffoldError, match="Existing project"):
            scaffold_project(target, keep_cache=True)

    def test_keep_cache_no_existing_cache(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KNOX_DIR).mkdir()
        (target / KNOX_DIR / "config.toml").write_text("# old", encoding="utf-8")

        scaffold_project(target, force=True, keep_cache=True)

        assert (target / KNOX_DIR / CACHE_SUBDIR).is_dir()
        assert not any((target / KNOX_DIR / CACHE_SUBDIR).iterdir())

    def test_keep_cache_empty_cache_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KNOX_DIR).mkdir()
        (target / KNOX_DIR / "config.toml").write_text("# old", encoding="utf-8")
        (target / KNOX_DIR / CACHE_SUBDIR).mkdir()

        scaffold_project(target, force=True, keep_cache=True)

        assert (target / KNOX_DIR / CACHE_SUBDIR).is_dir()
        assert not any((target / KNOX_DIR / CACHE_SUBDIR).iterdir())


def _git_available() -> bool:
    return shutil.which("git") is not None


class TestPublicApi:
    def test_project_markers_constant(self) -> None:
        assert "knox.toml" in PROJECT_MARKERS
        assert ".knox/config.toml" in PROJECT_MARKERS

    def test_authored_templates_constant(self) -> None:
        assert "NOTES.md" in AUTHORED_TEMPLATES

    def test_find_existing_project_returns_none_for_empty_dir(self, tmp_path: Path) -> None:
        assert find_existing_project(tmp_path) is None

    def test_find_existing_project_detects_knox_toml(self, tmp_path: Path) -> None:
        marker = tmp_path / "knox.toml"
        marker.write_text("# existing", encoding="utf-8")
        assert find_existing_project(tmp_path) == marker

    def test_find_existing_project_detects_config_toml(self, tmp_path: Path) -> None:
        (tmp_path / KNOX_DIR).mkdir()
        marker = tmp_path / KNOX_DIR / "config.toml"
        marker.write_text("# existing", encoding="utf-8")
        assert find_existing_project(tmp_path) == marker
        assert "README.md" in AUTHORED_TEMPLATES
