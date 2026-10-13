"""Unit tests for the kompend.scaffolding package."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from kompend.core.paths import CACHE_SUBDIR, KOMP_DIR
from kompend.scaffolding import (
    AUTHORED_TEMPLATES,
    PROJECT_MARKERS,
    RAIL_FILES,
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
        assert (target / KOMP_DIR).is_dir()
        assert (target / KOMP_DIR / CACHE_SUBDIR).is_dir()
        assert len(result.created_dirs) == 4

    def test_creates_config_file(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        scaffold_project(target)
        assert (target / KOMP_DIR / "config.toml").is_file()
        content = (target / KOMP_DIR / "config.toml").read_text(encoding="utf-8")
        assert "[compile]" in content
        assert "compile.budget.chunk_size_min" not in content  # TOML sections used instead of dotted keys

    def test_creates_gitignore(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        scaffold_project(target)
        gitignore = target / ".gitignore"
        assert gitignore.is_file()
        content = gitignore.read_text(encoding="utf-8")
        assert "wiki/" in content
        assert ".kompend/cache/" in content

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
        assert len(result.copied_files) == 5  # config, gitignore, notes, readme, agents

    def test_bare_skips_authored_templates(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        result = scaffold_project(target, bare=True)
        assert not (target / "NOTES.md").exists()
        assert not (target / "README.md").exists()
        assert (target / KOMP_DIR / "config.toml").is_file()
        assert (target / ".gitignore").is_file()
        assert set(result.skipped_files) == set(AUTHORED_TEMPLATES)
        assert len(result.copied_files) == 3  # config + gitignore + agents

    def test_creates_nested_target(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "wiki"
        result = scaffold_project(target)
        assert target.is_dir()
        assert (target / "sources").is_dir()
        assert len(result.created_dirs) == 4

    def test_existing_project_kompend_toml_rejected_without_force(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / "kompend.toml").write_text("# existing", encoding="utf-8")
        with pytest.raises(ScaffoldError, match="Existing project"):
            scaffold_project(target)

    def test_existing_project_config_toml_rejected_without_force(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KOMP_DIR).mkdir()
        (target / KOMP_DIR / "config.toml").write_text("# existing", encoding="utf-8")
        with pytest.raises(ScaffoldError, match="Existing project"):
            scaffold_project(target)

    def test_force_overwrites(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KOMP_DIR).mkdir()
        old_config = target / KOMP_DIR / "config.toml"
        old_config.write_text("# old", encoding="utf-8")
        scaffold_project(target, force=True)
        assert old_config.read_text(encoding="utf-8") != "# old"

    def test_target_is_file_raises(self, tmp_path: Path) -> None:
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
            patch("kompend.scaffolding._core.subprocess.run", side_effect=subprocess.CalledProcessError(1, "git", stderr=b"boom")),
            pytest.raises(ScaffoldError, match="git init failed"),
        ):
            scaffold_project(target, git=True)

    def test_keep_cache_preserves_cache_contents(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KOMP_DIR).mkdir()
        (target / KOMP_DIR / "config.toml").write_text("# old", encoding="utf-8")
        cache_dir = target / KOMP_DIR / CACHE_SUBDIR
        cache_dir.mkdir()
        (cache_dir / "doc1.json").write_text("{}", encoding="utf-8")
        (cache_dir / "doc2.json").write_text("{}", encoding="utf-8")

        scaffold_project(target, force=True, keep_cache=True)

        assert (cache_dir / "doc1.json").is_file()
        assert (cache_dir / "doc2.json").is_file()

    def test_keep_cache_removes_other_files(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KOMP_DIR).mkdir()
        (target / KOMP_DIR / "config.toml").write_text("# old", encoding="utf-8")
        (target / "stale_output.md").write_text("old wiki", encoding="utf-8")
        cache_dir = target / KOMP_DIR / CACHE_SUBDIR
        cache_dir.mkdir()
        (cache_dir / "doc1.json").write_text("{}", encoding="utf-8")

        scaffold_project(target, force=True, keep_cache=True)

        assert not (target / "stale_output.md").exists()
        assert (cache_dir / "doc1.json").is_file()

    def test_keep_cache_noop_without_force(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KOMP_DIR).mkdir()
        (target / KOMP_DIR / "config.toml").write_text("# existing", encoding="utf-8")

        with pytest.raises(ScaffoldError, match="Existing project"):
            scaffold_project(target, keep_cache=True)

    def test_keep_cache_no_existing_cache(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KOMP_DIR).mkdir()
        (target / KOMP_DIR / "config.toml").write_text("# old", encoding="utf-8")

        scaffold_project(target, force=True, keep_cache=True)

        assert (target / KOMP_DIR / CACHE_SUBDIR).is_dir()
        assert not any((target / KOMP_DIR / CACHE_SUBDIR).iterdir())

    def test_keep_cache_empty_cache_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KOMP_DIR).mkdir()
        (target / KOMP_DIR / "config.toml").write_text("# old", encoding="utf-8")
        (target / KOMP_DIR / CACHE_SUBDIR).mkdir()

        scaffold_project(target, force=True, keep_cache=True)

        assert (target / KOMP_DIR / CACHE_SUBDIR).is_dir()
        assert not any((target / KOMP_DIR / CACHE_SUBDIR).iterdir())

    # Rail-file clobber guard (regression for kompend-init clobbering tracked AGENTS.md / .gitignore / README.md).

    def test_rail_files_refused_without_force(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir(parents=True, exist_ok=True)
        sentinel = "# tracked repo content -- do not overwrite"
        (target / ".gitignore").write_text(sentinel, encoding="utf-8")
        (target / "AGENTS.md").write_text(sentinel, encoding="utf-8")
        (target / "README.md").write_text(sentinel, encoding="utf-8")
        (target / "README.md").write_text(sentinel, encoding="utf-8")
        with pytest.raises(ScaffoldError) as exc_info:
            scaffold_project(target)
        msg = str(exc_info.value)
        assert ".gitignore" in msg
        assert "AGENTS.md" in msg
        assert "README.md" in msg
        assert "--force" in msg
        # The originals must be untouched.
        assert (target / ".gitignore").read_text(encoding="utf-8") == sentinel
        assert (target / "AGENTS.md").read_text(encoding="utf-8") == sentinel
        assert (target / "README.md").read_text(encoding="utf-8") == sentinel

    def test_rail_files_overwritten_with_force(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir(parents=True, exist_ok=True)
        sentinel = "# user content"
        (target / ".gitignore").write_text(sentinel, encoding="utf-8")
        (target / "AGENTS.md").write_text(sentinel, encoding="utf-8")
        (target / "README.md").write_text(sentinel, encoding="utf-8")
        result = scaffold_project(target, force=True)
        # Scaffold content overwrote each one.
        assert (target / ".gitignore").read_text(encoding="utf-8") != sentinel
        assert (target / "AGENTS.md").read_text(encoding="utf-8") != sentinel
        # Every clobbered rail file is reported.
        clobbered_rel = {p.relative_to(target) for p in result.clobbered}
        assert Path(".gitignore") in clobbered_rel
        assert Path("AGENTS.md") in clobbered_rel
        assert Path("README.md") in clobbered_rel

    def test_partial_overwrite_records_only_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir(parents=True, exist_ok=True)
        sentinel = "# user content"
        (target / "AGENTS.md").write_text(sentinel, encoding="utf-8")
        (target / "README.md").write_text(sentinel, encoding="utf-8")
        result = scaffold_project(target, force=True)
        # Only the rail files that actually pre-existed are in clobbered.
        # .gitignore is fresh (scaffold-owned); AGENTS.md and README.md are the
        # two the user had, and both are recorded.
        clobbered_rel = {p.relative_to(target) for p in result.clobbered}
        assert clobbered_rel == {Path("AGENTS.md"), Path("README.md")}

    def test_notes_md_refused_when_user_owned(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir(parents=True, exist_ok=True)
        (target / "NOTES.md").write_text("# user notes", encoding="utf-8")
        with pytest.raises(ScaffoldError) as exc_info:
            scaffold_project(target)
        assert "NOTES.md" in str(exc_info.value)
        assert (target / "NOTES.md").read_text(encoding="utf-8") == "# user notes"

    def test_bare_does_not_count_notes_md_as_scaffold_owned(self, tmp_path: Path) -> None:
        """With --bare, NOTES.md is not in files_this_run, so an existing NOTES.md must still be guarded."""
        target = tmp_path / "wiki"
        target.mkdir(parents=True, exist_ok=True)
        (target / "NOTES.md").write_text("# user notes", encoding="utf-8")
        with pytest.raises(ScaffoldError):
            scaffold_project(target, bare=True)
        assert (target / "NOTES.md").read_text(encoding="utf-8") == "# user notes"

    def test_rail_files_constant_lists_three_owners(self) -> None:
        """RAIL_FILES is the single source of truth for what counts as a tracked rail file."""
        assert RAIL_FILES == (".gitignore", "AGENTS.md", "README.md")


class TestPublicApi:
    def test_project_markers_constant(self) -> None:
        assert "kompend.toml" in PROJECT_MARKERS
        assert ".kompend/config.toml" in PROJECT_MARKERS

    def test_authored_templates_constant(self) -> None:
        assert "NOTES.md" in AUTHORED_TEMPLATES

    def test_find_existing_project_returns_none_for_empty_dir(self, tmp_path: Path) -> None:
        assert find_existing_project(tmp_path) is None

    def test_find_existing_project_detects_kompend_toml(self, tmp_path: Path) -> None:
        marker = tmp_path / "kompend.toml"
        marker.write_text("# existing", encoding="utf-8")
        assert find_existing_project(tmp_path) == marker

    def test_find_existing_project_detects_config_toml(self, tmp_path: Path) -> None:
        (tmp_path / KOMP_DIR).mkdir()
        marker = tmp_path / KOMP_DIR / "config.toml"
        marker.write_text("# existing", encoding="utf-8")
        assert find_existing_project(tmp_path) == marker
        assert "README.md" in AUTHORED_TEMPLATES


def _git_available() -> bool:
    return shutil.which("git") is not None
