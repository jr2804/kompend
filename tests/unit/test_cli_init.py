"""CLI integration tests for the ``kompend init`` command."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kompend.cli.app import app
from kompend.core.paths import CACHE_SUBDIR, KOMP_DIR

runner = CliRunner()


class TestInitCommand:
    def test_creates_wiki_in_target_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "my-wiki"
        result = runner.invoke(app, ["init", str(target)])
        assert result.exit_code == 0, result.stdout
        assert (target / "sources").is_dir()
        assert (target / KOMP_DIR / "config.toml").is_file()
        assert "Created wiki project" in result.stdout

    def test_creates_in_cwd_when_no_target(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert (tmp_path / "sources").is_dir()
        assert (tmp_path / KOMP_DIR / "config.toml").is_file()

    def test_bare_flag_skips_authored_files(self, tmp_path: Path) -> None:
        target = tmp_path / "bare-wiki"
        result = runner.invoke(app, ["init", "--bare", str(target)])
        assert result.exit_code == 0
        assert not (target / "NOTES.md").exists()
        assert not (target / "README.md").exists()
        assert "NOTES.md" in result.stdout
        assert "README.md" in result.stdout

    def test_existing_project_exits_gracefully(self, tmp_path: Path) -> None:
        target = tmp_path / "existing"
        target.mkdir()
        (target / "kompend.toml").write_text("# existing", encoding="utf-8")
        result = runner.invoke(app, ["init", str(target)])
        assert result.exit_code == 0
        combined = result.stdout + (result.stderr or "")
        assert "existing kompend project" in combined
        assert "--force" in combined

    def test_force_flag_overwrites(self, tmp_path: Path) -> None:
        target = tmp_path / "force-wiki"
        target.mkdir()
        (target / KOMP_DIR).mkdir()
        (target / KOMP_DIR / "config.toml").write_text("# old", encoding="utf-8")
        result = runner.invoke(app, ["init", "--force", str(target)])
        assert result.exit_code == 0, result.stdout
        assert (target / KOMP_DIR / "config.toml").read_text(encoding="utf-8") != "# old"

    def test_target_is_file_fails(self, tmp_path: Path) -> None:
        file = tmp_path / "not-a-dir"
        file.write_text("file", encoding="utf-8")
        result = runner.invoke(app, ["init", str(file)])
        assert result.exit_code == 1

    def test_refuses_to_clobber_tracked_rail_files(self, tmp_path: Path) -> None:
        """A pre-existing AGENTS.md / .gitignore / README.md is not a scaffold marker; `kompend init` must refuse, not overwrite."""
        target = tmp_path / "my-wiki"
        target.mkdir()
        sentinel = "# tracked repository file -- do not overwrite"
        (target / ".gitignore").write_text(sentinel, encoding="utf-8")
        (target / "AGENTS.md").write_text(sentinel, encoding="utf-8")
        (target / "README.md").write_text(sentinel, encoding="utf-8")
        result = runner.invoke(app, ["init", str(target)])
        assert result.exit_code == 1
        combined = result.stdout + (result.stderr or "")
        # Every offender must be named in a single error message.
        assert ".gitignore" in combined
        assert "AGENTS.md" in combined
        assert "README.md" in combined
        assert "--force" in combined
        # None of the originals were modified.
        assert (target / ".gitignore").read_text(encoding="utf-8") == sentinel
        assert (target / "AGENTS.md").read_text(encoding="utf-8") == sentinel
        assert (target / "README.md").read_text(encoding="utf-8") == sentinel

    def test_force_flag_clobbers_tracked_rail_files(self, tmp_path: Path) -> None:
        """--force is the explicit, opt-in acknowledgement that tracked content will be overwritten."""
        target = tmp_path / "my-wiki"
        target.mkdir()
        (target / "AGENTS.md").write_text("# user content", encoding="utf-8")
        result = runner.invoke(app, ["init", "--force", str(target)])
        assert result.exit_code == 0, result.stdout
        combined = result.stdout + (result.stderr or "")
        # Scaffold overwrites it and warns.
        assert (target / "AGENTS.md").read_text(encoding="utf-8") != "# user content"
        assert "AGENTS.md" in combined

    def test_git_flag_runs_git_init(self, tmp_path: Path) -> None:
        if not _git_available():
            pytest.skip("git not on PATH")
        target = tmp_path / "git-wiki"
        result = runner.invoke(app, ["init", "--git", str(target)])
        assert result.exit_code == 0, result.stdout
        assert (target / ".git").is_dir()
        assert "git init" in result.stdout

    def test_help_lists_flags(self) -> None:
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "--bare" in result.stdout
        assert "--force" in result.stdout
        assert "--git" in result.stdout
        assert "--keep-cache" in result.stdout
        assert "--wiki-dir" in result.stdout

    def test_keep_cache_preserves_cache(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KOMP_DIR).mkdir()
        (target / KOMP_DIR / "config.toml").write_text("# old", encoding="utf-8")
        cache_dir = target / KOMP_DIR / CACHE_SUBDIR
        cache_dir.mkdir()
        (cache_dir / "doc.json").write_text("{}", encoding="utf-8")
        (target / "stale.md").write_text("old", encoding="utf-8")

        result = runner.invoke(app, ["init", "--force", "--keep-cache", str(target)])
        assert result.exit_code == 0, result.stdout
        assert (cache_dir / "doc.json").is_file()
        assert not (target / "stale.md").exists()

    def test_wiki_dir_option_scaffolds_into_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "via-flag"
        result = runner.invoke(app, ["init", "--wiki-dir", str(target)])
        assert result.exit_code == 0, result.stdout
        assert (target / "sources").is_dir()
        assert (target / KOMP_DIR / "config.toml").is_file()
        assert "Created wiki project" in result.stdout

    def test_wiki_dir_short_flag(self, tmp_path: Path) -> None:
        target = tmp_path / "short-flag"
        result = runner.invoke(app, ["init", "-w", str(target)])
        assert result.exit_code == 0, result.stdout
        assert (target / KOMP_DIR / "config.toml").is_file()

    def test_wiki_dir_matching_target_is_accepted(self, tmp_path: Path) -> None:
        target = tmp_path / "same"
        result = runner.invoke(app, ["init", str(target), "--wiki-dir", str(target)])
        assert result.exit_code == 0, result.stdout
        assert (target / KOMP_DIR / "config.toml").is_file()

    def test_wiki_dir_conflicting_target_fails(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["init", str(tmp_path / "a"), "--wiki-dir", str(tmp_path / "b")])
        assert result.exit_code == 1
        combined = result.stdout + (result.stderr or "")
        assert "either as TARGET or via --wiki-dir" in combined


def _git_available() -> bool:
    return shutil.which("git") is not None
