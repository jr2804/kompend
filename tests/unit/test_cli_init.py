"""CLI integration tests for the ``knox init`` command."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from knox.cli.app import app
from knox.core.paths import CACHE_SUBDIR, KNOX_DIR

runner = CliRunner()


class TestInitCommand:
    def test_creates_wiki_in_target_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "my-wiki"
        result = runner.invoke(app, ["init", str(target)])
        assert result.exit_code == 0, result.stdout
        assert (target / "sources").is_dir()
        assert (target / KNOX_DIR / "config.toml").is_file()
        assert "Created wiki project" in result.stdout

    def test_creates_in_cwd_when_no_target(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert (tmp_path / "sources").is_dir()
        assert (tmp_path / KNOX_DIR / "config.toml").is_file()

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
        (target / "knox.toml").write_text("# existing", encoding="utf-8")
        result = runner.invoke(app, ["init", str(target)])
        assert result.exit_code == 0
        combined = result.stdout + (result.stderr or "")
        assert "existing KNOX project" in combined
        assert "--force" in combined

    def test_force_flag_overwrites(self, tmp_path: Path) -> None:
        target = tmp_path / "force-wiki"
        target.mkdir()
        (target / KNOX_DIR).mkdir()
        (target / KNOX_DIR / "config.toml").write_text("# old", encoding="utf-8")
        result = runner.invoke(app, ["init", "--force", str(target)])
        assert result.exit_code == 0, result.stdout
        assert (target / KNOX_DIR / "config.toml").read_text(encoding="utf-8") != "# old"

    def test_target_path_is_file_fails(self, tmp_path: Path) -> None:
        file_path = tmp_path / "not-a-dir"
        file_path.write_text("file", encoding="utf-8")
        result = runner.invoke(app, ["init", str(file_path)])
        assert result.exit_code == 1

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

    def test_keep_cache_preserves_cache(self, tmp_path: Path) -> None:
        target = tmp_path / "wiki"
        target.mkdir()
        (target / KNOX_DIR).mkdir()
        (target / KNOX_DIR / "config.toml").write_text("# old", encoding="utf-8")
        cache_dir = target / KNOX_DIR / CACHE_SUBDIR
        cache_dir.mkdir()
        (cache_dir / "doc.json").write_text("{}", encoding="utf-8")
        (target / "stale.md").write_text("old", encoding="utf-8")

        result = runner.invoke(app, ["init", "--force", "--keep-cache", str(target)])
        assert result.exit_code == 0, result.stdout
        assert (cache_dir / "doc.json").is_file()
        assert not (target / "stale.md").exists()


def _git_available() -> bool:
    return shutil.which("git") is not None
