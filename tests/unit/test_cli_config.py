"""CLI integration tests for the ``knox config`` subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from knox.cli.app import app
from knox.core.paths import KNOX_DIR

runner = CliRunner()


class TestConfigGet:
    def test_get_known_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["config", "get", "compile.budget.chunk_size_min"])
        assert result.exit_code == 0
        assert result.stdout.strip() == "2048"

    def test_get_unknown_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["config", "get", "compile.unknown"])
        assert result.exit_code == 1
        assert "unknown config key" in (result.stdout + (result.stderr or ""))

    def test_get_respects_project_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / KNOX_DIR).mkdir()
        (tmp_path / KNOX_DIR / "config.toml").write_text("[compile.budget]\nchunk_size_min = 2048\n")
        result = runner.invoke(app, ["config", "get", "compile.budget.chunk_size_min"])
        assert result.exit_code == 0
        assert result.stdout.strip() == "2048"


class TestConfigSet:
    def test_set_integer_in_project_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        self._init_vault(tmp_path)
        result = runner.invoke(app, ["config", "set", "compile.budget.chunk_size_min", "4096"])
        assert result.exit_code == 0, result.stdout
        assert "set compile.budget.chunk_size_min = 4096" in result.stdout
        assert (tmp_path / KNOX_DIR / "config.toml").read_text(encoding="utf-8") == "[compile.budget]\nchunk_size_min = 4096\n"

    def test_set_boolean(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        self._init_vault(tmp_path)
        result = runner.invoke(app, ["config", "set", "compile.no_cache", "true"])
        assert result.exit_code == 0
        text = (tmp_path / KNOX_DIR / "config.toml").read_text(encoding="utf-8")
        assert "no_cache = true" in text

    def test_set_list(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        self._init_vault(tmp_path)
        result = runner.invoke(app, ["config", "set", "compile.llm.fallback_chain", "a, b, c"])
        assert result.exit_code == 0
        text = (tmp_path / KNOX_DIR / "config.toml").read_text(encoding="utf-8")
        assert "fallback_chain = [" in text
        assert '"a"' in text
        assert '"b"' in text
        assert '"c"' in text

    def test_set_unknown_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["config", "set", "compile.unknown", "x"])
        assert result.exit_code == 1
        assert "unknown config key" in (result.stdout + (result.stderr or ""))

    def test_set_global_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        result = runner.invoke(app, ["config", "set", "--global", "compile.budget.chunk_size_min", "4096"])
        assert result.exit_code == 0, result.stdout
        assert (tmp_path / "knox" / "config.toml").read_text(encoding="utf-8") == "[compile.budget]\nchunk_size_min = 4096\n"

    def test_set_creates_backup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        config = tmp_path / KNOX_DIR / "config.toml"
        config.parent.mkdir()
        config.write_text("[compile.budget]\nchunk_size_min = 4096\n")
        result = runner.invoke(app, ["config", "set", "compile.budget.chunk_size_min", "6144"])
        assert result.exit_code == 0
        backup = config.with_suffix(config.suffix + ".bak")
        assert backup.is_file()
        assert "chunk_size_min = 4096" in backup.read_text(encoding="utf-8")

    def _init_vault(self, tmp_path: Path) -> Path:
        """Create .knox/config.toml so config set targets a valid vault."""
        cfg_dir = tmp_path / KNOX_DIR
        cfg_dir.mkdir()
        (cfg_dir / "config.toml").write_text("")
        return cfg_dir


class TestConfigList:
    def test_list_includes_default_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["config", "list"])
        assert result.exit_code == 0
        assert "compile.budget.chunk_size_min = 2048" in result.stdout
        assert "# default" in result.stdout

    def test_list_shows_project_source(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / KNOX_DIR).mkdir()
        (tmp_path / KNOX_DIR / "config.toml").write_text("[compile.budget]\nchunk_size_min = 4096\n")
        result = runner.invoke(app, ["config", "list"])
        assert result.exit_code == 0
        assert "compile.budget.chunk_size_min = 4096" in result.stdout
        assert "# project" in result.stdout


class TestConfigPath:
    def test_path_project_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tmp_path = tmp_path.resolve()
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0
        assert result.stdout.strip() == str(tmp_path / KNOX_DIR / "config.toml")

    def test_path_global(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        result = runner.invoke(app, ["config", "path", "--global"])
        assert result.exit_code == 0
        assert result.stdout.strip() == str(tmp_path / "knox" / "config.toml")

    def test_path_prefers_existing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tmp_path = tmp_path.resolve()
        monkeypatch.chdir(tmp_path)
        (tmp_path / "knox.toml").write_text("[compile]\nchunk_size = 128\n")
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0
        assert result.stdout.strip() == str(tmp_path / "knox.toml")


def test_config_list_redacts_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / KNOX_DIR).mkdir()
    (tmp_path / KNOX_DIR / "config.toml").write_text('[providers.omniroute]\nbase_url = "https://x/v1"\napi_key = "sk-secret"\n')
    result = runner.invoke(app, ["config", "list"])
    assert result.exit_code == 0
    assert "providers.omniroute.api_key = ***" in result.stdout
    assert "sk-secret" not in result.stdout
