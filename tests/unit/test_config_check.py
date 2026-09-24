"""Tests for ``knox config check`` — diff a user TOML against schema defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from knox.cli.app import app
from knox.core.config import (
    STATUS_ERROR,
    STATUS_MODIFIED,
    STATUS_OK,
    STATUS_REMOVED,
    ConfigCheckReport,
    check_config_file,
    check_config_files,
)

runner = CliRunner()


# ── Core helper tests ───────────────────────────────────────────────────────


class TestCheckConfigFile:
    def test_ok_on_default_values(self, tmp_path: pytest.TempPathFactory) -> Path:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text("", encoding="utf-8")
        report = check_config_file(cfg)
        assert isinstance(report, ConfigCheckReport)
        assert report.path == cfg
        # An empty TOML is fully defaulted — every reportable row is ok.
        for entry in report.entries:
            assert entry.status == STATUS_OK, entry.key

    def test_modified_entry(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(
            "[compile]\nworkers = 8\n",
            encoding="utf-8",
        )
        report = check_config_file(cfg)
        modified = [e for e in report.entries if e.status == STATUS_MODIFIED]
        assert len(modified) == 1
        assert modified[0].key == "compile.workers"
        assert modified[0].current == 8
        # The default is 1 (or whatever the schema says); just verify it's something else.
        assert modified[0].default != 8

    def test_explicit_value_equals_default(self, tmp_path: Path) -> None:
        """A key set to its literal default value is still 'ok' (not modified)."""
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(
            "[compile]\nworkers = 1\n",
            encoding="utf-8",
        )
        report = check_config_file(cfg)
        workers = next(e for e in report.entries if e.key == "compile.workers")
        assert workers.status == STATUS_OK
        assert workers.set_in_toml is True

    def test_unknown_key_reported_as_error(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(
            "[compile]\nlegacy_removed_field = 'oops'\n",
            encoding="utf-8",
        )
        report = check_config_file(cfg)
        errors = [e for e in report.entries if e.status == STATUS_ERROR]
        assert any(e.key == "compile.legacy_removed_field" for e in errors)
        assert any("unknown" in (e.message or "").lower() for e in errors)

    def test_missing_file(self, tmp_path: Path) -> None:
        cfg = tmp_path / "does-not-exist.toml"
        report = check_config_file(cfg)
        assert report.has_errors
        assert all(e.status == STATUS_ERROR for e in report.entries)
        assert all("does not exist" in (e.message or "") for e in report.entries)

    def test_removed_key_reported(self, tmp_path: Path) -> None:
        """Keys retired from the schema (REMOVED_KEYS) are reported as removed,
        not as unknown-key errors, so old vaults get a clear cleanup hint.
        """
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(
            "[compile.llm]\nrole_models = { compile = 'openai/gpt-4o-mini' }\n",
            encoding="utf-8",
        )
        report = check_config_file(cfg)
        removed = next(e for e in report.entries if e.key == "compile.llm.role_models")
        assert removed.status == STATUS_REMOVED
        assert removed.set_in_toml is True
        assert removed.current == {"compile": "openai/gpt-4o-mini"}
        assert report.removed_count >= 1
        assert not report.has_errors

    def test_nested_model_leaves_are_reachable(self, tmp_path: Path) -> None:
        """Optional nested model leaves (e.g. compile.llm.rlm.enable) MUST appear
        in the diff once the user touches the parent section, even when the
        schema default holds None at that path.
        """
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(
            "[compile.llm.rlm]\nenable = true\n",
            encoding="utf-8",
        )
        report = check_config_file(cfg)
        enable = next(
            (e for e in report.entries if e.key == "compile.llm.rlm.enable"),
            None,
        )
        assert enable is not None
        assert enable.set_in_toml is True
        assert enable.current is True
        # schema default for enable is False → user modified it.
        assert enable.default is False
        assert enable.status == STATUS_MODIFIED

    def test_nested_model_leaves_skipped_when_parent_absent(self, tmp_path: Path) -> None:
        """If the user did not touch compile.llm.rlm at all, we don't surface
        those leaves as 'ok' rows — keeping output focused on what the user wrote.
        """
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(
            "[compile]\nworkers = 4\n",
            encoding="utf-8",
        )
        report = check_config_file(cfg)
        keys = [e.key for e in report.entries if e.key.startswith("compile.llm.rlm")]
        assert keys == []

    def test_list_field_with_default_factory(self, tmp_path: Path) -> None:
        """Pydantic field with ``default_factory=list`` must compare as equal
        to ``[]``, not as ``PydanticUndefined``.
        """
        cfg = tmp_path / "cfg.toml"
        # compile.source_tags has default_factory=list → default == []
        cfg.write_text(
            "[compile]\nsource_tags = []\n",
            encoding="utf-8",
        )
        report = check_config_file(cfg)
        tags = next(e for e in report.entries if e.key == "compile.source_tags")
        assert tags.status == STATUS_OK

    def test_to_dict_round_trips(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(
            "[compile]\nworkers = 8\n",
            encoding="utf-8",
        )
        report = check_config_file(cfg)
        payload = report.to_dict()
        assert payload["ok"] == report.ok_count
        assert payload["modified"] == report.modified_count
        assert payload["error"] == report.error_count
        assert payload["path"] == str(cfg)
        # JSON-serialisable
        json.dumps(payload)

    def test_entry_to_dict(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(
            "[compile]\nlegacy_x = 'oops'\n",
            encoding="utf-8",
        )
        report = check_config_file(cfg)
        errors = [e for e in report.entries if e.status == STATUS_ERROR]
        assert errors
        entry = errors[0]
        d = entry.to_dict()
        assert d["key"] == entry.key
        assert d["status"] == STATUS_ERROR
        assert d["default"] is None
        assert d["current"] == "oops"
        assert "message" in d


class TestCheckConfigFiles:
    def test_merging_two_files(self, tmp_path: Path) -> None:
        base = tmp_path / "base.toml"
        base.write_text(
            "[compile]\nworkers = 4\nno_cache = true\n",
            encoding="utf-8",
        )
        override = tmp_path / "local.toml"
        override.write_text(
            "[compile]\nworkers = 8\n",
            encoding="utf-8",
        )
        report = check_config_files([base, override])
        workers = next(e for e in report.entries if e.key == "compile.workers")
        assert workers.status == STATUS_MODIFIED
        assert workers.current == 8
        no_cache = next(e for e in report.entries if e.key == "compile.no_cache")
        assert no_cache.set_in_toml is True

    def test_missing_file_during_merge(self, tmp_path: Path) -> None:
        report = check_config_files([tmp_path / "ghost.toml"])
        assert report.has_errors
        assert any("does not exist" in (e.message or "") for e in report.entries)


# ── CLI command tests ───────────────────────────────────────────────────────


class TestConfigCheckCli:
    def test_single_file_text_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(
            "[compile]\nworkers = 8\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        # Use --no-merge so even a single path produces a per-file report.
        result = runner.invoke(app, ["config", "check", str(cfg), "--no-merge"])
        assert result.exit_code == 1, result.output
        assert "MODIFIED" in result.output
        assert "compile.workers" in result.output

    def test_single_file_json(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(
            "[compile]\nworkers = 8\n",
            encoding="utf-8",
        )
        result = runner.invoke(app, ["config", "check", str(cfg), "--json"])
        payload = json.loads(result.stdout)
        assert isinstance(payload, list)
        assert payload
        report = payload[0]
        assert report["modified"] >= 1
        modified_keys = {e["key"] for e in report["entries"] if e["status"] == "modified"}
        assert "compile.workers" in modified_keys

    def test_error_unknown_key_exits_2(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad.toml"
        cfg.write_text(
            "[compile]\nlegacy_removed = 'x'\n",
            encoding="utf-8",
        )
        result = runner.invoke(app, ["config", "check", str(cfg), "--no-merge"])
        assert result.exit_code == 2
        assert "unknown" in result.output.lower()

    def test_no_modifications_no_errors_exits_0(self, tmp_path: Path) -> None:
        cfg = tmp_path / "empty.toml"
        cfg.write_text("", encoding="utf-8")
        result = runner.invoke(app, ["config", "check", str(cfg), "--no-merge"])
        assert result.exit_code == 0, result.output

    def test_merged_two_files(self, tmp_path: Path) -> None:
        a = tmp_path / "a.toml"
        a.write_text("[compile]\nworkers = 4\n", encoding="utf-8")
        b = tmp_path / "b.toml"
        b.write_text("[compile]\nworkers = 8\n", encoding="utf-8")
        result = runner.invoke(app, ["config", "check", str(a), str(b)])
        # Exit 1: only modified (no errors).
        assert result.exit_code == 1
        assert "MODIFIED" in result.output
        assert "merged" in result.output.lower()

    def test_no_merge_two_files(self, tmp_path: Path) -> None:
        a = tmp_path / "a.toml"
        a.write_text("", encoding="utf-8")
        b = tmp_path / "b.toml"
        b.write_text("[compile]\nworkers = 8\n", encoding="utf-8")
        result = runner.invoke(
            app,
            ["config", "check", str(a), str(b), "--no-merge"],
        )
        assert result.exit_code == 1
        # Two report headers (one per file).
        assert result.output.count("config check") == 2

    def test_no_args_discovers_project_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """`knox config check` with no paths uses the discovered project config."""
        (tmp_path / "knox.toml").write_text(
            "[compile]\nworkers = 8\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["config", "check"])
        assert result.exit_code == 1, result.output
        assert "MODIFIED" in result.output
        assert "compile.workers" in result.output

    def test_no_args_empty_vault_reports_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a config file the vault is all-defaults — clean exit, no errors."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["config", "check"])
        assert result.exit_code == 0, result.output
        assert "No config file found" in result.output

    def test_no_args_with_wiki_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """`-w <vault>` with no paths checks that vault's config."""
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "knox.toml").write_text(
            "[compile]\nworkers = 8\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["-w", str(vault), "config", "check"])
        assert result.exit_code == 1, result.output
        assert "MODIFIED" in result.output
        assert "compile.workers" in result.output
