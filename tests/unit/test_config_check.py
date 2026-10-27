"""Tests for ``kompend config check`` — diff a user TOML against schema defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kompend.cli.app import app
from kompend.core.config import (
    STATUS_DEPRECATED,
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
    def test_ok_on_default_values(self, tmp_path: Path) -> None:
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

    def test_deprecated_alias_is_not_an_unknown_key_error(self, tmp_path: Path) -> None:
        """A renamed key is reported as deprecated, not as an unknown-key error.

        ``compile.llm.context_window_cap`` left the schema but is still
        migrated at resolve time, so ``config check`` must say "deprecated —
        use X" (exit 1) rather than fail it as unknown (exit 2).
        """
        cfg = tmp_path / "cfg.toml"
        cfg.write_text("[compile.llm]\ncontext_window_cap = 65536\n", encoding="utf-8")
        report = check_config_file(cfg)

        entry = next(e for e in report.entries if e.key == "compile.llm.context_window_cap")
        assert entry.status == STATUS_DEPRECATED
        assert entry.current == 65536
        assert entry.message is not None
        assert "compile.llm.context_window" in entry.message
        assert report.deprecated_count >= 1
        assert not report.has_errors
        assert not any(e.status == STATUS_ERROR for e in report.entries)

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


class TestUserNamedNamespaces:
    """A ``dict[str, Model]`` field is a namespace the user names, not a typo.

    ``providers``, ``compile.skills`` and ``providers.<name>.models`` hold models
    keyed by name; the checker used to compare their leaves against a fixed key
    list, so *every* provider key was reported as unknown — 18 false errors on a
    real global config.
    """

    def test_provider_keys_are_not_unknown(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(
            '[providers.local]\nbase_url = "http://localhost:8001/v1"\napi_type = "anthropic"\ncontext_window = 32768\n',
            encoding="utf-8",
        )
        report = check_config_file(cfg)

        assert report.error_count == 0
        by_key = {e.key: e for e in report.entries}
        assert by_key["providers.local.base_url"].status == STATUS_MODIFIED
        assert by_key["providers.local.base_url"].current == "http://localhost:8001/v1"
        # The default comes from the pattern's field, not from one provider's row.
        assert by_key["providers.local.api_type"].default == "openai-chat"
        assert by_key["providers.local.context_window"].default is None

    def test_unknown_provider_field_is_still_an_error(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text('[providers.local]\nbase_uri = "http://x/v1"\n', encoding="utf-8")
        report = check_config_file(cfg)

        assert report.error_count == 1
        broken = [e for e in report.entries if e.status == STATUS_ERROR]
        assert broken[0].key == "providers.local.base_uri"
        assert "unknown config key" in (broken[0].message or "")

    def test_misspelled_section_is_still_an_error(self, tmp_path: Path) -> None:
        """The namespace fix must not blind the check to a typo'd section name."""
        cfg = tmp_path / "cfg.toml"
        cfg.write_text('[provider.local]\nbase_url = "http://x/v1"\n', encoding="utf-8")
        report = check_config_file(cfg)

        assert report.error_count >= 1
        assert any(e.key.startswith("provider.") for e in report.entries if e.status == STATUS_ERROR)

    def test_nested_namespace_with_a_dotted_name(self, tmp_path: Path) -> None:
        """``providers.<p>.models.<m>`` where the model name itself has dots.

        ``flat_keys()`` joins with ``.``, so a model called ``hf.co/qwen3.6``
        arrives as three extra segments; ``*`` has to span all of them.
        """
        cfg = tmp_path / "cfg.toml"
        cfg.write_text(
            '[providers.ollama.models."hf.co/qwen3.6"]\ncontext_window = 262144\n',
            encoding="utf-8",
        )
        report = check_config_file(cfg)

        assert report.error_count == 0
        entry = next(e for e in report.entries if "qwen3.6" in e.key)
        assert entry.status == STATUS_MODIFIED
        assert entry.current == 262144

    def test_skills_namespace(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.toml"
        cfg.write_text('[compile.skills."domain-terms"]\ninstructions = "use ISO terms"\n', encoding="utf-8")
        report = check_config_file(cfg)

        assert report.error_count == 0
        entry = next(e for e in report.entries if e.key == "compile.skills.domain-terms.instructions")
        assert entry.status == STATUS_MODIFIED

    def test_free_form_table_keys_are_user_data(self, tmp_path: Path) -> None:
        """``categories`` is ``dict[str, str]``: its keys are content, not config keys."""
        cfg = tmp_path / "cfg.toml"
        cfg.write_text('[compile.extraction.categories]\nfigure = "figure"\ntable = "table"\n', encoding="utf-8")
        report = check_config_file(cfg)

        assert report.error_count == 0
        assert not any("categories." in e.key and e.status == STATUS_ERROR for e in report.entries)

    def test_file_that_cannot_be_loaded_lists_literal_keys(self, tmp_path: Path) -> None:
        """Pattern namespaces are not listed as missing keys — a pattern is not a key."""
        report = check_config_file(tmp_path / "nope.toml")

        assert report.error_count > 0
        assert not any("*" in e.key for e in report.entries)


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
        """`kompend config check` with no paths uses the discovered project config."""
        (tmp_path / "kompend.toml").write_text(
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
        (vault / "kompend.toml").write_text(
            "[compile]\nworkers = 8\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["-w", str(vault), "config", "check"])
        assert result.exit_code == 1, result.output
        assert "MODIFIED" in result.output
        assert "compile.workers" in result.output


class TestConfigCommandsHonourTheFlag:
    """The root ``--config`` flag must reach every config-introspection command."""

    def test_get_reads_the_flag_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "kompend.toml").write_text("[compile]\nworkers = 8\n", encoding="utf-8")
        explicit = tmp_path / "other.toml"
        explicit.write_text("[compile]\nworkers = 3\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["--config", str(explicit), "-w", str(vault), "config", "get", "compile.workers"])

        assert result.exit_code == 0, result.output
        assert result.output.strip() == "3"

    def test_list_marks_values_from_the_flag_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "kompend.toml").write_text("[compile]\nworkers = 8\n", encoding="utf-8")
        explicit = tmp_path / "other.toml"
        explicit.write_text("[compile]\nworkers = 3\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["--config", str(explicit), "-w", str(vault), "config", "list"])

        assert result.exit_code == 0, result.output
        line = next(ln for ln in result.output.splitlines() if ln.startswith("compile.workers = "))
        assert line.startswith("compile.workers = 3")

    def test_check_uses_the_flag_file_without_a_path_argument(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        explicit = tmp_path / "bad.toml"
        explicit.write_text("[compile]\nlegacy_removed = 'x'\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["--config", str(explicit), "config", "check"])

        assert result.exit_code == 2, result.output
        assert "bad.toml" in result.output
        assert "unknown" in result.output.lower()
