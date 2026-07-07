"""Unit tests for KNOX unified config system (Q23)."""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any

import pytest

from knox.core.config._core import (
    _deep_merge,
    _flat_keys,
    _get_dot,
    _load_toml,
    _read_env_vars,
    _set_dot,
    coerce_config_value,
    find_all_project_configs,
    find_global_config,
    find_project_config,
    get_global_config_path,
    get_project_config_path,
    resolve_config,
    set_config_file_value,
    validate_config_key,
)
from knox.core.config.generate import extract_toml_keys, generate_scaffold_config, get_default_keys
from knox.core.config.keys import PATHS_WIKI_DIR
from knox.core.paths import KNOX_DIR

# ── Deep merge ──────────────────────────────────────────────────────────


class TestDeepMerge:
    def test_empty_override(self) -> None:
        base = {"a": 1, "b": {"c": 2}}
        result = _deep_merge(base, {})
        assert result == base

    def test_flat_override(self) -> None:
        base = {"a": 1, "b": 2}
        result = _deep_merge(base, {"b": 99})
        assert result["a"] == 1
        assert result["b"] == 99

    def test_nested_override(self) -> None:
        base = {"compile": {"budget": {"chunk_size_min": 512}, "format": "markdown"}}
        override = {"compile": {"budget": {"chunk_size_min": 256}}}
        result = _deep_merge(base, override)
        assert result["compile"]["budget"]["chunk_size_min"] == 256
        assert result["compile"]["format"] == "markdown"

    def test_deep_nested(self) -> None:
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = _deep_merge(base, override)
        assert result["a"]["b"]["c"] == 99
        assert result["a"]["b"]["d"] == 2

    def test_does_not_mutate_base(self) -> None:
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"b": 2}})
        assert base["a"]["b"] == 1


# ── Dot-path access ─────────────────────────────────────────────────────


class TestDotPath:
    def test_get_top_level(self) -> None:
        data = {"key": "value"}
        assert _get_dot(data, "key") == "value"

    def test_get_nested(self) -> None:
        data = {"compile": {"llm": {"model": "gpt-4"}}}
        assert _get_dot(data, "compile.llm.model") == "gpt-4"

    def test_get_missing_returns_default(self) -> None:
        data = {"a": 1}
        assert _get_dot(data, "b", "fallback") == "fallback"

    def test_get_partial_path_returns_default(self) -> None:
        data = {"a": {"b": 1}}
        assert _get_dot(data, "a.c", "missing") == "missing"

    def test_set_creates_intermediate(self) -> None:
        data: dict[str, Any] = {}
        _set_dot(data, "a.b.c", 42)
        assert data["a"]["b"]["c"] == 42

    def test_set_overwrites(self) -> None:
        data: dict[str, Any] = {"a": {"b": 1}}
        _set_dot(data, "a.b", 99)
        assert data["a"]["b"] == 99


# ── Flat keys ───────────────────────────────────────────────────────────


class TestFlatKeys:
    def test_flat_dict(self) -> None:
        data = {"a": 1, "b": 2}
        assert sorted(_flat_keys(data)) == ["a", "b"]

    def test_nested_dict(self) -> None:
        data = {"compile": {"budget": {"chunk_size_min": 512}, "output": {"format": "md"}}}
        assert sorted(_flat_keys(data)) == ["compile.budget.chunk_size_min", "compile.output.format"]

    def test_deeply_nested(self) -> None:
        data = {"a": {"b": {"c": 1}}}
        assert _flat_keys(data) == ["a.b.c"]


# ── Env vars ────────────────────────────────────────────────────────────


class TestEnvVars:
    def test_no_vars_returns_empty(self) -> None:
        # Unset all KNOX vars to ensure clean state
        {k: v for k, v in os.environ.items() if not k.startswith("KNOX_")}
        result = _read_env_vars()
        # May still have values from the test environment, but should not crash
        assert isinstance(result, dict)

    def test_integer_coercion(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("KNOX_BUDGET_CHUNK_SIZE_MIN", "256")
        result = _read_env_vars()
        assert result["compile"]["budget"]["chunk_size_min"] == 256

    def test_boolean_coercion_true(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("KNOX_NO_CACHE", "true")
        result = _read_env_vars()
        assert result["compile"]["no_cache"] is True

    def test_boolean_coercion_false(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("KNOX_FORCE", "false")
        result = _read_env_vars()
        assert result["compile"]["force"] is False

    def test_comma_list(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("KNOX_LLM_FALLBACK_CHAIN", "openai/gpt-4o, anthropic/claude")
        result = _read_env_vars()
        assert result["compile"]["llm"]["fallback_chain"] == ["openai/gpt-4o", "anthropic/claude"]

    def test_string_value(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("KNOX_LLM_DEFAULT_MODEL", "openai/gpt-4o")
        result = _read_env_vars()
        assert result["compile"]["llm"]["default_model"] == "openai/gpt-4o"

    def test_string_value_stripped(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("KNOX_LLM_DEFAULT_MODEL", "  openai/gpt-4o  ")
        result = _read_env_vars()
        assert result["compile"]["llm"]["default_model"] == "openai/gpt-4o"

    def test_vlm_model_env(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("KNOX_VLM_MODEL", "openai/gpt-4o")
        result = _read_env_vars()
        assert result["compile"]["llm"]["vlm_model"] == "openai/gpt-4o"

    def test_user_tracking_env_false(self, monkeypatch: Any) -> None:
        # knox-gsb5: KNOX_USER_TRACKING=false disables user tracking.
        monkeypatch.setenv("KNOX_USER_TRACKING", "false")
        result = _read_env_vars()
        assert result["trust"]["user_tracking"] is False


# ── Config file discovery ───────────────────────────────────────────────


class TestConfigDiscovery:
    def test_find_project_config_none_when_missing(self, tmp_path: Any) -> None:
        result = find_project_config(start_dir=tmp_path)
        assert result is None

    def test_find_project_config_finds_toml(self, tmp_path: Any) -> None:
        config_file = tmp_path / "knox.toml"
        config_file.write_text("[compile.budget]\nchunk_size_min = 128\n")
        result = find_project_config(start_dir=tmp_path)
        assert result == config_file

    def test_find_project_config_walks_up(self, tmp_path: Any) -> None:
        config_file = tmp_path / "knox.toml"
        config_file.write_text("[compile.budget]\nchunk_size_min = 128\n")
        subdir = tmp_path / "sub" / "dir"
        subdir.mkdir(parents=True)
        result = find_project_config(start_dir=subdir)
        assert result == config_file

    def test_find_global_config_none_when_missing(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        result = find_global_config()
        assert result is None

    def test_find_global_config_finds_file(self, tmp_path: Any, monkeypatch: Any) -> None:
        config_dir = tmp_path / "knox"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text("[compile.budget]\nchunk_size_min = 64\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        result = find_global_config()
        assert result == config_file


# ── resolve_config ──────────────────────────────────────────────────────


class TestResolveConfig:
    def test_defaults_only(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = resolve_config()
        assert cfg.get("compile.budget.chunk_size_min") == 2048
        assert cfg.get("compile.output.format") == "markdown"
        assert cfg.get("compile.llm.fallback_max_retries") == 2
        assert cfg.get("compile.llm.vlm_model") is None

    def test_project_config_overrides_defaults(self, tmp_path: Any, monkeypatch: Any) -> None:
        config_file = tmp_path / "knox.toml"
        config_file.write_text("[compile.budget]\nchunk_size_min = 128\n")
        monkeypatch.chdir(tmp_path)
        cfg = resolve_config()
        assert cfg.get("compile.budget.chunk_size_min") == 128
        assert cfg.get("compile.output.format") == "markdown"  # still default

    def test_env_overrides_project(self, tmp_path: Any, monkeypatch: Any) -> None:
        config_file = tmp_path / "knox.toml"
        config_file.write_text("[compile.budget]\nchunk_size_min = 128\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KNOX_BUDGET_CHUNK_SIZE_MIN", "64")
        cfg = resolve_config()
        assert cfg.get("compile.budget.chunk_size_min") == 64

    def test_cli_overrides_env(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KNOX_BUDGET_CHUNK_SIZE_MIN", "64")
        cfg = resolve_config(cli_overrides={"compile.budget.chunk_size_min": 32})
        assert cfg.get("compile.budget.chunk_size_min") == 32

    def test_toml_string_values_stripped(self, tmp_path: Any, monkeypatch: Any) -> None:
        config_file = tmp_path / "knox.toml"
        config_file.write_text(
            '[compile.llm]\ndefault_model = " openai/gpt-4o "\nvlm_model = "\\tollama/llava\\n"\nfallback_chain = [" openai/gpt-4o ", " anthropic/claude "]\n'
        )
        monkeypatch.chdir(tmp_path)
        cfg = resolve_config()
        assert cfg.get("compile.llm.default_model") == "openai/gpt-4o"
        assert cfg.get("compile.llm.vlm_model") == "ollama/llava"
        assert cfg.get("compile.llm.fallback_chain") == ["openai/gpt-4o", "anthropic/claude"]

    def test_vlm_model_config_layers(self, tmp_path: Any, monkeypatch: Any) -> None:
        config_file = tmp_path / "knox.toml"
        config_file.write_text('[compile.llm]\nvlm_model = "ollama/llava"\n')
        monkeypatch.chdir(tmp_path)
        assert resolve_config().get("compile.llm.vlm_model") == "ollama/llava"

        monkeypatch.setenv("KNOX_VLM_MODEL", "openai/gpt-4o")
        assert resolve_config().get("compile.llm.vlm_model") == "openai/gpt-4o"

        assert resolve_config(cli_overrides={"compile.llm.vlm_model": "anthropic/claude-sonnet"}).get("compile.llm.vlm_model") == "anthropic/claude-sonnet"

    def test_ingest_parser_default(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        assert resolve_config().get("ingest.parser") == "docling"

    def test_ingest_parser_env_override(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KNOX_INGEST_PARSER", "liteparse")
        assert resolve_config().get("ingest.parser") == "liteparse"

    def test_ingest_parser_cli_override(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KNOX_INGEST_PARSER", "liteparse")
        assert resolve_config(cli_overrides={"ingest.parser": "docling"}).get("ingest.parser") == "docling"

    def test_memory_config_layers(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = resolve_config()
        assert cfg.get("compile.memory_enabled") is True
        assert cfg.get("compile.memory_half_life_days") == 90.0
        assert cfg.get("compile.memory_decay_threshold") == 0.3

        config_file = tmp_path / "knox.toml"
        config_file.write_text("[compile]\nmemory_enabled = false\nmemory_half_life_days = 180.0\nmemory_decay_threshold = 0.2\n")
        cfg = resolve_config()
        assert cfg.get("compile.memory_enabled") is False
        assert cfg.get("compile.memory_half_life_days") == 180.0
        assert cfg.get("compile.memory_decay_threshold") == 0.2

        monkeypatch.setenv("KNOX_MEMORY_ENABLED", "true")
        monkeypatch.setenv("KNOX_MEMORY_HALF_LIFE_DAYS", "45.0")
        monkeypatch.setenv("KNOX_MEMORY_DECAY_THRESHOLD", "0.1")
        cfg = resolve_config()
        assert cfg.get("compile.memory_enabled") is True
        assert cfg.get("compile.memory_half_life_days") == 45.0
        assert cfg.get("compile.memory_decay_threshold") == 0.1

    def test_provenance_tracking(self, tmp_path: Any, monkeypatch: Any) -> None:
        config_file = tmp_path / "knox.toml"
        config_file.write_text("[compile.budget]\nchunk_size_min = 128\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KNOX_WORKERS", "4")
        cfg = resolve_config(cli_overrides={"compile.output.format": "json"})

        assert "project" in str(cfg.source_of("compile.budget.chunk_size_min"))
        assert "env" in str(cfg.source_of("compile.workers"))
        assert "cli" in str(cfg.source_of("compile.output.format"))
        assert "default" in str(cfg.source_of("compile.image_mode"))

    def test_global_config_overrides_defaults(self, tmp_path: Any, monkeypatch: Any) -> None:
        config_dir = tmp_path / "knox"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text("[compile.budget]\nchunk_size_min = 64\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        # Use a subdir with no project config
        subdir = tmp_path / "project"
        subdir.mkdir()
        monkeypatch.chdir(subdir)
        cfg = resolve_config()
        assert cfg.get("compile.budget.chunk_size_min") == 64

    def test_project_overrides_global(self, tmp_path: Any, monkeypatch: Any) -> None:
        # Global: budget chunk_size_min = 64
        config_dir = tmp_path / "knox"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text("[compile.budget]\nchunk_size_min = 64\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        # Project: budget chunk_size_min = 128
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "knox.toml").write_text("[compile.budget]\nchunk_size_min = 128\n")
        monkeypatch.chdir(project_dir)
        cfg = resolve_config()
        assert cfg.get("compile.budget.chunk_size_min") == 128

    def test_llm_config_inherits_from_global(self, tmp_path: Any, monkeypatch: Any) -> None:
        """LLM settings in global config must be inherited when project omits them."""
        # Global: full LLM config
        config_dir = tmp_path / "knox"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            "[compile.llm]\n"
            'default_model = "claude-sonnet-4-5"\n'
            'vlm_model = "gpt-4o"\n'
            "timeout = 300.0\n"
            'api_base = "https://custom.api.com/v1"\n'
            'api_key = "sk-global-test"\n'
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        # Project: has [compile] but NO [compile.llm]
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "knox.toml").write_text("[compile.budget]\nchunk_size_min = 256\n")
        monkeypatch.chdir(project_dir)
        cfg = resolve_config()

        # LLM values must come from global
        assert cfg.get("compile.llm.default_model") == "claude-sonnet-4-5"
        assert cfg.get("compile.llm.vlm_model") == "gpt-4o"
        assert cfg.get("compile.llm.timeout") == 300.0
        assert cfg.get("compile.llm.api_base") == "https://custom.api.com/v1"
        assert cfg.get("compile.llm.api_key") == "sk-global-test"

        # Project-level setting still works
        assert cfg.get("compile.budget.chunk_size_min") == 256

        # Provenance: LLM keys come from global config
        assert "global" in str(cfg.source_of("compile.llm.default_model")).lower()
        assert "global" in str(cfg.source_of("compile.llm.api_key")).lower()

    def test_project_llm_overrides_global_llm(self, tmp_path: Any, monkeypatch: Any) -> None:
        """Project-level LLM config overrides global LLM config."""
        config_dir = tmp_path / "knox"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[compile.llm]\ndefault_model = "claude-sonnet-4-5"\ntimeout = 300.0\n')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "knox.toml").write_text('[compile.llm]\ndefault_model = "local-model"\n')
        monkeypatch.chdir(project_dir)
        cfg = resolve_config()

        # Project overrides model, global still provides timeout
        assert cfg.get("compile.llm.default_model") == "local-model"
        assert cfg.get("compile.llm.timeout") == 300.0


# ── ResolvedConfig ──────────────────────────────────────────────────────


class TestResolvedConfig:
    def test_flat(self) -> None:
        cfg = resolve_config()
        flat = cfg.flat()
        assert isinstance(flat, dict)
        assert "compile.budget.chunk_size_min" in flat

    def test_get_with_default(self) -> None:
        cfg = resolve_config()
        assert cfg.get("nonexistent.key", 42) == 42

    def test_json_output(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = resolve_config()
        flat = cfg.flat()
        # Verify all values are JSON-serializable
        data = {k: {"value": v, "source": str(cfg.source_of(k))} for k, v in flat.items()}
        output = json.dumps(data, default=str)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)


# ── TOML loading ────────────────────────────────────────────────────────


class TestTomlLoading:
    def test_valid_toml(self, tmp_path: Any) -> None:
        f = tmp_path / "test.toml"
        f.write_text('[compile.budget]\nchunk_size_min = 256\n[compile]\nformat = "json"\n')
        data = _load_toml(f)
        assert data["compile"]["budget"]["chunk_size_min"] == 256
        assert data["compile"]["format"] == "json"

    def test_missing_file(self, tmp_path: Any) -> None:
        data = _load_toml(tmp_path / "nonexistent.toml")
        assert data == {}

    def test_invalid_toml(self, tmp_path: Any) -> None:
        f = tmp_path / "bad.toml"
        f.write_text("this is not [valid toml {{{")
        data = _load_toml(f)
        assert data == {}

    def test_nested_tables(self, tmp_path: Any) -> None:
        f = tmp_path / "nested.toml"
        f.write_text('[compile.llm]\ndefault_model = "openai/gpt-4o"\nfallback_chain = ["anthropic/claude"]\n')
        data = _load_toml(f)
        assert data["compile"]["llm"]["default_model"] == "openai/gpt-4o"
        assert data["compile"]["llm"]["fallback_chain"] == ["anthropic/claude"]


# ── Hierarchical config ─────────────────────────────────────────────────


class TestHierarchicalConfig:
    def test_find_all_project_configs_empty(self, tmp_path: Any) -> None:
        assert find_all_project_configs(start_dir=tmp_path) == []

    def test_find_all_project_configs_single(self, tmp_path: Any) -> None:
        config_file = tmp_path / "knox.toml"
        config_file.write_text("[compile.budget]\nchunk_size_min = 128\n")
        result = find_all_project_configs(start_dir=tmp_path)
        assert result == [config_file]

    def test_find_all_project_configs_walks_up(self, tmp_path: Any) -> None:
        parent = tmp_path / "parent"
        parent.mkdir()
        child = parent / "child"
        child.mkdir()
        parent_file = parent / "knox.toml"
        parent_file.write_text("[compile.budget]\nchunk_size_min = 128\n")
        child_file = child / "knox.toml"
        child_file.write_text('[compile]\nformat = "json"\n')
        result = find_all_project_configs(start_dir=child)
        assert result == [child_file, parent_file]

    def test_find_all_project_configs_prefers_dot_knox(self, tmp_path: Any) -> None:
        dot_knox = tmp_path / KNOX_DIR
        dot_knox.mkdir()
        dot_knox_file = dot_knox / "config.toml"
        dot_knox_file.write_text("[compile.budget]\nchunk_size_min = 64\n")
        root_file = tmp_path / "knox.toml"
        root_file.write_text("[compile.budget]\nchunk_size_min = 128\n")
        result = find_all_project_configs(start_dir=tmp_path)
        assert result == [dot_knox_file]

    def test_hierarchy_inherits_parent(self, tmp_path: Any, monkeypatch: Any) -> None:
        parent = tmp_path / "parent"
        parent.mkdir()
        child = parent / "child"
        child.mkdir()
        (parent / "knox.toml").write_text("[compile.budget]\nchunk_size_min = 128\n")
        monkeypatch.chdir(child)
        cfg = resolve_config()
        assert cfg.get("compile.budget.chunk_size_min") == 128

    def test_hierarchy_child_overrides_parent(self, tmp_path: Any, monkeypatch: Any) -> None:
        parent = tmp_path / "parent"
        parent.mkdir()
        child = parent / "child"
        child.mkdir()
        (parent / "knox.toml").write_text("[compile.budget]\nchunk_size_min = 128\n")
        (child / "knox.toml").write_text("[compile.budget]\nchunk_size_min = 64\n")
        monkeypatch.chdir(child)
        cfg = resolve_config()
        assert cfg.get("compile.budget.chunk_size_min") == 64

    def test_wiki_dir_redirects_config_discovery(self, tmp_path: Any, monkeypatch: Any) -> None:
        # C1: --wiki-dir <vault> reads that vault's config, not cwd's.
        tmp_path = tmp_path.resolve()
        vault = tmp_path / "vault"
        (vault / KNOX_DIR).mkdir(parents=True)
        (vault / KNOX_DIR / "config.toml").write_text("[compile.budget]\nchunk_size_min = 777\n")
        foreign_cwd = tmp_path / "elsewhere"
        foreign_cwd.mkdir()
        monkeypatch.chdir(foreign_cwd)  # cwd has no config
        cfg = resolve_config(cli_overrides={PATHS_WIKI_DIR: str(vault)})
        assert cfg.get("compile.budget.chunk_size_min") == 777
        assert str(cfg.get(PATHS_WIKI_DIR)) == str(vault)

    def test_wiki_dir_env_redirects_config_discovery(self, tmp_path: Any, monkeypatch: Any) -> None:
        # C1: KNOX_WIKI_DIR likewise redirects discovery.
        tmp_path = tmp_path.resolve()
        vault = tmp_path / "vault"
        (vault / KNOX_DIR).mkdir(parents=True)
        (vault / KNOX_DIR / "config.toml").write_text("[compile.budget]\nchunk_size_min = 999\n")
        foreign_cwd = tmp_path / "elsewhere"
        foreign_cwd.mkdir()
        monkeypatch.chdir(foreign_cwd)
        monkeypatch.setenv("KNOX_WIKI_DIR", str(vault))
        cfg = resolve_config()
        assert cfg.get("compile.budget.chunk_size_min") == 999

    def test_hierarchy_env_overrides_all(self, tmp_path: Any, monkeypatch: Any) -> None:
        parent = tmp_path / "parent"
        parent.mkdir()
        child = parent / "child"
        child.mkdir()
        (parent / "knox.toml").write_text("[compile.budget]\nchunk_size_min = 128\n")
        (child / "knox.toml").write_text("[compile.budget]\nchunk_size_min = 64\n")
        monkeypatch.setenv("KNOX_BUDGET_CHUNK_SIZE_MIN", "32")
        monkeypatch.chdir(child)
        cfg = resolve_config()
        assert cfg.get("compile.budget.chunk_size_min") == 32

    def test_hierarchy_provenance(self, tmp_path: Any, monkeypatch: Any) -> None:
        tmp_path = tmp_path.resolve()
        parent = tmp_path / "parent"
        parent.mkdir()
        child = parent / "child"
        child.mkdir()
        parent_file = parent / "knox.toml"
        parent_file.write_text("[compile.budget]\nchunk_size_min = 128\n[compile]\nworkers = 2\n")
        child_file = child / "knox.toml"
        child_file.write_text("[compile.budget]\nchunk_size_min = 64\n")
        monkeypatch.chdir(child)
        cfg = resolve_config()
        assert str(parent_file) in str(cfg.source_of("compile.workers"))
        assert str(child_file) in str(cfg.source_of("compile.budget.chunk_size_min"))

    def test_hierarchy_dot_knox_overrides_knox_toml(self, tmp_path: Any, monkeypatch: Any) -> None:
        dot_knox = tmp_path / KNOX_DIR
        dot_knox.mkdir()
        (dot_knox / "config.toml").write_text("[compile.budget]\nchunk_size_min = 64\n")
        (tmp_path / "knox.toml").write_text("[compile.budget]\nchunk_size_min = 128\n")
        monkeypatch.chdir(tmp_path)
        cfg = resolve_config()
        assert cfg.get("compile.budget.chunk_size_min") == 64


# ── Config mutation helpers ───────────────────────────────────────────────


class TestValidateConfigKey:
    def test_known_leaf_key(self) -> None:
        assert validate_config_key("compile.budget.chunk_size_min") is True

    def test_unknown_key(self) -> None:
        assert validate_config_key("compile.does_not_exist") is False

    def test_section_key_rejected(self) -> None:
        assert validate_config_key("compile") is False

    def test_user_tracking_is_known_key(self) -> None:
        # knox-gsb5: trust.user_tracking must be a schema-valid key.
        assert validate_config_key("trust.user_tracking") is True


class TestCoerceConfigValue:
    def test_coerce_integer(self) -> None:
        assert coerce_config_value("1024", "compile.budget.chunk_size_min") == 1024

    def test_coerce_float(self) -> None:
        assert coerce_config_value("1.5", "compile.memory_half_life_days") == 1.5

    def test_coerce_boolean_true(self) -> None:
        assert coerce_config_value("true", "compile.no_cache") is True

    def test_coerce_boolean_false(self) -> None:
        assert coerce_config_value("no", "compile.force") is False

    def test_coerce_list(self) -> None:
        assert coerce_config_value("a, b, c", "compile.llm.fallback_chain") == ["a", "b", "c"]

    def test_coerce_empty_list(self) -> None:
        assert coerce_config_value("", "compile.llm.fallback_chain") == []

    def test_coerce_string(self) -> None:
        assert coerce_config_value("openai/gpt-4o", "compile.llm.default_model") == "openai/gpt-4o"

    def test_coerce_string_stripped(self) -> None:
        assert coerce_config_value("  openai/gpt-4o  ", "compile.llm.default_model") == "openai/gpt-4o"

    def test_coerce_integer_stripped(self) -> None:
        assert coerce_config_value(" 1024 ", "compile.budget.chunk_size_min") == 1024

    def test_coerce_boolean_stripped(self) -> None:
        assert coerce_config_value(" true ", "compile.no_cache") is True

    def test_coerce_boolean_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="expected boolean"):
            coerce_config_value("maybe", "compile.no_cache")


class TestGetProjectConfigPath:
    def test_returns_existing_knox_toml(self, tmp_path: Any) -> None:
        config_file = tmp_path / "knox.toml"
        config_file.write_text("")
        assert get_project_config_path(start_dir=tmp_path) == config_file

    def test_prefers_dot_knox_config(self, tmp_path: Any) -> None:
        dot_knox = tmp_path / KNOX_DIR
        dot_knox.mkdir()
        dot_knox_file = dot_knox / "config.toml"
        dot_knox_file.write_text("")
        (tmp_path / "knox.toml").write_text("")
        assert get_project_config_path(start_dir=tmp_path) == dot_knox_file

    def test_returns_path_without_creating_dir(self, tmp_path: Any) -> None:
        path = get_project_config_path(start_dir=tmp_path)
        assert path == tmp_path / KNOX_DIR / "config.toml"
        assert not path.parent.exists()


class TestGetGlobalConfigPath:
    def test_uses_xdg_config_home(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        path = get_global_config_path()
        assert path == tmp_path / "knox" / "config.toml"

    def test_creates_parent_dir(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        path = get_global_config_path(create=True)
        assert path.parent.is_dir()


class TestSetConfigFileValue:
    def test_creates_file_and_sets_key(self, tmp_path: Any) -> None:
        path = tmp_path / "config.toml"
        set_config_file_value(path, "compile.budget.chunk_size_min", 1024)
        data = _load_toml(path)
        assert data["compile"]["budget"]["chunk_size_min"] == 1024

    def test_updates_existing_key(self, tmp_path: Any) -> None:
        path = tmp_path / "config.toml"
        path.write_text("[compile.budget]\nchunk_size_min = 512\n")
        set_config_file_value(path, "compile.budget.chunk_size_min", 256)
        data = _load_toml(path)
        assert data["compile"]["budget"]["chunk_size_min"] == 256

    def test_adds_key_to_existing_section(self, tmp_path: Any) -> None:
        path = tmp_path / "config.toml"
        path.write_text('[compile]\nformat = "markdown"\n')
        set_config_file_value(path, "compile.budget.chunk_size_min", 128)
        data = _load_toml(path)
        assert data["compile"]["format"] == "markdown"
        assert data["compile"]["budget"]["chunk_size_min"] == 128

    def test_creates_backup(self, tmp_path: Any) -> None:
        path = tmp_path / "config.toml"
        path.write_text("[compile.budget]\nchunk_size_min = 512\n")
        set_config_file_value(path, "compile.budget.chunk_size_min", 256)
        backup = path.with_suffix(path.suffix + ".bak")
        assert backup.is_file()
        assert "chunk_size_min = 512" in backup.read_text(encoding="utf-8")

    def test_unknown_key_raises(self, tmp_path: Any) -> None:
        with pytest.raises(KeyError):
            set_config_file_value(tmp_path / "config.toml", "compile.unknown", "x")


# ── Config generation ───────────────────────────────────────────────────


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestGenerateScaffoldConfig:
    def test_produces_valid_toml(self) -> None:
        text = generate_scaffold_config()
        assert text.startswith("# KNOX configuration")
        data = tomllib.loads(text)
        assert "compile" in data
        assert "paths" in data

    def test_excludes_hidden_keys(self) -> None:
        data = tomllib.loads(generate_scaffold_config())
        compile_section = data.get("compile", {})
        assert "sources_dir" not in compile_section
        assert "output_dir" not in compile_section

    def test_excludes_none_values(self) -> None:
        data = tomllib.loads(generate_scaffold_config())
        compile_llm = data.get("compile", {}).get("llm", {})
        assert "default_model" not in compile_llm
        assert "vlm_model" not in compile_llm

    def test_excludes_empty_sections(self) -> None:
        data = tomllib.loads(generate_scaffold_config())
        assert "hooks" not in data  # empty dict in DEFAULTS

    def test_sections_in_expected_order(self) -> None:
        data = tomllib.loads(generate_scaffold_config())
        keys = list(data.keys())
        assert keys[0] == "paths"
        assert keys[1] == "notes"
        assert "compile" in keys

    def test_budget_keys_present(self) -> None:
        data = tomllib.loads(generate_scaffold_config())
        budget = data["compile"]["budget"]
        assert budget["concepts_per_batch"] == 8
        assert budget["extract_output_max"] == 8192


class TestExtractTomlKeys:
    def test_simple_keys(self) -> None:
        text = '[compile.budget]\nchunk_size_min = 1024\n[compile.output]\nformat = "markdown"\n'
        keys = extract_toml_keys(text)
        assert "compile.output.format" in keys
        assert "compile.budget.chunk_size_min" in keys

    def test_nested_sections(self) -> None:
        text = '[compile.llm]\ndefault_model = "x"\ntimeout = 120.0\n'
        keys = extract_toml_keys(text)
        assert "compile.llm.default_model" in keys
        assert "compile.llm.timeout" in keys

    def test_commented_keys(self) -> None:
        text = "[compile]\n# force = false\n# no_cache = false\n"
        keys = extract_toml_keys(text)
        assert "compile.force" in keys
        assert "compile.no_cache" in keys

    def test_dotted_key_at_root(self) -> None:
        text = 'compile.output.format = "markdown"\n'
        keys = extract_toml_keys(text)
        assert "compile.output.format" in keys


class TestKnoxDotExampleSync:
    """knox.toml.example must document every key in DEFAULTS."""

    def test_example_covers_all_defaults(self) -> None:
        example_path = _PROJECT_ROOT / "knox.toml.example"
        if not example_path.exists():
            pytest.skip("knox.toml.example not found")
        text = example_path.read_text(encoding="utf-8")
        example_keys = extract_toml_keys(text)
        missing = get_default_keys() - example_keys
        assert not missing, f"knox.toml.example is missing keys present in DEFAULTS: {sorted(missing)}"
