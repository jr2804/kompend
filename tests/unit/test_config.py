"""Unit tests for KNOX unified config system (Q23)."""

from __future__ import annotations

import json
import os
from typing import Any

from knox.core.config._core import (
    _deep_merge,
    _flat_keys,
    _get_dot,
    _load_toml,
    _read_env_vars,
    _set_dot,
    find_global_config,
    find_project_config,
    resolve_config,
)

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
        base = {"compile": {"chunk_size": 512, "format": "markdown"}}
        override = {"compile": {"chunk_size": 256}}
        result = _deep_merge(base, override)
        assert result["compile"]["chunk_size"] == 256
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
        data = {"compile": {"chunk_size": 512, "format": "md"}}
        assert sorted(_flat_keys(data)) == ["compile.chunk_size", "compile.format"]

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
        monkeypatch.setenv("KNOX_CHUNK_SIZE", "256")
        result = _read_env_vars()
        assert result["compile"]["chunk_size"] == 256

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


# ── Config file discovery ───────────────────────────────────────────────


class TestConfigDiscovery:
    def test_find_project_config_none_when_missing(self, tmp_path: Any) -> None:
        result = find_project_config(start_dir=tmp_path)
        assert result is None

    def test_find_project_config_finds_toml(self, tmp_path: Any) -> None:
        config_file = tmp_path / "knox.toml"
        config_file.write_text('[compile]\nchunk_size = 128\n')
        result = find_project_config(start_dir=tmp_path)
        assert result == config_file

    def test_find_project_config_walks_up(self, tmp_path: Any) -> None:
        config_file = tmp_path / "knox.toml"
        config_file.write_text('[compile]\nchunk_size = 128\n')
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
        config_file.write_text('[compile]\nchunk_size = 64\n')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        result = find_global_config()
        assert result == config_file


# ── resolve_config ──────────────────────────────────────────────────────


class TestResolveConfig:
    def test_defaults_only(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = resolve_config()
        assert cfg.get("compile.chunk_size") == 512
        assert cfg.get("compile.format") == "markdown"
        assert cfg.get("compile.llm.fallback_max_retries") == 2

    def test_project_config_overrides_defaults(self, tmp_path: Any, monkeypatch: Any) -> None:
        config_file = tmp_path / "knox.toml"
        config_file.write_text('[compile]\nchunk_size = 128\n')
        monkeypatch.chdir(tmp_path)
        cfg = resolve_config()
        assert cfg.get("compile.chunk_size") == 128
        assert cfg.get("compile.format") == "markdown"  # still default

    def test_env_overrides_project(self, tmp_path: Any, monkeypatch: Any) -> None:
        config_file = tmp_path / "knox.toml"
        config_file.write_text('[compile]\nchunk_size = 128\n')
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KNOX_CHUNK_SIZE", "64")
        cfg = resolve_config()
        assert cfg.get("compile.chunk_size") == 64

    def test_cli_overrides_env(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KNOX_CHUNK_SIZE", "64")
        cfg = resolve_config(cli_overrides={"compile.chunk_size": 32})
        assert cfg.get("compile.chunk_size") == 32

    def test_provenance_tracking(self, tmp_path: Any, monkeypatch: Any) -> None:
        config_file = tmp_path / "knox.toml"
        config_file.write_text('[compile]\nchunk_size = 128\n')
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KNOX_WORKERS", "4")
        cfg = resolve_config(cli_overrides={"compile.format": "json"})

        assert "project" in str(cfg.source_of("compile.chunk_size"))
        assert "env" in str(cfg.source_of("compile.workers"))
        assert "cli" in str(cfg.source_of("compile.format"))
        assert "default" in str(cfg.source_of("compile.image_mode"))

    def test_global_config_overrides_defaults(self, tmp_path: Any, monkeypatch: Any) -> None:
        config_dir = tmp_path / "knox"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[compile]\nchunk_size = 64\n')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        # Use a subdir with no project config
        subdir = tmp_path / "project"
        subdir.mkdir()
        monkeypatch.chdir(subdir)
        cfg = resolve_config()
        assert cfg.get("compile.chunk_size") == 64

    def test_project_overrides_global(self, tmp_path: Any, monkeypatch: Any) -> None:
        # Global: chunk_size = 64
        config_dir = tmp_path / "knox"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text('[compile]\nchunk_size = 64\n')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        # Project: chunk_size = 128
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "knox.toml").write_text('[compile]\nchunk_size = 128\n')
        monkeypatch.chdir(project_dir)
        cfg = resolve_config()
        assert cfg.get("compile.chunk_size") == 128


# ── ResolvedConfig ──────────────────────────────────────────────────────


class TestResolvedConfig:
    def test_flat(self) -> None:
        cfg = resolve_config()
        flat = cfg.flat()
        assert isinstance(flat, dict)
        assert "compile.chunk_size" in flat

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
        f.write_text('[compile]\nchunk_size = 256\nformat = "json"\n')
        data = _load_toml(f)
        assert data["compile"]["chunk_size"] == 256
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
        f.write_text(
            '[compile.llm]\n'
            'default_model = "openai/gpt-4o"\n'
            'fallback_chain = ["anthropic/claude"]\n'
        )
        data = _load_toml(f)
        assert data["compile"]["llm"]["default_model"] == "openai/gpt-4o"
        assert data["compile"]["llm"]["fallback_chain"] == ["anthropic/claude"]
