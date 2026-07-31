"""5-layer config precedence: merge, resolve, and provenance tracking."""

# ruff: noqa: I001

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knox.core.config.engine._env import read_env_vars
from knox.core.config.engine._loading import (
    load_toml,
    wiki_dir_for_discovery,
    find_all_project_configs,
    find_global_config,
)
from knox.core.config.engine._schema import DEFAULTS, flat_keys
from knox.core.paths import VaultPaths
from knox.core.config.keys import (
    COMPILE_OUTPUT_DIR,
    COMPILE_SOURCES_DIR,
    PATHS_OUTPUT_DIR,
    PATHS_SOURCES_DIR,
    PATHS_WIKI_DIR,
)
from knox.core.utils.dicts import deep_merge, get_dot, nest_flat, set_dot
from knox.logging import get_logger
from knox.types import ConfigValue

logger = get_logger(__name__)


@dataclass
class ConfigSource:
    """Tracks where each config value came from."""

    path: str  # "default", "global", "project", "env", "cli"
    location: str = ""  # file path or env var name

    def __str__(self) -> str:
        if self.location:
            return f"{self.path} ({self.location})"
        return self.path


@dataclass
class ResolvedConfig:
    """Fully merged configuration with provenance tracking."""

    raw: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, ConfigSource] = field(default_factory=dict)
    project_config_file: Path | None = None
    project_config_paths: list[Path] = field(default_factory=list)
    global_config_file: Path | None = None

    def get(self, key: str, default: ConfigValue = None) -> ConfigValue:
        """Get a dotted key from the resolved config."""
        return get_dot(self.raw, key, default)

    def source_of(self, key: str) -> ConfigSource:
        """Return where a specific key's value came from."""
        return self.provenance.get(key, ConfigSource("default"))

    def flat(self) -> dict[str, Any]:
        """Return a flat dotted-key dict of all values."""
        result: dict[str, Any] = {}
        for key in flat_keys(self.raw):
            result[key] = get_dot(self.raw, key)
        return result


def resolve_config(
    *,
    cli_overrides: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    base_dir: Path | str | None = None,
) -> ResolvedConfig:
    """Build a fully resolved config by merging all 5 layers.

    Parameters
    ----------
    cli_overrides:
        Flat dotted-key dict of CLI-provided values.
    config_path:
        Explicit config file path (``--config`` flag).
    base_dir:
        Explicit vault root directory. Takes precedence over the
        ``paths.wiki_dir`` key in any other layer. Prefer passing this
        from a :class:`~knox.core.paths.VaultPaths` instance.

    Returns:
    -------
    ResolvedConfig
        The merged config with provenance.
    """
    provenance: dict[str, ConfigSource] = {}

    # Layer 5: defaults
    result: dict[str, Any] = deep_merge({}, DEFAULTS)
    for key in flat_keys(DEFAULTS):
        provenance[key] = ConfigSource("default")

    # Layer 4: global config
    global_path = find_global_config()
    if global_path is not None:
        global_data = load_toml(global_path)
        if global_data:
            result = _apply_layer(result, global_data, provenance, ConfigSource("global", str(global_path)))

    # Layer 3: project config
    project_dir: Path | None = None
    project_configs: list[Path] = []
    if config_path is not None:
        project_dir = Path(config_path)
        if project_dir.is_file():
            project_data = load_toml(project_dir)
            if project_data:
                result = _apply_layer(result, project_data, provenance, ConfigSource("project", str(project_dir)))
        else:
            logger.warning("Config file not found: %s", config_path)
    else:
        start_dir = wiki_dir_for_discovery(cli_overrides)
        project_configs = find_all_project_configs(start_dir)
        if project_configs:
            merged_project, project_provenance = _merge_project_configs(project_configs)
            if merged_project:
                result = deep_merge(result, merged_project)
                provenance.update(project_provenance)

    # Layer 2: env vars
    env_data = read_env_vars()
    if env_data:
        result = _apply_layer(result, env_data, provenance, ConfigSource("env", "KNOX_*"))

    # Layer 1: CLI overrides
    cli = cli_overrides or {}
    if base_dir is not None:
        cli = {**cli, PATHS_WIKI_DIR: str(VaultPaths(base_dir).base_dir)}
    if cli:
        cli_nested = nest_flat(cli)
        result = _apply_layer(result, cli_nested, provenance, ConfigSource("cli"))

    _resolve_paths(result, provenance)

    return ResolvedConfig(
        raw=result,
        provenance=provenance,
        project_config_file=project_configs[0] if project_configs else project_dir,
        project_config_paths=project_configs,
        global_config_file=global_path,
    )


# ── Layer application ────────────────────────────────────────────────────


def _apply_layer(
    result: dict[str, Any],
    layer: dict[str, Any],
    provenance: dict[str, ConfigSource],
    source: ConfigSource,
) -> dict[str, Any]:
    """Merge *layer* into *result* and stamp provenance for its keys."""
    merged = deep_merge(result, layer)
    for key in flat_keys(layer):
        if get_dot(layer, key) is not None:
            provenance[key] = source
    return merged


def _merge_project_configs(
    project_configs: list[Path],
) -> tuple[dict[str, Any], dict[str, ConfigSource]]:
    """Merge a project config hierarchy (farthest-to-closest) into one layer."""
    merged: dict[str, Any] = {}
    prov: dict[str, ConfigSource] = {}
    for config_file in reversed(project_configs):  # closest wins
        file_data = load_toml(config_file)
        for key in flat_keys(file_data):
            prov[key] = ConfigSource("project", str(config_file))
        merged = deep_merge(merged, file_data)
    return merged, prov


def _resolve_paths(
    result: dict[str, Any],
    provenance: dict[str, ConfigSource],
) -> None:
    """Normalize path values under ``paths``."""
    paths = result.setdefault("paths", {})

    # Backwards compatibility: old [compile] path keys migrate to [paths]
    for old_key, new_key in (
        (COMPILE_SOURCES_DIR, PATHS_SOURCES_DIR),
        (COMPILE_OUTPUT_DIR, PATHS_OUTPUT_DIR),
    ):
        old_val = get_dot(result, old_key)
        new_val = get_dot(result, new_key)
        old_default = get_dot(DEFAULTS, old_key)
        new_default = get_dot(DEFAULTS, new_key)
        if old_val is not None and old_val != old_default and new_val == new_default:
            set_dot(result, new_key, old_val)
            provenance[new_key] = ConfigSource(
                provenance.get(old_key, ConfigSource("project")).path,
                f"deprecated {old_key}",
            )

    wiki_dir_raw = paths.get("wiki_dir")
    vp = VaultPaths(wiki_dir_raw)
    base_dir = vp.base_dir
    paths["wiki_dir"] = base_dir
    paths["sources_dir"] = vp.sources_dir
    paths["output_dir"] = vp.output_dir
    paths["cache_dir"] = vp.cache_dir
    paths["state_dir"] = vp.knox_dir
