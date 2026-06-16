"""Unified configuration system for KNOX.

5-layer precedence (highest wins):

1. CLI flags           — explicit per-invocation overrides
2. Shell env vars      — ``KNOX_*`` per-session overrides
3. Project config      — ``./knox.toml`` and ``./.knox/config.toml``,
                         discovered hierarchically by walking up from cwd
4. Global config       — ``~/.config/knox/config.toml`` (per-user)
5. Built-in defaults   — hard-coded in :data:`DEFAULTS`

Config files use TOML. Example ``knox.toml``::

    [compile]
    sources_dir = "sources"
    output_dir = "wiki"
    format = "markdown"
    chunk_size = 1024
    image_mode = "file"
    mode = "full"
    workers = 1
    two_phase = true

    [compile.llm]
    default_model = "ollama/llama3.2"
    fallback_chain = ["openai/gpt-4o"]
    fallback_max_retries = 2

    [compile.output]
    flavor = "obsidian"

    [compile.output.frontmatter]
    language = "en"
    domain = "ml"

    [lint]
    enabled = ["broken-wikilinks", "orphan-page"]
    threshold = 80

    [eval]
    threshold = 80
    history_size = 20

    [ignore]
    patterns = ["*.tmp", ".drafts/"]
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

logger = logging.getLogger(__name__)

# ── File paths ───────────────────────────────────────────────────────────

_PROJECT_CONFIG_NAMES: tuple[str, ...] = (".knox/config.toml", "knox.toml")
_GLOBAL_CONFIG_DIR: str = "knox"
_GLOBAL_CONFIG_NAME: str = "config.toml"

# ── Built-in defaults ───────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "compile": {
        "sources_dir": "sources",
        "output_dir": "wiki",
        "format": "markdown",
        "chunk_size": 1024,
        "image_mode": "file",
        "mode": "full",
        "workers": 1,
        "no_cache": False,
        "force": False,
        "page_kind": None,
        "resolve_relations": True,
        "relation_confidence_threshold": 0.5,
        "memory_enabled": True,
        "memory_half_life_days": 90.0,
        "memory_decay_threshold": 0.3,
        "review": False,
        "bypass_review": False,
        "provenance": False,
        "private": False,
        "verify": False,
        "synthesize": False,
        "index": False,
        "lint": False,
        "budget": {
            "concepts_per_batch": 8,
            "verify_min_ngram": 3,
            "wicer_max_iterations": 3,
            "min_content_length": 50,
            "min_concept_sources": 2,
            "max_concept_pages": 50,
            "chunk_size_min": 2048,
            "chunk_size_max": 131072,
            "extract_output_ratio": 0.3,
            "extract_output_min": 512,
            "extract_output_max": 8192,
            "gen_output_max": 16384,
            "overview_output_ratio": 0.1,
            "overview_output_min": 256,
            "overview_output_max": 4096,
            "verify_output_ratio": 0.2,
            "verify_output_min": 256,
            "verify_output_max": 4096,
            "lint_output_max": 8192,
            "wicer_probe_output_max": 4096,
            "wicer_eval_output_max": 4096,
        },
        "llm": {
            "vlm_model": None,
            "fallback_chain": [],
            "fallback_max_retries": 2,
            "timeout": 120.0,
            "fallback_max_input": 8192,
            "fallback_max_output": 4096,
            "num_ctx_min": 2048,
            "num_ctx_max": 131072,
            "context_window_cap": None,
        },
        "output": {
            "flavor": None,
            "frontmatter": {},
            "rumdl_config": None,
            "max_related_links": 15,
        },
        "embedding": {
            "model": None,
            "dim": 384,
            "batch_size": 10,
        },
    },
    "lint": {
        "enabled": None,  # None = all structural rules
        "threshold": None,
    },
    "eval": {
        "threshold": None,
        "history_size": 20,
    },
    "trust": {
        "privacy_filter": True,
    },
    "ignore": {
        "patterns": [],
    },
    "hooks": {},
    "paths": {
        "wiki_dir": None,
        "sources_dir": "sources",
        "output_dir": "wiki",
        "cache_dir": ".knox/cache",
        "state_dir": ".knox",
    },
    "notes": {
        "file": "NOTES.md",
    },
}

# ── Env var mapping ──────────────────────────────────────────────────────
# Maps dotted config key → env var name.
# Only keys that make sense as env overrides are listed.

_ENV_MAP: dict[str, str] = {
    "paths.wiki_dir": "KNOX_WIKI_DIR",
    "paths.sources_dir": "KNOX_SOURCES_DIR",
    "paths.output_dir": "KNOX_OUTPUT_DIR",
    "paths.cache_dir": "KNOX_CACHE_DIR",
    "paths.state_dir": "KNOX_STATE_DIR",
    "compile.format": "KNOX_FORMAT",
    "compile.chunk_size": "KNOX_CHUNK_SIZE",
    "compile.image_mode": "KNOX_IMAGE_MODE",
    "compile.start_at_tier": "KNOX_START_AT_TIER",
    "compile.workers": "KNOX_WORKERS",
    "compile.no_cache": "KNOX_NO_CACHE",
    "compile.force": "KNOX_FORCE",
    "compile.lint": "KNOX_LINT",
    "compile.page_kind": "KNOX_PAGE_KIND",
    "compile.budget.concepts_per_batch": "KNOX_BUDGET_CONCEPTS_PER_BATCH",
    "compile.budget.wicer_max_iterations": "KNOX_BUDGET_WICER_MAX_ITERATIONS",
    "compile.budget.min_content_length": "KNOX_BUDGET_MIN_CONTENT_LENGTH",
    "compile.budget.min_concept_sources": "KNOX_BUDGET_MIN_CONCEPT_SOURCES",
    "compile.budget.max_concept_pages": "KNOX_BUDGET_MAX_CONCEPT_PAGES",
    "compile.budget.chunk_size_min": "KNOX_BUDGET_CHUNK_SIZE_MIN",
    "compile.budget.chunk_size_max": "KNOX_BUDGET_CHUNK_SIZE_MAX",
    "compile.budget.extract_output_ratio": "KNOX_BUDGET_EXTRACT_OUTPUT_RATIO",
    "compile.budget.extract_output_min": "KNOX_BUDGET_EXTRACT_OUTPUT_MIN",
    "compile.budget.extract_output_max": "KNOX_BUDGET_EXTRACT_OUTPUT_MAX",
    "compile.budget.gen_output_max": "KNOX_BUDGET_GEN_OUTPUT_MAX",
    "compile.budget.overview_output_ratio": "KNOX_BUDGET_OVERVIEW_OUTPUT_RATIO",
    "compile.budget.overview_output_min": "KNOX_BUDGET_OVERVIEW_OUTPUT_MIN",
    "compile.budget.overview_output_max": "KNOX_BUDGET_OVERVIEW_OUTPUT_MAX",
    "compile.llm.default_model": "KNOX_LLM_DEFAULT_MODEL",
    "compile.llm.vlm_model": "KNOX_VLM_MODEL",
    "compile.resolve_relations": "KNOX_RESOLVE_RELATIONS",
    "compile.relation_confidence_threshold": "KNOX_RELATION_CONFIDENCE_THRESHOLD",
    "compile.memory_enabled": "KNOX_MEMORY_ENABLED",
    "compile.memory_half_life_days": "KNOX_MEMORY_HALF_LIFE_DAYS",
    "compile.memory_decay_threshold": "KNOX_MEMORY_DECAY_THRESHOLD",
    "compile.llm.fallback_chain": "KNOX_LLM_FALLBACK_CHAIN",
    "compile.llm.fallback_max_retries": "KNOX_LLM_FALLBACK_MAX_RETRIES",
    "compile.llm.timeout": "KNOX_LLM_TIMEOUT",
    "compile.llm.fallback_max_input": "KNOX_LLM_FALLBACK_MAX_INPUT",
    "compile.llm.fallback_max_output": "KNOX_LLM_FALLBACK_MAX_OUTPUT",
    "compile.llm.num_ctx_min": "KNOX_LLM_NUM_CTX_MIN",
    "compile.llm.num_ctx_max": "KNOX_LLM_NUM_CTX_MAX",
    "compile.llm.context_window_cap": "KNOX_LLM_CONTEXT_WINDOW_CAP",
    "compile.output.flavor": "KNOX_OUTPUT_FLAVOR",
    "compile.output.rumdl_config": "KNOX_RUMDL_CONFIG",
    "compile.output.max_related_links": "KNOX_OUTPUT_MAX_RELATED_LINKS",
    "compile.embedding.model": "KNOX_EMBEDDING_MODEL",
    "compile.embedding.provider": "KNOX_EMBEDDING_PROVIDER",
    "compile.embedding.dim": "KNOX_EMBEDDING_DIM",
    "compile.embedding.batch_size": "KNOX_EMBEDDING_BATCH_SIZE",
    "lint.threshold": "KNOX_LINT_THRESHOLD",
    "trust.privacy_filter": "KNOX_PRIVACY_FILTER",
    "eval.threshold": "KNOX_EVAL_THRESHOLD",
    "notes.file": "KNOX_NOTES_FILE",
}


# ── Deep merge ──────────────────────────────────────────────────────────


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Dict values are merged recursively. Non-dict values from *override*
    replace those in *base*. All values are deep-copied so the returned
    dict never shares references with the inputs.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value) if isinstance(value, dict) else value
    return result


def _nest_flat(flat: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat dotted-key dict to a nested dict."""
    result: dict[str, Any] = {}
    for key, value in flat.items():
        _set_dot(result, key, value)
    return result


# ── Dot-path access ─────────────────────────────────────────────────────


def _get_dot(data: dict[str, Any], key: str, default: Any = None) -> Any:
    """Retrieve a dotted key from a nested dict."""
    parts = key.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return default
    return current if current is not None else default


def _set_dot(data: dict[str, Any], key: str, value: Any) -> None:
    """Set a dotted key in a nested dict, creating intermediate dicts."""
    parts = key.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


# ── Config file discovery ───────────────────────────────────────────────


def find_project_config(start_dir: Path | None = None) -> Path | None:
    """Walk up from *start_dir* looking for project config files.

    Prefers ``.knox/config.toml`` over ``knox.toml`` in the same directory.
    Returns the first found path, or ``None``.
    """
    current = (start_dir or Path.cwd()).resolve()
    for parent in [current] + list(current.parents):
        for name in _PROJECT_CONFIG_NAMES:
            candidate = parent / name
            if candidate.is_file():
                return candidate
    return None


def find_all_project_configs(start_dir: Path | None = None) -> list[Path]:
    """Walk up from *start_dir* collecting all project config files.

    Returns a list ordered from closest to farthest from *start_dir*.
    Within each directory, ``.knox/config.toml`` is preferred over
    ``knox.toml``; only one file per directory is returned.
    """
    current = (start_dir or Path.cwd()).resolve()
    found: list[Path] = []
    seen_dirs: set[Path] = set()
    for parent in [current] + list(current.parents):
        if parent in seen_dirs:
            continue
        seen_dirs.add(parent)
        for name in _PROJECT_CONFIG_NAMES:
            candidate = parent / name
            if candidate.is_file():
                found.append(candidate)
                break
    return found


def find_global_config() -> Path | None:
    """Return the global config path if it exists.

    Location follows the XDG convention:
    ``$XDG_CONFIG_HOME/knox/config.toml`` (defaults to
    ``~/.config/knox/config.toml``).
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_dir = Path(xdg) if xdg else Path.home() / ".config"
    candidate = config_dir / _GLOBAL_CONFIG_DIR / _GLOBAL_CONFIG_NAME
    return candidate if candidate.is_file() else None


# ── TOML loading ────────────────────────────────────────────────────────


def _expand_env_vars(value: str) -> str:
    """Expand ``$VAR`` and ``${VAR}`` patterns in a string.

    Unknown variables are left as-is. This is applied to string values
    when loading TOML config files so users can reference env vars like
    ``$HOME`` in path settings.
    """

    def _replace(match: re.Match) -> str:
        var = match.group(1) or match.group(2) or ""
        return os.environ.get(var, match.group(0))

    return re.sub(r"\$([A-Z_][A-Z0-9_]*|\{[A-Z_][A-Z0-9_]*\})|\$\{(\w+)", _replace, value)


def _expand_env_vars_recursive(data: Any) -> Any:
    """Recursively expand env vars in all string values."""
    if isinstance(data, str):
        return _expand_env_vars(data)
    if isinstance(data, dict):
        return {k: _expand_env_vars_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_expand_env_vars_recursive(v) for v in data]
    return data


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file, returning an empty dict on failure.

    String values are processed through ``_expand_env_vars`` to support
    ``$VAR`` and ``${VAR}`` patterns.
    """
    try:
        raw = path.read_bytes()
        data: dict[str, Any] = tomllib.loads(raw.decode("utf-8"))
        logger.debug("Loaded config from %s", path)
        return _expand_env_vars_recursive(data)
    except Exception as exc:
        logger.warning("Failed to load config %s: %s", path, exc)
        return {}


# ── Env var layer ───────────────────────────────────────────────────────


def _read_env_vars() -> dict[str, Any]:
    """Read KNOX_* env vars and return as a nested config dict."""
    result: dict[str, Any] = {}
    for dotted_key, env_name in _ENV_MAP.items():
        value = os.environ.get(env_name)
        if value is None:
            continue
        # Type coerce: booleans, integers, comma-separated lists
        coerced: Any = value
        if value.lower() in ("true", "1", "yes"):
            coerced = True
        elif value.lower() in ("false", "0", "no"):
            coerced = False
        elif value.isdigit():
            coerced = int(value)
        else:
            with contextlib.suppress(ValueError):
                coerced = float(value)
        if "," in value and not isinstance(coerced, (int, float)):
            coerced = [v.strip() for v in value.split(",") if v.strip()]
        _set_dot(result, dotted_key, coerced)
    return result


# ── ResolvedConfig ──────────────────────────────────────────────────────


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
    """Fully merged configuration with provenance tracking.

    Layers are applied in order: defaults → global → project → env → cli.
    Each value carries a :class:`ConfigSource` so ``knox config show``
    can report where it came from.
    """

    raw: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, ConfigSource] = field(default_factory=dict)
    project_config_path: Path | None = None
    project_config_paths: list[Path] = field(default_factory=list)
    global_config_path: Path | None = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get a dotted key from the resolved config."""
        return _get_dot(self.raw, key, default)

    def source_of(self, key: str) -> ConfigSource:
        """Return where a specific key's value came from."""
        return self.provenance.get(key, ConfigSource("default"))

    def flat(self) -> dict[str, Any]:
        """Return a flat dotted-key dict of all values."""
        result: dict[str, Any] = {}
        self._flatten(self.raw, "", result)
        return result

    def _flatten(self, data: dict[str, Any], prefix: str, out: dict[str, Any]) -> None:
        for key, value in data.items():
            full = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._flatten(value, full, out)
            else:
                out[full] = value


def _merge_layer(
    base: dict[str, Any],
    layer: dict[str, Any],
    provenance: dict[str, ConfigSource],
    source: ConfigSource,
) -> dict[str, Any]:
    """Merge a layer into base, recording provenance for new/changed keys."""
    result = _deep_merge(base, layer)
    # Record provenance for keys this layer contributed
    for key in _flat_keys(layer):
        if _get_dot(layer, key) is not None:
            provenance[key] = source
    return result


def _flat_keys(data: dict[str, Any], prefix: str = "") -> list[str]:
    """Return all dotted keys in a nested dict."""
    keys: list[str] = []
    for key, value in data.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys.extend(_flat_keys(value, full))
        else:
            keys.append(full)
    return keys


def _resolve_paths(
    result: dict[str, Any],
    provenance: dict[str, ConfigSource],
) -> None:
    """Normalize path values under ``paths``.

    - Migrates deprecated ``compile.sources_dir`` / ``compile.output_dir``
      values to ``paths.sources_dir`` / ``paths.output_dir`` when the new
      keys were not explicitly set.
    - Resolves relative paths against ``paths.wiki_dir`` (or cwd).
    """
    paths = result.setdefault("paths", {})

    # Backwards compatibility: old [compile] path keys migrate to [paths]
    for old_key, new_key in (
        ("compile.sources_dir", "paths.sources_dir"),
        ("compile.output_dir", "paths.output_dir"),
    ):
        old_val = _get_dot(result, old_key)
        new_val = _get_dot(result, new_key)
        old_default = _get_dot(DEFAULTS, old_key)
        new_default = _get_dot(DEFAULTS, new_key)
        if old_val is not None and old_val != old_default and new_val == new_default:
            _set_dot(result, new_key, old_val)
            provenance[new_key] = ConfigSource(
                provenance.get(old_key, ConfigSource("project")).path,
                f"deprecated {old_key}",
            )

    wiki_dir_raw = paths.get("wiki_dir")
    base = Path(wiki_dir_raw).expanduser().resolve() if wiki_dir_raw else Path.cwd()
    paths["wiki_dir"] = base

    for key in ("sources_dir", "output_dir", "cache_dir", "state_dir"):
        dotted = f"paths.{key}"
        value = paths.get(key)
        if value is None:
            value = _get_dot(DEFAULTS, dotted)
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base / path
        paths[key] = path.resolve()


def resolve_config(
    *,
    cli_overrides: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> ResolvedConfig:
    """Build a fully resolved config by merging all 5 layers.

    Parameters
    ----------
    cli_overrides:
        Flat dotted-key dict of CLI-provided values. Only non-``None``
        values should be included — the caller filters out unspecified
        flags.
    config_path:
        Explicit config file path (``--config`` flag). When set, this
        is treated as a project-level override — it replaces the
        auto-discovered project hierarchy but is still merged below CLI.

    Returns:
    -------
    ResolvedConfig
        The merged config with provenance.
    """
    provenance: dict[str, ConfigSource] = {}

    # Layer 5: defaults
    result: dict[str, Any] = _deep_merge({}, DEFAULTS)
    for key in _flat_keys(DEFAULTS):
        provenance[key] = ConfigSource("default")

    # Layer 4: global config
    global_path = find_global_config()
    if global_path is not None:
        global_data = _load_toml(global_path)
        if global_data:
            result = _merge_layer(result, global_data, provenance, ConfigSource("global", str(global_path)))

    # Layer 3: project config (explicit --config overrides auto-discovered)
    project_path: Path | None = None
    project_configs: list[Path] = []
    if config_path is not None:
        project_path = Path(config_path)
        if project_path.is_file():
            project_data = _load_toml(project_path)
            if project_data:
                result = _merge_layer(result, project_data, provenance, ConfigSource("project", str(project_path)))
        else:
            logger.warning("Config file not found: %s", config_path)
    else:
        project_configs = find_all_project_configs()
        if project_configs:
            # Merge from farthest to closest so closest wins.
            merged_project: dict[str, Any] = {}
            project_provenance: dict[str, ConfigSource] = {}
            for path in reversed(project_configs):
                file_data = _load_toml(path)
                for key in _flat_keys(file_data):
                    project_provenance[key] = ConfigSource("project", str(path))
                merged_project = _deep_merge(merged_project, file_data)
            if merged_project:
                result = _deep_merge(result, merged_project)
                provenance.update(project_provenance)

    # Layer 2: env vars
    env_data = _read_env_vars()
    if env_data:
        # Record per-key provenance with specific env var names
        env_result = _deep_merge(result, env_data)
        for key in _flat_keys(env_data):
            env_name = _ENV_MAP.get(key, "KNOX_*")
            provenance[key] = ConfigSource("env", env_name)
        result = env_result

    # Layer 1: CLI overrides
    cli = cli_overrides or {}
    if cli:
        cli_nested = _nest_flat(cli)
        cli_result = _deep_merge(result, cli_nested)
        for key in cli:
            if cli[key] is not None:
                provenance[key] = ConfigSource("cli")
        result = cli_result

    _resolve_paths(result, provenance)

    return ResolvedConfig(
        raw=result,
        provenance=provenance,
        project_config_path=project_configs[0] if project_configs else project_path,
        project_config_paths=project_configs,
        global_config_path=global_path,
    )


# ── Config mutation helpers ───────────────────────────────────────────────


def _flat_schema() -> set[str]:
    """Return all leaf dotted keys present in :data:`DEFAULTS`."""
    return set(_flat_keys(DEFAULTS))


def validate_config_key(key: str) -> bool:
    """Return ``True`` if *key* is a known leaf config key."""
    return key in _flat_schema()


def coerce_config_value(value: str, key: str) -> Any:
    """Coerce a CLI string to the type expected by ``DEFAULTS`` for *key*.

    Raises:
        ValueError: when *value* cannot be coerced to the expected type.
    """
    current = _get_dot(DEFAULTS, key)
    lowered = value.lower()
    true_values = {"true", "1", "yes", "on"}
    false_values = {"false", "0", "no", "off"}
    coerced: Any

    if isinstance(current, bool):
        if lowered in true_values:
            coerced = True
        elif lowered in false_values:
            coerced = False
        else:
            raise ValueError(f"expected boolean for {key}")
    elif isinstance(current, int):
        coerced = int(value)
    elif isinstance(current, float):
        coerced = float(value)
    elif isinstance(current, list):
        coerced = [] if value == "" else [v.strip() for v in value.split(",") if v.strip()]
    elif lowered in true_values:
        coerced = True
    elif lowered in false_values:
        coerced = False
    else:
        coerced = value
        with contextlib.suppress(ValueError):
            coerced = int(value)
        if coerced == value:
            with contextlib.suppress(ValueError):
                coerced = float(value)

    return coerced


def get_project_config_path(start_dir: Path | None = None, *, create: bool = False) -> Path:
    """Return the project config file path used by ``knox config set``.

    Prefers ``.knox/config.toml`` over ``knox.toml``. If no project config
    exists and *create* is ``True``, returns ``<start_dir>/.knox/config.toml``
    and creates the parent directory.
    """
    current = (start_dir or Path.cwd()).resolve()
    found = find_project_config(current)
    if found is not None:
        return found
    target = current / ".knox" / "config.toml"
    if create:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def get_global_config_path(*, create: bool = False) -> Path:
    """Return the global config file path.

    Location follows the XDG convention. When *create* is ``True``, the
    parent directory is created if it does not exist.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_dir = Path(xdg) if xdg else Path.home() / ".config"
    target = config_dir / _GLOBAL_CONFIG_DIR / _GLOBAL_CONFIG_NAME
    if create:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _atomic_write_toml(path: Path, data: dict[str, Any]) -> None:
    """Write *data* to *path* atomically, keeping a ``.bak`` copy."""
    if path.is_file():
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_bytes(path.read_bytes())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(tomli_w.dumps(data), encoding="utf-8")
    tmp.replace(path)


def set_config_file_value(path: Path, key: str, value: Any) -> None:
    """Set *key* to *value* in *path*, creating the file if needed.

    Raises:
        KeyError: if *key* is not a known config key.
        ValueError: if *value* cannot be serialised as TOML.
    """
    if not validate_config_key(key):
        raise KeyError(f"unknown config key: {key}")
    data = _load_toml(path) if path.is_file() else {}
    _set_dot(data, key, value)
    _atomic_write_toml(path, data)
