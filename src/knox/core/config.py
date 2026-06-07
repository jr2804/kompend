"""Unified configuration system for KNOX.

5-layer precedence (highest wins):

1. CLI flags           — explicit per-invocation overrides
2. Shell env vars      — ``KNOX_*`` per-session overrides
3. Project config      — ``./knox.toml`` (checked into repo)
4. Global config       — ``~/.config/knox/config.toml`` (per-user)
5. Built-in defaults   — hard-coded in :data:`DEFAULTS`

Config files use TOML. Example ``knox.toml``::

    [compile]
    sources_dir = "sources"
    output_dir = "wiki"
    format = "markdown"
    chunk_size = 512
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

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── File paths ───────────────────────────────────────────────────────────

_PROJECT_CONFIG_NAMES: tuple[str, ...] = ("knox.toml",)
_GLOBAL_CONFIG_DIR: str = "knox"
_GLOBAL_CONFIG_NAME: str = "config.toml"

# ── Built-in defaults ───────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "compile": {
        "sources_dir": "sources",
        "output_dir": "wiki",
        "format": "markdown",
        "chunk_size": 512,
        "image_mode": "file",
        "mode": "full",
        "workers": 1,
        "two_phase": True,
        "prompt_budget": 200_000,
        "no_cache": False,
        "force": False,
        "review": False,
        "bypass_review": False,
        "provenance": False,
        "private": False,
        "verify": False,
        "synthesize": False,
        "index": False,
        "lint": False,
        "llm": {
            "default_model": None,
            "fallback_chain": [],
            "fallback_max_retries": 2,
        },
        "output": {
            "flavor": None,
            "frontmatter": {},
            "rumdl_config": None,
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
    "ignore": {
        "patterns": [],
    },
}

# ── Env var mapping ──────────────────────────────────────────────────────
# Maps dotted config key → env var name.
# Only keys that make sense as env overrides are listed.

_ENV_MAP: dict[str, str] = {
    "compile.sources_dir": "KNOX_SOURCES_DIR",
    "compile.output_dir": "KNOX_OUTPUT_DIR",
    "compile.format": "KNOX_FORMAT",
    "compile.chunk_size": "KNOX_CHUNK_SIZE",
    "compile.image_mode": "KNOX_IMAGE_MODE",
    "compile.mode": "KNOX_MODE",
    "compile.workers": "KNOX_WORKERS",
    "compile.two_phase": "KNOX_TWO_PHASE",
    "compile.prompt_budget": "KNOX_PROMPT_BUDGET",
    "compile.no_cache": "KNOX_NO_CACHE",
    "compile.force": "KNOX_FORCE",
    "compile.lint": "KNOX_LINT",
    "compile.llm.default_model": "KNOX_LLM_DEFAULT_MODEL",
    "compile.llm.fallback_chain": "KNOX_LLM_FALLBACK_CHAIN",
    "compile.llm.fallback_max_retries": "KNOX_LLM_FALLBACK_MAX_RETRIES",
    "compile.output.flavor": "KNOX_OUTPUT_FLAVOR",
    "compile.output.rumdl_config": "KNOX_RUMDL_CONFIG",
    "lint.threshold": "KNOX_LINT_THRESHOLD",
    "eval.threshold": "KNOX_EVAL_THRESHOLD",
}


# ── Deep merge ──────────────────────────────────────────────────────────


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Dict values are merged recursively. Non-dict values from *override*
    replace those in *base*.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
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
    """Walk up from *start_dir* looking for ``knox.toml``.

    Returns the first found path, or ``None``.
    """
    current = (start_dir or Path.cwd()).resolve()
    for parent in [current] + list(current.parents):
        for name in _PROJECT_CONFIG_NAMES:
            candidate = parent / name
            if candidate.is_file():
                return candidate
    return None


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


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file, returning an empty dict on failure."""
    try:
        # Python 3.11+ has tomllib in stdlib
        import tomllib  # noqa: PLC0415
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]  # noqa: PLC0415
        except ImportError:
            logger.warning("No TOML parser available — cannot read %s", path)
            return {}

    try:
        raw = path.read_bytes()
        data: dict[str, Any] = tomllib.loads(raw.decode("utf-8"))
        logger.debug("Loaded config from %s", path)
        return data
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
        elif "," in value:
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


def resolve_config(
    *,
    cli_overrides: dict[str, Any] | None = None,
) -> ResolvedConfig:
    """Build a fully resolved config by merging all 5 layers.

    Parameters
    ----------
    cli_overrides:
        Flat dotted-key dict of CLI-provided values. Only non-``None``
        values should be included — the caller filters out unspecified
        flags.

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

    # Layer 3: project config
    project_path = find_project_config()
    if project_path is not None:
        project_data = _load_toml(project_path)
        if project_data:
            result = _merge_layer(result, project_data, provenance, ConfigSource("project", str(project_path)))

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

    return ResolvedConfig(
        raw=result,
        provenance=provenance,
        project_config_path=project_path,
        global_config_path=global_path,
    )
