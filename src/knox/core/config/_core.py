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
    image_mode = "file"
    vlm_min_image_bytes = 2048
    vlm_min_image_dimension = 64
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

    [eval]
    threshold = 80
    history_size = 20

    [ignore]
    patterns = ["*.tmp", ".drafts/"]
"""

from __future__ import annotations

import contextlib
import copy
import os
import re
import tomllib
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, get_origin

import tomli_w
from pydantic import BaseModel, Field, TypeAdapter, create_model
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

from knox.core.config._configs import (
    KnoxConfig,
)
from knox.core.config.keys import (
    COMPILE_CACHE_DIR,
    COMPILE_OUTPUT_DIR,
    COMPILE_RESPECT_GITIGNORE,
    COMPILE_SOURCES_DIR,
    COMPILE_START_AT_TIER,
    COMPILE_STOP_AT_TIER,
    COMPILE_TIER_STATE_PATH,
    PATHS_OUTPUT_DIR,
    PATHS_SOURCES_DIR,
    PATHS_WIKI_DIR,
)
from knox.core.paths import CONFIG_FILE, KNOX_CONFIG_FILE, PROJECT_CONFIG_FILE, VaultPaths
from knox.logging import get_logger

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


@dataclass
class ResolvedConfig:
    """Fully merged configuration with provenance tracking.

    Layers are applied in order: defaults → global → project → env → cli.
    Each value carries a :class:`ConfigSource` so ``knox config show``
    can report where it came from.
    """

    raw: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, ConfigSource] = field(default_factory=dict)
    project_config_file: Path | None = None
    project_config_paths: list[Path] = field(default_factory=list)
    global_config_file: Path | None = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get a dotted key from the resolved config."""
        return _get_dot(self.raw, key, default)

    def source_of(self, key: str) -> ConfigSource:
        """Return where a specific key's value came from."""
        return self.provenance.get(key, ConfigSource("default"))

    def _flatten(self, data: dict[str, Any], prefix: str, out: dict[str, Any]) -> None:
        for key, value in data.items():
            full = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._flatten(value, full, out)
            else:
                out[full] = value

    def flat(self) -> dict[str, Any]:
        """Return a flat dotted-key dict of all values."""
        result: dict[str, Any] = {}
        self._flatten(self.raw, "", result)
        return result

# ── File paths ───────────────────────────────────────────────────────────

_PROJECT_CONFIG_NAMES: tuple[str, ...] = (KNOX_CONFIG_FILE, PROJECT_CONFIG_FILE)
_GLOBAL_CONFIG_DIR: str = "knox"
_GLOBAL_CONFIG_NAME: str = CONFIG_FILE

# ── Built-in defaults (auto-generated from KnoxConfig) ──────────────────

def _build_defaults() -> dict[str, Any]:
    """Generate DEFAULTS from the :class:`KnoxConfig` schema model.

    ``model_dump()`` produces a nested dict where StrEnum members are
    serialized as their ``.value`` (string), ``Path`` objects as ``Path``,
    and ``None`` stays ``None``. This is the single source of truth —
    no hand-written DEFAULTS dict to keep in sync.
    """
    return KnoxConfig().model_dump()


DEFAULTS: dict[str, Any] = _build_defaults()


def _build_env_map() -> dict[str, str]:
    """Auto-generate _ENV_MAP from :class:`KnoxConfig` validation aliases.

    Walks the model's fields recursively. For each leaf field with a
    ``validation_alias`` (a plain string), records ``(dotted_path, alias)``.
    CLI-only env vars (keys not in DEFAULTS) are added manually.
    """
    env_map: dict[str, str] = {}

    def _walk(cls: type[BaseModel], prefix: str = "") -> None:
        for name, info in cls.model_fields.items():
            dotted = f"{prefix}.{name}" if prefix else name
            ann = info.annotation
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                _walk(ann, dotted)
            else:
                alias = info.validation_alias
                if isinstance(alias, str):
                    env_map[dotted] = alias

    _walk(KnoxConfig)
    # CLI-only env vars (keys not in the config schema model)
    env_map.update({
        COMPILE_START_AT_TIER: "KNOX_START_AT_TIER",
        COMPILE_STOP_AT_TIER: "KNOX_STOP_AT_TIER",
        COMPILE_TIER_STATE_PATH: "KNOX_TIER_STATE_PATH",
        COMPILE_CACHE_DIR: "KNOX_COMPILE_CACHE_DIR",
        COMPILE_RESPECT_GITIGNORE: "KNOX_RESPECT_GITIGNORE",
    })
    return env_map


_ENV_MAP: dict[str, str] = _build_env_map()


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


def _set_dot(data: dict[str, Any], key: str, value: Any) -> None:
    """Set a dotted key in a nested dict, creating intermediate dicts."""
    parts = key.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _nest_flat(flat: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat dotted-key dict to a nested dict."""
    result: dict[str, Any] = {}
    for key, value in flat.items():
        _set_dot(result, key, value)
    return result


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


def _wiki_dir_for_discovery(cli_overrides: dict[str, Any] | None) -> Path | None:
    """Return the dir to start project-config discovery from.

    Reflects the *explicit* vault selection — ``--wiki-dir`` on the CLI or
    ``KNOX_WIKI_DIR`` in the environment. Returns ``None`` when neither is
    set, so the caller falls back to the current working directory.

    A vault selected this way owns its config: discovery walks up from the
    vault dir rather than from cwd. See ``resolve_config`` Layer 3.
    """
    raw: str | None = None
    if cli_overrides:
        raw = cli_overrides.get(PATHS_WIKI_DIR)
    if raw is None:
        raw = os.environ.get("KNOX_WIKI_DIR")
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


# ── TOML loading ────────────────────────────────────────────────────────


def _expand_env_vars(value: str) -> str:
    """Expand ``$VAR`` and ``${VAR}`` patterns in a string.

    Unknown variables are left as-is. This is applied to string values
    when loading TOML config files so users can reference env vars like
    ``$HOME`` or ``${KNOX_CLOUD_LLM}`` in path/model settings.
    """

    def _replace(match: re.Match) -> str:
        var = match.group(2) or match.group(1) or ""
        return os.environ.get(var, match.group(0))

    return re.sub(r"\$(\w+)|\$\{(\w+)\}", _replace, value)


def _expand_env_vars_recursive(data: Any) -> Any:
    """Recursively expand env vars in all string values."""
    if isinstance(data, str):
        return _expand_env_vars(data)
    if isinstance(data, dict):
        return {k: _expand_env_vars_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_expand_env_vars_recursive(v) for v in data]
    return data


def _strip_strings_recursive(data: Any) -> Any:
    """Recursively strip leading/trailing whitespace from all string values."""
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        return {k: _strip_strings_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_strip_strings_recursive(v) for v in data]
    return data


def _load_toml(toml_file: Path) -> dict[str, Any]:
    """Load a TOML file, returning an empty dict on failure.

    String values are processed through ``_expand_env_vars`` to support
    ``$VAR`` and ``${VAR}`` patterns.
    """
    try:
        raw = toml_file.read_bytes()
        data: dict[str, Any] = tomllib.loads(raw.decode("utf-8"))
        data = _expand_env_vars_recursive(data)
        data = _strip_strings_recursive(data)
        logger.debug("Loaded config from %s", toml_file)
        return data
    except Exception as exc:
        logger.warning("Failed to load config %s: %s", toml_file, exc, exc_info=True)
        return {}


# ── Env var layer ───────────────────────────────────────────────────────

# ── Env settings model (cached singleton) ──

# Env vars that carry comma-separated lists. These are stored as ``str``
# in the settings model and split into lists in ``_read_env_vars``.
# NOTE: ``KNOX_LINT`` is intentionally listed here even though its schema
# field (``compile.lint``) is a bool — the list form targets ``lint.enabled``.
_ENV_LIST_VARS: frozenset[str] = frozenset({
    "KNOX_LLM_FALLBACK_CHAIN",
    "KNOX_LINT",
    "KNOX_DEDUP_MATCHERS",
    "KNOX_SEARCH_EXPANSION_VARIANTS",
})


def _classify_env_field(ann: Any) -> tuple[Any, Any]:
    """Map a pydantic field annotation to a (type, default) tuple for the env model.

    List fields are kept as ``str`` (split later in ``_read_env_vars``).
    """
    if get_origin(ann) is list:
        return (str | None, Field(None))
    if ann is bool:
        return (bool | None, Field(None))
    if ann is float:
        return (float | None, Field(None))
    if ann is int:
        return (int | None, Field(None))
    return (str | None, Field(None))


@cache
def _get_env_settings_model() -> BaseSettings:
    """Build pydantic-settings model dynamically from _ENV_MAP.

    Field types are derived from the :class:`KnoxConfig` schema so adding a
    new config key with a typed annotation automatically gets the right env
    coercion — no manual ``if/elif`` ladder to keep in sync. Returns a cached
    singleton so the model class is only created once.
    """
    _KnoxEnvFields: dict[str, tuple[Any, Any]] = {}
    for _env_name in _ENV_MAP.values():
        _fn = _env_name.removeprefix("KNOX_").lower()
        if _env_name in _ENV_LIST_VARS:
            _KnoxEnvFields[_fn] = (str | None, Field(None))
            continue
        _field_info = _resolve_field_info(_env_map_key_for(_env_name))
        _ann = _field_info.annotation if _field_info is not None else None
        _KnoxEnvFields[_fn] = _classify_env_field(_ann)

    return create_model(
        "_KnoxEnvSettings",
        __base__=BaseSettings,
        __config__=SettingsConfigDict(env_prefix="KNOX_", extra="ignore"),
        **_KnoxEnvFields,
    )


def _env_map_key_for(env_name: str) -> str:
    """Return the dotted config key for an env var name (reverse _ENV_MAP lookup)."""
    for _key, _name in _ENV_MAP.items():
        if _name == env_name:
            return _key
    return ""


def _read_env_vars() -> dict[str, Any]:
    """Read KNOX_* env vars via pydantic-settings and return a nested config dict."""
    settings = _get_env_settings_model()()
    result: dict[str, Any] = {}
    for dotted_key, env_name in _ENV_MAP.items():
        field_name = env_name.removeprefix("KNOX_").lower()
        value = getattr(settings, field_name, None)
        if value is not None:
            # Strip whitespace for string values
            if isinstance(value, str):
                value = value.strip()
            # Comma-separated list fields
            if env_name in _ENV_LIST_VARS:
                if isinstance(value, str):
                    value = [v.strip() for v in value.split(",") if v.strip()]
                elif isinstance(value, list):
                    pass  # already a list
            _set_dot(result, dotted_key, value)
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


def _apply_layer(
    result: dict[str, Any],
    layer: dict[str, Any],
    provenance: dict[str, ConfigSource],
    source: ConfigSource,
) -> dict[str, Any]:
    """Merge *layer* into *result* and stamp provenance for its keys.

    Used by every precedence layer (defaults, global, project, env, cli) so
    the merge + provenance bookkeeping stays in one place. Only non-``None``
    leaf values from *layer* claim provenance — ``None`` means "unset".
    """
    merged = _deep_merge(result, layer)
    for key in _flat_keys(layer):
        if _get_dot(layer, key) is not None:
            provenance[key] = source
    return merged


def _merge_project_configs(
    project_configs: list[Path],
) -> tuple[dict[str, Any], dict[str, ConfigSource]]:
    """Merge a project config hierarchy (farthest-to-closest) into one layer.

    Returns the merged data and per-key provenance pointing at the winning file.
    Closest file wins, matching ``find_all_project_configs`` ordering.
    """
    merged: dict[str, Any] = {}
    prov: dict[str, ConfigSource] = {}
    for config_file in reversed(project_configs):  # closest wins
        file_data = _load_toml(config_file)
        for key in _flat_keys(file_data):
            prov[key] = ConfigSource("project", str(config_file))
        merged = _deep_merge(merged, file_data)
    return merged, prov


def _resolve_paths(
    result: dict[str, Any],
    provenance: dict[str, ConfigSource],
) -> None:
    """Normalize path values under ``paths``.

    - Resolves relative paths against ``paths.wiki_dir`` (or cwd).
    """
    paths = result.setdefault("paths", {})

    # Backwards compatibility: old [compile] path keys migrate to [paths]
    for old_key, new_key in (
        (COMPILE_SOURCES_DIR, PATHS_SOURCES_DIR),
        (COMPILE_OUTPUT_DIR, PATHS_OUTPUT_DIR),
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
    base_dir = Path(wiki_dir_raw).expanduser().resolve() if wiki_dir_raw else Path.cwd()
    paths["wiki_dir"] = base_dir

    for key in ("sources_dir", "output_dir", "cache_dir", "state_dir"):
        dotted = f"paths.{key}"
        value = paths.get(key)
        if value is None:
            value = _get_dot(DEFAULTS, dotted)
        resolved_dir = Path(value).expanduser()
        if not resolved_dir.is_absolute():
            resolved_dir = base_dir / resolved_dir
        paths[key] = resolved_dir.resolve()


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
            result = _apply_layer(result, global_data, provenance, ConfigSource("global", str(global_path)))

    # Layer 3: project config.
    # Precedence for locating the project config file:
    #   1. Explicit --config (uses that file directly).
    #   2. --wiki-dir / KNOX_WIKI_DIR: discover from that vault dir.
    #      Operating on a vault means reading that vault's config, not the
    #      cwd's — this is what users expect "--wiki-dir <vault>" to mean.
    #   3. Otherwise discover from cwd (walk up the tree).
    project_dir: Path | None = None
    project_configs: list[Path] = []
    if config_path is not None:
        project_dir = Path(config_path)
        if project_dir.is_file():
            project_data = _load_toml(project_dir)
            if project_data:
                result = _apply_layer(result, project_data, provenance, ConfigSource("project", str(project_dir)))
        else:
            logger.warning("Config file not found: %s", config_path)
    else:
        start_dir = _wiki_dir_for_discovery(cli_overrides)
        project_configs = find_all_project_configs(start_dir)
        if project_configs:
            # Merge from farthest to closest so closest wins.
            merged_project, project_provenance = _merge_project_configs(project_configs)
            if merged_project:
                result = _deep_merge(result, merged_project)
                provenance.update(project_provenance)

    # Layer 2: env vars
    env_data = _read_env_vars()
    if env_data:
        # Record per-key provenance with specific env var names
        result = _apply_layer(result, env_data, provenance, ConfigSource("env", "KNOX_*"))

    # Layer 1: CLI overrides
    cli = cli_overrides or {}
    if cli:
        cli_nested = _nest_flat(cli)
        result = _apply_layer(result, cli_nested, provenance, ConfigSource("cli"))

    _resolve_paths(result, provenance)

    return ResolvedConfig(
        raw=result,
        provenance=provenance,
        project_config_file=project_configs[0] if project_configs else project_dir,
        project_config_paths=project_configs,
        global_config_file=global_path,
    )


# ── Config mutation helpers ───────────────────────────────────────────────


def _flat_schema() -> set[str]:
    """Return all leaf dotted keys present in :data:`DEFAULTS`."""
    return set(_flat_keys(DEFAULTS))


def validate_config_key(key: str) -> bool:
    """Return ``True`` if *key* is a known leaf config key."""
    return key in _flat_schema()


def _resolve_field_info(key: str) -> FieldInfo | None:
    """Resolve a dotted config key to its pydantic FieldInfo in :class:`KnoxConfig`."""
    parts = key.split(".")
    cls: type[BaseModel] = KnoxConfig
    for part in parts[:-1]:
        info = cls.model_fields.get(part)
        if info is None:
            return None
        ann = info.annotation
        if not (isinstance(ann, type) and issubclass(ann, BaseModel)):
            return None
        cls = ann
    return cls.model_fields.get(parts[-1])


def coerce_config_value(value: str, key: str) -> Any:
    """Coerce a CLI string to the config schema type for *key*.

    Uses the :class:`KnoxConfig` model's field type for coercion via
    pydantic :class:`TypeAdapter`. List fields are split by comma (pydantic
    does not split strings into lists automatically). Unknown keys fall
    back to generic int/float/str coercion.

    Raises:
        ValueError: when *value* cannot be coerced to the expected type.
    """
    value = value.strip()
    field_info = _resolve_field_info(key)
    if field_info is None:
        with contextlib.suppress(ValueError):
            return int(value)
        with contextlib.suppress(ValueError):
            return float(value)
        return value

    ann = field_info.annotation
    # pydantic doesn't split comma-separated strings into lists
    if get_origin(ann) is list:
        return [v.strip() for v in value.split(",") if v.strip()]

    try:
        return TypeAdapter(ann).validate_python(value)
    except Exception as e:
        raise ValueError(f"cannot coerce {value!r} to {ann} for {key}") from e


def get_project_config_path(start_dir: Path | None = None) -> Path:
    """Return the project config file path used by ``knox config set``.

    Prefers ``.knox/config.toml`` over ``knox.toml``. When no project config
    exists, returns the expected path ``<start_dir>/.knox/config.toml``
    without creating it — callers must check existence.
    """
    current = (start_dir or Path.cwd()).resolve()
    found = find_project_config(current)
    if found is not None:
        return found
    return VaultPaths(current).config_path


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


def _atomic_write_toml(toml_file: Path, data: dict[str, Any]) -> None:
    """Write *data* to *toml_file* atomically, keeping a ``.bak`` copy."""
    if toml_file.is_file():
        backup = toml_file.with_suffix(toml_file.suffix + ".bak")
        backup.write_bytes(toml_file.read_bytes())
    tmp = toml_file.with_suffix(toml_file.suffix + ".tmp")
    tmp.write_text(tomli_w.dumps(data), encoding="utf-8")
    tmp.replace(toml_file)


def set_config_file_value(config_file: Path, key: str, value: Any) -> None:
    """Set *key* to *value* in *config_file*, creating the file if needed.

    Raises:
        KeyError: if *key* is not a known config key.
        ValueError: if *value* cannot be serialised as TOML.
    """
    if not validate_config_key(key):
        raise KeyError(f"unknown config key: {key}")
    data = _load_toml(config_file) if config_file.is_file() else {}
    _set_dot(data, key, value)
    _atomic_write_toml(config_file, data)
