"""Configuration engine — 5-layer precedence with provenance tracking.

Sub-modules (load in dependency order):
1. _schema — DEFAULTS, env map, validation
2. _env — env var reading via pydantic-settings
3. _loading — TOML loading, discovery
4. _precedence — merge, resolve, provenance
5. _overlay — config mutation helpers
"""

# Import in dependency order to avoid circular-ish chains.
# _schema has no engine deps → _env needs _schema → _loading has no engine deps
# → _precedence needs _schema, _env, _loading → _overlay needs _loading, _schema
from knox.core.config.engine._env import _read_env_vars
from knox.core.config.engine._loading import (
    _expand_env_vars,
    _expand_env_vars_recursive,
    _load_toml,
    _strip_strings_recursive,
    _wiki_dir_for_discovery,
    find_all_project_configs,
    find_global_config,
    find_project_config,
    get_global_config_path,
    get_project_config_path,
)
from knox.core.config.engine._overlay import (
    _atomic_write_toml,
    set_config_file_value,
)
from knox.core.config.engine._precedence import (
    ConfigSource,
    ResolvedConfig,
    _apply_layer,
    _deep_merge,
    _get_dot,
    _merge_project_configs,
    _nest_flat,
    _resolve_paths,
    _set_dot,
    resolve_config,
)
from knox.core.config.engine._schema import (
    _ENV_MAP,
    DEFAULTS,
    _build_defaults,
    _build_env_map,
    _flat_keys,
    _flat_schema,
    _resolve_field_info,
    coerce_config_value,
    validate_config_key,
)

__all__ = [
    "DEFAULTS",
    "_ENV_MAP",
    "ConfigSource",
    "ResolvedConfig",
    "_apply_layer",
    "_atomic_write_toml",
    "_build_defaults",
    "_build_env_map",
    "_deep_merge",
    "_expand_env_vars",
    "_expand_env_vars_recursive",
    "_flat_keys",
    "_flat_schema",
    "_get_dot",
    "_load_toml",
    "_merge_project_configs",
    "_nest_flat",
    "_read_env_vars",
    "_resolve_field_info",
    "_resolve_paths",
    "_set_dot",
    "_strip_strings_recursive",
    "_wiki_dir_for_discovery",
    "coerce_config_value",
    "find_all_project_configs",
    "find_global_config",
    "find_project_config",
    "get_global_config_path",
    "get_project_config_path",
    "resolve_config",
    "set_config_file_value",
    "validate_config_key",
]
