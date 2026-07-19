"""Unified configuration system for KNOX.

5-layer precedence (highest wins):

1. CLI flags           — explicit per-invocation overrides
2. Shell env vars      — ``KNOX_*`` per-session overrides
3. Project config      — ``./knox.toml`` and ``./.knox/config.toml``,
                         discovered hierarchically by walking up from cwd
4. Global config       — ``~/.config/knox/config.toml`` (per-user)
5. Built-in defaults   — hard-coded in :data:`DEFAULTS`

Config files use TOML. All implementation lives in :mod:`knox.core.config.engine`;
this module re-exports the public API for backward compatibility.
"""

from knox.core.config.engine import (  # noqa: F401
    _ENV_MAP,
    DEFAULTS,
    ConfigSource,
    ResolvedConfig,
    _apply_layer,
    _atomic_write_toml,
    _build_defaults,
    _build_env_map,
    _deep_merge,
    _expand_env_vars,
    _expand_env_vars_recursive,
    _flat_keys,
    _flat_schema,
    _get_dot,
    _load_toml,
    _merge_project_configs,
    _nest_flat,
    _read_env_vars,
    _resolve_field_info,
    _resolve_paths,
    _set_dot,
    _strip_strings_recursive,
    _wiki_dir_for_discovery,
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
