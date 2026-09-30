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
    DEFAULTS,
    STATUS_DEPRECATED,
    STATUS_ERROR,
    STATUS_MODIFIED,
    STATUS_OK,
    STATUS_REMOVED,
    ConfigCheckEntry,
    ConfigCheckReport,
    ConfigSource,
    ResolvedConfig,
    check_config_file,
    check_config_files,
    coerce_config_value,
    deep_merge,
    find_all_project_configs,
    find_global_config,
    find_project_config,
    flat_keys,
    get_dot,
    get_global_config_file,
    get_project_config_file,
    load_toml,
    read_env_vars,
    resolve_config,
    set_config_file_value,
    set_dot,
    validate_config_key,
)
