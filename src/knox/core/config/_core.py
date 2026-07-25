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
    ConfigSource,
    ResolvedConfig,
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
