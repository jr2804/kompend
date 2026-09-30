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
#
# Generic dict helpers live in knox.core.utils.dicts.
from knox.core.config.engine._check import (
    STATUS_DEPRECATED,
    STATUS_ERROR,
    STATUS_MODIFIED,
    STATUS_OK,
    STATUS_REMOVED,
    ConfigCheckEntry,
    ConfigCheckReport,
    check_config_file,
    check_config_files,
)
from knox.core.config.engine._env import read_env_vars
from knox.core.config.engine._loading import (
    find_all_project_configs,
    find_global_config,
    find_project_config,
    get_global_config_file,
    get_project_config_file,
    load_toml,
)
from knox.core.config.engine._overlay import set_config_file_value
from knox.core.config.engine._precedence import (
    ConfigSource,
    ResolvedConfig,
    resolve_config,
)
from knox.core.config.engine._schema import (
    DEFAULTS,
    coerce_config_value,
    validate_config_key,
)
from knox.core.utils.dicts import deep_merge, flat_keys, get_dot, set_dot

__all__ = [
    "DEFAULTS",
    "STATUS_DEPRECATED",
    "STATUS_ERROR",
    "STATUS_MODIFIED",
    "STATUS_OK",
    "STATUS_REMOVED",
    "ConfigCheckEntry",
    "ConfigCheckReport",
    "ConfigSource",
    "ResolvedConfig",
    "check_config_file",
    "check_config_files",
    "coerce_config_value",
    "deep_merge",
    "find_all_project_configs",
    "find_global_config",
    "find_project_config",
    "flat_keys",
    "get_dot",
    "get_global_config_file",
    "get_project_config_file",
    "load_toml",
    "read_env_vars",
    "resolve_config",
    "set_config_file_value",
    "set_dot",
    "validate_config_key",
]
