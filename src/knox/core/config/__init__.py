"""KNOX unified configuration system.

5-layer precedence (highest wins):

1. CLI flags           — explicit per-invocation overrides
2. Shell env vars      — ``KNOX_*`` per-session overrides
3. Project config      — ``./knox.toml`` and ``./.knox/config.toml``,
                         discovered hierarchically by walking up from cwd
4. Global config       — ``~/.config/knox/config.toml`` (per-user)
5. Built-in defaults   — hard-coded in :data:`DEFAULTS`

See ``knox.core.config.AGENTS.md`` for coding conventions.
"""

from knox.core.config._configs import BudgetConfig, EmbeddingConfig, ImageMode, LLMConfig, OutputConfig, OutputFormat
from knox.core.config._core import (
    DEFAULTS,
    ConfigSource,
    ResolvedConfig,
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

__all__ = [
    "DEFAULTS",
    "BudgetConfig",
    "ConfigSource",
    "EmbeddingConfig",
    "ImageMode",
    "LLMConfig",
    "OutputConfig",
    "OutputFormat",
    "ResolvedConfig",
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
