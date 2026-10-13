"""kompend unified configuration system.

5-layer precedence (highest wins):

1. CLI flags           — explicit per-invocation overrides
2. Shell env vars      — ``KOMP_*`` per-session overrides
3. Project config      — ``./kompend.toml`` and ``./.kompend/config.toml``,
                         discovered hierarchically by walking up from cwd
4. Global config       — ``~/.config/kompend/config.toml`` (per-user)
5. Built-in defaults   — hard-coded in :data:`DEFAULTS`

See ``kompend.core.config.AGENTS.md`` for coding conventions.
"""

from kompend.core.config._configs import (
    BudgetConfig,
    DisplayConfig,
    EmbeddingConfig,
    ExportConfig,
    ExportMode,
    ImageMode,
    LLMConfig,
    LlmDataFormat,
    OrganizeConfig,
    OutputConfig,
    OutputFormat,
    RlmConfig,
)
from kompend.core.config._core import (
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
    find_all_project_configs,
    find_global_config,
    find_project_config,
    get_global_config_file,
    get_project_config_file,
    resolve_config,
    set_config_file_value,
    validate_config_key,
)
from kompend.core.config.generate import generate_scaffold_config

__all__ = [
    "DEFAULTS",
    "STATUS_DEPRECATED",
    "STATUS_ERROR",
    "STATUS_MODIFIED",
    "STATUS_OK",
    "STATUS_REMOVED",
    "BudgetConfig",
    "ConfigCheckEntry",
    "ConfigCheckReport",
    "ConfigSource",
    "DisplayConfig",
    "EmbeddingConfig",
    "ExportConfig",
    "ExportMode",
    "ImageMode",
    "LLMConfig",
    "LlmDataFormat",
    "OrganizeConfig",
    "OutputConfig",
    "OutputFormat",
    "ResolvedConfig",
    "RlmConfig",
    "check_config_file",
    "check_config_files",
    "coerce_config_value",
    "find_all_project_configs",
    "find_global_config",
    "find_project_config",
    "generate_scaffold_config",
    "get_global_config_file",
    "get_project_config_file",
    "resolve_config",
    "set_config_file_value",
    "validate_config_key",
]
