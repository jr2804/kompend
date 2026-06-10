"""KNOX unified configuration system.

5-layer precedence (highest wins):

1. CLI flags           — explicit per-invocation overrides
2. Shell env vars      — ``KNOX_*`` per-session overrides
3. Project config      — ``./knox.toml`` (checked into repo)
4. Global config       — ``~/.config/knox/config.toml`` (per-user)
5. Built-in defaults   — hard-coded in :data:`DEFAULTS`

See ``knox.core.config.AGENTS.md`` for coding conventions.
"""

from knox.core.config._configs import BudgetConfig, EmbeddingConfig, LLMConfig, OutputConfig
from knox.core.config._core import (
    DEFAULTS,
    ConfigSource,
    ResolvedConfig,
    find_global_config,
    find_project_config,
    resolve_config,
)

__all__ = [
    "DEFAULTS",
    "BudgetConfig",
    "ConfigSource",
    "EmbeddingConfig",
    "LLMConfig",
    "OutputConfig",
    "ResolvedConfig",
    "find_global_config",
    "find_project_config",
    "resolve_config",
]
