"""TOML loading, env var expansion, and config file discovery."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any

from knox.core.config.keys import PATHS_WIKI_DIR
from knox.core.paths import CONFIG_FILE, KNOX_CONFIG_FILE, PROJECT_CONFIG_FILE, VaultPaths
from knox.logging import get_logger
from knox.types import ConfigValue

logger = get_logger(__name__)

# ── File paths ───────────────────────────────────────────────────────────

_PROJECT_CONFIG_NAMES: tuple[str, ...] = (KNOX_CONFIG_FILE, PROJECT_CONFIG_FILE)
_GLOBAL_CONFIG_DIR: str = "knox"
_GLOBAL_CONFIG_NAME: str = CONFIG_FILE

# ── Env var expansion ───────────────────────────────────────────────────


def _expand_env_vars(value: str) -> str:
    """Expand ``$VAR`` and ``${VAR}`` patterns in a string."""

    def _replace(match: re.Match) -> str:
        var = match.group(2) or match.group(1) or ""
        return os.environ.get(var, match.group(0))

    return re.sub(r"\$(\w+)|\$\{(\w+)\}", _replace, value)


def _expand_env_vars_recursive(data: ConfigValue) -> ConfigValue:
    """Recursively expand env vars in all string values."""
    if isinstance(data, str):
        return _expand_env_vars(data)
    if isinstance(data, dict):
        return {k: _expand_env_vars_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_expand_env_vars_recursive(v) for v in data]
    return data


def _strip_strings_recursive(data: ConfigValue) -> ConfigValue:
    """Recursively strip leading/trailing whitespace from all string values."""
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        return {k: _strip_strings_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_strip_strings_recursive(v) for v in data]
    return data


def load_toml(toml_file: Path) -> dict[str, Any]:
    """Load a TOML file, returning an empty dict on failure."""
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


# ── Config file discovery ───────────────────────────────────────────────


def find_project_config(start_dir: Path | None = None) -> Path | None:
    """Walk up from *start_dir* looking for project config files."""
    current = (start_dir or Path.cwd()).resolve()
    for parent in [current] + list(current.parents):
        for name in _PROJECT_CONFIG_NAMES:
            candidate = parent / name
            if candidate.is_file():
                return candidate
    return None


def find_all_project_configs(start_dir: Path | None = None) -> list[Path]:
    """Walk up from *start_dir* collecting all project config files."""
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

    Location follows XDG convention: ``$XDG_CONFIG_HOME/knox/config.toml``.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_dir = Path(xdg) if xdg else Path.home() / ".config"
    candidate = config_dir / _GLOBAL_CONFIG_DIR / _GLOBAL_CONFIG_NAME
    return candidate if candidate.is_file() else None


def wiki_dir_for_discovery(cli_overrides: dict[str, Any] | None) -> Path | None:
    """Return the dir to start project-config discovery from.

    Reflects explicit vault selection (``--wiki-dir`` or ``KNOX_WIKI_DIR``).
    """
    raw: str | None = None
    if cli_overrides:
        raw = cli_overrides.get(PATHS_WIKI_DIR)
    if raw is None:
        raw = os.environ.get("KNOX_WIKI_DIR")
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def get_project_config_path(start_dir: Path | None = None) -> Path:
    """Return the project config file path used by ``knox config set``."""
    current = (start_dir or Path.cwd()).resolve()
    found = find_project_config(current)
    if found is not None:
        return found
    return VaultPaths(current).config_path


def get_global_config_path(*, create: bool = False) -> Path:
    """Return the global config file path (XDG convention)."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_dir = Path(xdg) if xdg else Path.home() / ".config"
    target = config_dir / _GLOBAL_CONFIG_DIR / _GLOBAL_CONFIG_NAME
    if create:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target
