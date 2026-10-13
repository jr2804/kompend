"""TOML loading, env var expansion, and config file discovery."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any

from kompend.core.config.keys import PATHS_WIKI_DIR, EnvVar
from kompend.core.paths import CONFIG_FILE, KOMP_CONFIG_FILE, PROJECT_CONFIG_FILE, VaultPaths
from kompend.logging import get_logger
from kompend.types import JSONDict, JSONValue

# ── File paths ───────────────────────────────────────────────────────────

_PROJECT_CONFIG_NAMES: tuple[str, ...] = (KOMP_CONFIG_FILE, PROJECT_CONFIG_FILE)
_GLOBAL_CONFIG_DIR: str = "kompend"
_GLOBAL_CONFIG_NAME: str = CONFIG_FILE

logger = get_logger(__name__)


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


def _expand_env_vars_recursive(data: JSONDict) -> JSONDict:
    """Recursively expand env vars in all string values of a config table."""

    def walk(value: JSONValue) -> JSONValue:
        if isinstance(value, str):
            return _expand_env_vars(value)
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return {k: walk(v) for k, v in data.items()}


# ── Env var expansion ───────────────────────────────────────────────────


def _expand_env_vars(value: str) -> str:
    """Expand ``$VAR`` and ``${VAR}`` patterns in a string."""

    def _replace(match: re.Match) -> str:
        var = match.group(2) or match.group(1) or ""
        return os.environ.get(var, match.group(0))

    return re.sub(r"\$(\w+)|\$\{(\w+)\}", _replace, value)


def _strip_strings_recursive(data: JSONDict) -> JSONDict:
    """Recursively strip leading/trailing whitespace from all string values."""

    def walk(value: JSONValue) -> JSONValue:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return {k: walk(v) for k, v in data.items()}


def find_all_project_configs(start_dir: Path | None = None) -> list[Path]:
    """Walk up from *start_dir* collecting all project config files.

    Per directory, a gitignored local overlay (``<config>.local.toml``,
    mise-style) is listed before its committed sibling so the closest-wins
    merge in :func:`_merge_project_configs` applies it last.
    """
    current = VaultPaths(start_dir).base_dir
    found: list[Path] = []
    seen_dirs: set[Path] = set()
    for parent in [current] + list(current.parents):
        if parent in seen_dirs:
            continue
        seen_dirs.add(parent)
        for name in _PROJECT_CONFIG_NAMES:
            candidate = parent / name
            if candidate.is_file():
                local = candidate.with_name(candidate.name.removesuffix(".toml") + ".local.toml")
                if local.is_file():
                    found.append(local)
                found.append(candidate)
                break
    return found


def find_global_config() -> Path | None:
    """Return the global config path if it exists.

    Location follows XDG convention: ``$XDG_CONFIG_HOME/kompend/config.toml``.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_dir = Path(xdg) if xdg else Path.home() / ".config"
    candidate = config_dir / _GLOBAL_CONFIG_DIR / _GLOBAL_CONFIG_NAME
    return candidate if candidate.is_file() else None


def wiki_dir_for_discovery(cli_overrides: dict[str, Any] | None) -> Path | None:
    """Return the dir to start project-config discovery from.

    Reflects explicit vault selection (``--wiki-dir`` or ``KOMP_WIKI_DIR``).

    Kept as a direct read rather than delegating to ``vault_paths_from_ctx``:
    this runs during config discovery, below the CLI layer, so it can't depend
    on ``ctx``.
    """
    raw: str | None = None
    if cli_overrides:
        raw = cli_overrides.get(PATHS_WIKI_DIR)
    if raw is None:
        # KOMP_WIKI_DIR is the config-map env alias for PATHS_WIKI_DIR; read it
        # directly here because config discovery needs the dir to start from.
        raw = os.environ.get(EnvVar.WIKI_DIR)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def get_project_config_file(start_dir: Path | None = None) -> Path:
    """Return the project config file path used by ``kompend config set``."""
    current = VaultPaths(start_dir).base_dir
    found = find_project_config(current)
    if found is not None:
        return found
    return VaultPaths(current).config_file


# ── Config file discovery ───────────────────────────────────────────────


def find_project_config(start_dir: Path | None = None) -> Path | None:
    """Walk up from *start_dir* looking for project config files."""
    current = VaultPaths(start_dir).base_dir
    for parent in [current] + list(current.parents):
        for name in _PROJECT_CONFIG_NAMES:
            candidate = parent / name
            if candidate.is_file():
                return candidate
    return None


def get_global_config_file(*, create: bool = False) -> Path:
    """Return the global config file path (XDG convention)."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_dir = Path(xdg) if xdg else Path.home() / ".config"
    target = config_dir / _GLOBAL_CONFIG_DIR / _GLOBAL_CONFIG_NAME
    if create:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target
