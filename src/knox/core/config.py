"""Config file loading for KNOX.

Handles loading ``.knox/config.yaml`` and merging with environment
variables and defaults.

Precedence (highest to lowest):
1. CLI flags (passed in externally)
2. Environment variables
3. ``.knox/config.yaml``
4. Built-in defaults
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Discovery ───────────────────────────────────────────────────────────


def find_config(start_dir: Path | None = None) -> Path | None:
    """Walk up from *start_dir* looking for ``.knox/config.yaml``.

    Returns the first found path, or ``None``.
    """
    current = (start_dir or Path.cwd()).resolve()
    for parent in [current] + list(current.parents):
        candidate = parent / ".knox" / "config.yaml"
        if candidate.is_file():
            return candidate
    return None


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load KNOX config from a YAML file.

    If *config_path* is ``None``, searches up from ``cwd``.
    Returns an empty dict if no config is found.
    """
    path = config_path or find_config()
    if path is None:
        return {}

    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        logger.warning("pyyaml not installed — cannot read config file %s", path)
        return {}

    try:
        raw = path.read_text(encoding="utf-8")
        data: dict[str, Any] = yaml.safe_load(raw) or {}
        logger.debug("Loaded config from %s", path)
        return data
    except Exception as exc:
        logger.warning("Failed to load config %s: %s", path, exc)
        return {}


# ── Config accessors ────────────────────────────────────────────────────


def get_config_value(
    key: str,
    config: dict[str, Any] | None = None,
    default: Any = None,
) -> Any:
    """Get a dotted key from config dict (e.g. ``output.flavor``)."""
    cfg = config or load_config()
    parts = key.split(".")
    current: Any = cfg
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return default
    return current if current is not None else default


def get_output_flavor(config: dict[str, Any] | None = None) -> str | None:
    """Get ``output.flavor`` from config, or ``None``."""
    return get_config_value("output.flavor", config)


def get_frontmatter_defaults(config: dict[str, Any] | None = None) -> dict[str, str]:
    """Get ``output.frontmatter.defaults`` from config."""
    raw = get_config_value("output.frontmatter.defaults", config, {})
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def get_rumdl_config_path(config: dict[str, Any] | None = None) -> Path | None:
    """Get ``output.rumdl_config`` from config, resolved as a path."""
    raw = get_config_value("output.rumdl_config", config)
    if raw is None:
        return None
    return Path(str(raw))


def get_ignore_patterns(config: dict[str, Any] | None = None) -> list[str]:
    """Get ``ignore`` from config (list of glob patterns)."""
    raw = get_config_value("ignore", config, [])
    if not isinstance(raw, list):
        return []
    return [str(p) for p in raw]
