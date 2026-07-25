"""Config overlay and mutation — set config values via CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli_w

from knox.core.config.engine._loading import load_toml
from knox.core.config.engine._schema import validate_config_key
from knox.core.utils.dicts import set_dot
from knox.logging import get_logger

logger = get_logger(__name__)


def _atomic_write_toml(toml_file: Path, data: dict[str, Any]) -> None:
    """Write *data* to *toml_file* atomically, keeping a ``.bak`` copy."""
    if toml_file.is_file():
        backup = toml_file.with_suffix(toml_file.suffix + ".bak")
        backup.write_bytes(toml_file.read_bytes())
    tmp = toml_file.with_suffix(toml_file.suffix + ".tmp")
    tmp.write_text(tomli_w.dumps(data), encoding="utf-8")
    tmp.replace(toml_file)


def set_config_file_value(config_file: Path, key: str, value: Any) -> None:
    """Set *key* to *value* in *config_file*, creating the file if needed.

    Raises:
        KeyError: if *key* is not a known config key.
        ValueError: if *value* cannot be serialised as TOML.
    """
    if not validate_config_key(key):
        raise KeyError(f"unknown config key: {key}")
    data = load_toml(config_file) if config_file.is_file() else {}
    set_dot(data, key, value)
    _atomic_write_toml(config_file, data)
