"""Generic nested-dict helpers for dotted-key access, merging, and flattening.

These are pure utilities with no domain logic. Used by the config engine
(``knox.core.config.engine``) and any other module that needs dotted-key
traversal of nested ``dict[str, Any]`` structures.
"""

from __future__ import annotations

import copy
from typing import Any

from knox.types import ConfigValue

__all__ = [
    "deep_merge",
    "flat_keys",
    "get_dot",
    "nest_flat",
    "set_dot",
]


def get_dot(
    data: dict[str, ConfigValue],
    key: str,
    default: ConfigValue = None,
) -> ConfigValue:
    """Retrieve a dotted *key* from a nested dict, returning *default* if absent."""
    parts = key.split(".")
    current: ConfigValue = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return default
    return current if current is not None else default


def nest_flat(flat: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat dotted-key dict to a nested dict."""
    result: dict[str, Any] = {}
    for key, value in flat.items():
        set_dot(result, key, value)
    return result


def set_dot(data: dict[str, Any], key: str, value: Any) -> None:
    """Set a dotted *key* in a nested dict, creating intermediate dicts."""
    parts = key.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def flat_keys(data: dict[str, Any], prefix: str = "") -> list[str]:
    """Return all dotted keys in a nested dict (descends into sub-dicts)."""
    keys: list[str] = []
    for key, value in data.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys.extend(flat_keys(value, full))
        else:
            keys.append(full)
    return keys


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Dicts are merged key-by-key; non-dict values in *override* replace those
    in *base*. The input dicts are not mutated.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value) if isinstance(value, dict) else value
    return result
