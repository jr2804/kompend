"""Generic nested-dict helpers for dotted-key access, merging, and flattening.

These are pure utilities with no domain logic. Used by the config engine
(``kompend.core.config.engine``) and any other module that needs dotted-key
traversal of nested ``dict[str, Any]`` structures.

ponytail: ``dict[str, Any]`` here is the correct signature (C3) — the whole
point of these helpers is arbitrary-depth, heterogeneous nested dicts.
"""

from __future__ import annotations

import copy
from typing import Any

from kompend.types import ConfigValue

__all__ = [
    "as_int",
    "as_str",
    "deep_merge",
    "del_dot",
    "flat_keys",
    "get_dot",
    "nest_flat",
    "resolve_dotted",
    "set_dot",
]


def as_str(value: ConfigValue) -> str | None:
    """Narrow an optional JSON/config value to ``str``.

    Raises TypeError on non-string values so misconfiguration fails fast
    instead of silently flowing into string-typed APIs.
    """
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"Expected a string value, got {type(value).__name__}: {value!r}")


def as_int(value: ConfigValue) -> int | None:
    """Narrow an optional JSON/config value to ``int`` (bools rejected)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected an integer value, got {type(value).__name__}: {value!r}")
    return value


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


def resolve_dotted(data: dict[str, Any], key: str) -> tuple[bool, Any]:
    """Resolve a dotted *key*, tolerating dict keys that themselves contain dots.

    :func:`get_dot` splits on every ``.``, so a name like a model id
    (``models."hf.co/qwen3.6"``) cannot be resolved: the split invents segments
    the dict does not have. This walks the path consuming the **longest** run of
    segments that names a key at each level, which resolves plain and dotted key
    names alike. Returns ``(found, value)``.

    Ambiguity (a dict holding both ``a.b`` and ``a: {b: ...}``) resolves to the
    longer key — a dotted name is one key, never a coincidence of nesting.
    """
    parts = key.split(".")
    node: Any = data
    index = 0
    while index < len(parts):
        if not isinstance(node, dict):
            return False, None
        for end in range(len(parts), index, -1):
            candidate = ".".join(parts[index:end])
            if candidate in node:
                node = node[candidate]
                index = end
                break
        else:
            return False, None
    return True, node


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


def del_dot(data: dict[str, Any], key: str) -> None:
    """Remove a dotted *key* from a nested dict; no-op when it is absent."""
    parts = key.split(".")
    current: Any = data
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


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
