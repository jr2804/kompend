"""Rows for config keys retired from or renamed in the schema."""

from __future__ import annotations

from typing import Any

from kompend.core.config.engine._schema import DEPRECATED_ALIASES, REMOVED_KEYS
from kompend.core.utils.dicts import get_dot

from .messages import STATUS_DEPRECATED, STATUS_REMOVED, ConfigCheckEntry


def removed_leaf_keys() -> set[str]:
    """Flatten ``REMOVED_KEYS`` (section -> keys) into full dotted leaf keys."""
    out: set[str] = set()
    for section, keys in REMOVED_KEYS.items():
        out.update(f"{section}.{k}" for k in keys)
    return out


def removed_ancestor(key: str, removed_keys: set[str]) -> str | None:
    """Return the retired key that *key* equals or sits under, if any.

    Table-valued retired keys (e.g. ``compile.llm.role_models``) flatten to
    nested leaves like ``compile.llm.role_models.<role>``, so matching must be
    prefix-aware, not exact.
    """
    for removed in removed_keys:
        if key == removed or key.startswith(f"{removed}."):
            return removed
    return None


def removed_rows(raw: dict[str, Any], keys: set[str]) -> list[ConfigCheckEntry]:
    """One ``removed`` row per retired key the user still has set."""
    return [
        ConfigCheckEntry(
            key=key,
            status=STATUS_REMOVED,
            current=get_dot(raw, key),
            default=None,
            set_in_toml=True,
            message="key removed from schema; value is ignored at runtime — please delete it from your config",
        )
        for key in sorted(keys)
    ]


def deprecated_rows(raw: dict[str, Any], keys: set[str]) -> list[ConfigCheckEntry]:
    """One ``deprecated`` row per renamed key the user still has set."""
    return [
        ConfigCheckEntry(
            key=key,
            status=STATUS_DEPRECATED,
            current=get_dot(raw, key),
            default=get_dot(raw, DEPRECATED_ALIASES[key]),
            set_in_toml=True,
            message=f"deprecated — use '{DEPRECATED_ALIASES[key]}' instead; the value is migrated automatically",
        )
        for key in sorted(keys)
    ]
