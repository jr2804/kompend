"""Narrowing helpers for the heterogeneous node metadata mapping.

``WikiNode.metadata`` (and the wiki-level metadata) holds values of several
shapes: scalars, string lists, and nested mappings. Every consumer states the
shape it expects, so the narrowing happens once — here — instead of a cast at
each use site. A value of the wrong shape yields the empty/default result
instead of raising, because metadata is optional by nature.

Use :func:`meta_str_list` for tag-like lists, :func:`meta_str_dict` for
label mappings, :func:`meta_dict` for nested mappings whose values stay
untyped, and :func:`meta_str` / :func:`meta_int` / :func:`meta_float` for
scalars.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from kompend.types import MetadataValue

__all__ = [
    "meta_dict",
    "meta_dict_list",
    "meta_float",
    "meta_float_dict",
    "meta_int",
    "meta_list_dict",
    "meta_str",
    "meta_str_dict",
    "meta_str_list",
    "meta_str_list_dict",
]


def meta_str_list_dict(value: MetadataValue | None) -> dict[str, list[str]]:
    """The value as a mapping of keys to string lists; empty when it is not one."""
    if not isinstance(value, Mapping):
        return {}
    return {str(key): meta_str_list(item) for key, item in value.items()}


def meta_float_dict(value: MetadataValue | None) -> dict[str, float]:
    """The value as a mapping of keys to floats; empty when it is not one."""
    if not isinstance(value, Mapping):
        return {}
    return {str(key): meta_float(item) for key, item in value.items()}


def meta_str_list(value: MetadataValue | None) -> list[str]:
    """The value as a list of strings; empty when it is not a sequence."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        return []
    return [item if isinstance(item, str) else str(item) for item in value]


def meta_list_dict(value: MetadataValue | None) -> list[dict[str, str]]:
    """The value as a list of string mappings; empty when it is not one."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        return []
    return [meta_str_dict(item) for item in value if isinstance(item, Mapping)]


def meta_dict_list(value: MetadataValue | None) -> list[dict[str, MetadataValue]]:
    """The value as a list of nested mappings, values kept as they are."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        return []
    return [meta_dict(item) for item in value if isinstance(item, Mapping)]


def meta_dict(value: MetadataValue | None) -> dict[str, MetadataValue]:
    """The value as a nested mapping; empty when it is not one."""
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def meta_str_dict(value: MetadataValue | None) -> dict[str, str]:
    """The value as a mapping of strings to strings; empty when it is not one."""
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item if isinstance(item, str) else str(item) for key, item in value.items()}


def meta_str(value: MetadataValue | None, default: str = "") -> str:
    """The value as a string; *default* when it is not a scalar."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float)):
        return str(value)
    return default


def meta_int(value: MetadataValue | None, default: int = 0) -> int:
    """The value as an integer; *default* when it is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def meta_float(value: MetadataValue | None, default: float = 0.0) -> float:
    """The value as a float; *default* when it is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)
