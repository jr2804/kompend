"""Config schema introspection: DEFAULTS, env map, validation."""

from __future__ import annotations

import contextlib
from typing import Any, get_origin

from pydantic import BaseModel, TypeAdapter
from pydantic.fields import FieldInfo

from knox.core.config._configs import KnoxConfig
from knox.core.config.keys import (
    COMPILE_CACHE_DIR,
    COMPILE_RESPECT_GITIGNORE,
    COMPILE_START_AT_TIER,
    COMPILE_STOP_AT_TIER,
    COMPILE_TIER_STATE_PATH,
)
from knox.core.utils.dicts import flat_keys
from knox.logging import get_logger
from knox.types import ConfigValue

logger = get_logger(__name__)


def _build_defaults() -> dict[str, Any]:
    """Generate DEFAULTS from the :class:`KnoxConfig` schema model."""
    return KnoxConfig().model_dump()


def _build_env_map() -> dict[str, str]:
    """Auto-generate _ENV_MAP from :class:`KnoxConfig` validation aliases."""
    env_map: dict[str, str] = {}

    def _walk(cls: type[BaseModel], prefix: str = "") -> None:
        for name, info in cls.model_fields.items():
            dotted = f"{prefix}.{name}" if prefix else name
            ann = info.annotation
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                _walk(ann, dotted)
            else:
                alias = info.validation_alias
                if isinstance(alias, str):
                    env_map[dotted] = alias

    _walk(KnoxConfig)
    env_map.update(
        {
            COMPILE_START_AT_TIER: "KNOX_START_AT_TIER",
            COMPILE_STOP_AT_TIER: "KNOX_STOP_AT_TIER",
            COMPILE_TIER_STATE_PATH: "KNOX_TIER_STATE_PATH",
            COMPILE_CACHE_DIR: "KNOX_COMPILE_CACHE_DIR",
            COMPILE_RESPECT_GITIGNORE: "KNOX_RESPECT_GITIGNORE",
        }
    )
    return env_map


DEFAULTS: dict[str, Any] = _build_defaults()


ENV_MAP: dict[str, str] = _build_env_map()


def validate_config_key(key: str) -> bool:
    """Return ``True`` if *key* is a known leaf config key."""
    return key in _flat_schema()


def _flat_schema() -> set[str]:
    """Return all leaf dotted keys present in :data:`DEFAULTS`."""
    return set(flat_keys(DEFAULTS))


def coerce_config_value(value: str, key: str) -> ConfigValue:
    """Coerce a CLI string to the config schema type for *key*.

    Raises:
        ValueError: when *value* cannot be coerced to the expected type.
    """
    value = value.strip()
    field_info = resolve_field_info(key)
    if field_info is None:
        with contextlib.suppress(ValueError):
            return int(value)
        with contextlib.suppress(ValueError):
            return float(value)
        return value

    ann = field_info.annotation
    if get_origin(ann) is list:
        return [v.strip() for v in value.split(",") if v.strip()]

    try:
        return TypeAdapter(ann).validate_python(value)
    except Exception as e:
        raise ValueError(f"cannot coerce {value!r} to {ann} for {key}") from e


def resolve_field_info(key: str) -> FieldInfo | None:
    """Resolve a dotted config key to its pydantic FieldInfo in :class:`KnoxConfig`."""
    parts = key.split(".")
    cls: type[BaseModel] = KnoxConfig
    for part in parts[:-1]:
        info = cls.model_fields.get(part)
        if info is None:
            return None
        ann = info.annotation
        if not (isinstance(ann, type) and issubclass(ann, BaseModel)):
            return None
        cls = ann
    return cls.model_fields.get(parts[-1])
