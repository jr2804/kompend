"""Environment variable layer — read KNOX_* vars via pydantic-settings."""

from __future__ import annotations

from functools import cache
from typing import Any, cast, get_origin

from pydantic import ConfigDict, Field, create_model
from pydantic_settings import BaseSettings, SettingsConfigDict

from knox.core.config.engine._schema import ENV_MAP, resolve_field_info
from knox.core.utils.dicts import set_dot


def _is_list_env_key(dotted: str) -> bool:
    """Return True if *dotted* config key maps to a comma-separated list field."""
    info = resolve_field_info(dotted)
    return info is not None and get_origin(info.annotation) is list


# Env vars that carry comma-separated lists, derived from the schema's
# list-typed fields so the set can't drift from the config-map.
_ENV_LIST_VARS: frozenset[str] = frozenset(env for dotted, env in ENV_MAP.items() if _is_list_env_key(dotted))


def read_env_vars() -> dict[str, Any]:
    """Read KNOX_* env vars via pydantic-settings and return a nested config dict."""
    settings = _get_env_settings_model()()
    result: dict[str, Any] = {}

    for dotted_key, env_name in ENV_MAP.items():
        field_name = env_name.removeprefix("KNOX_").lower()
        value = getattr(settings, field_name, None)
        if value is not None:
            if isinstance(value, str):
                value = value.strip()
            if env_name in _ENV_LIST_VARS and isinstance(value, str):
                value = [v.strip() for v in value.split(",") if v.strip()]
            set_dot(result, dotted_key, value)
    return result


@cache
def _get_env_settings_model() -> type[BaseSettings]:
    """Build pydantic-settings model dynamically from _ENV_MAP."""
    _KnoxEnvFields: dict[str, Any] = {}
    for _env_name in ENV_MAP.values():
        _fn = _env_name.removeprefix("KNOX_").lower()
        if _env_name in _ENV_LIST_VARS:
            _KnoxEnvFields[_fn] = (str | None, Field(None))
            continue
        _field_info = resolve_field_info(_env_map_key_for(_env_name))
        _ann = _field_info.annotation if _field_info is not None else None
        _KnoxEnvFields[_fn] = _classify_env_field(_ann)

    model = create_model(
        "_KnoxEnvSettings",
        __base__=BaseSettings,
        __config__=cast("ConfigDict", SettingsConfigDict(env_prefix="KNOX_", extra="ignore")),
        **_KnoxEnvFields,
    )
    return model


def _classify_env_field(ann: Any) -> tuple[Any, Any]:
    """Map a pydantic field annotation to a (type, default) tuple for the env model."""
    if get_origin(ann) is list:
        return (str | None, Field(None))
    if ann is bool:
        return (bool | None, Field(None))
    if ann is float:
        return (float | None, Field(None))
    if ann is int:
        return (int | None, Field(None))
    return (str | None, Field(None))


def _env_map_key_for(env_name: str) -> str:
    """Return the dotted config key for an env var name (reverse _ENV_MAP lookup)."""
    return next((key for key, name in ENV_MAP.items() if name == env_name), "")
