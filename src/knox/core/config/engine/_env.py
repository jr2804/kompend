"""Environment variable layer — read KNOX_* vars via pydantic-settings."""

from __future__ import annotations

from functools import cache
from typing import Any, get_origin

from pydantic import Field, create_model
from pydantic_settings import BaseSettings, SettingsConfigDict

from knox.core.config.engine._schema import _ENV_MAP, _resolve_field_info

# Env vars that carry comma-separated lists.
_ENV_LIST_VARS: frozenset[str] = frozenset(
    {
        "KNOX_LLM_FALLBACK_CHAIN",
        "KNOX_LINT",
        "KNOX_DEDUP_MATCHERS",
        "KNOX_SEARCH_EXPANSION_VARIANTS",
    }
)


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


@cache
def _get_env_settings_model() -> BaseSettings:
    """Build pydantic-settings model dynamically from _ENV_MAP."""
    _KnoxEnvFields: dict[str, tuple[Any, Any]] = {}
    for _env_name in _ENV_MAP.values():
        _fn = _env_name.removeprefix("KNOX_").lower()
        if _env_name in _ENV_LIST_VARS:
            _KnoxEnvFields[_fn] = (str | None, Field(None))
            continue
        _field_info = _resolve_field_info(_env_map_key_for(_env_name))
        _ann = _field_info.annotation if _field_info is not None else None
        _KnoxEnvFields[_fn] = _classify_env_field(_ann)

    return create_model(
        "_KnoxEnvSettings",
        __base__=BaseSettings,
        __config__=SettingsConfigDict(env_prefix="KNOX_", extra="ignore"),
        **_KnoxEnvFields,
    )  # ty: ignore


def _env_map_key_for(env_name: str) -> str:
    """Return the dotted config key for an env var name (reverse _ENV_MAP lookup)."""
    return next((key for key, name in _ENV_MAP.items() if name == env_name), "")


def _read_env_vars() -> dict[str, Any]:
    """Read KNOX_* env vars via pydantic-settings and return a nested config dict."""
    settings = _get_env_settings_model()()  # ty: ignore
    result: dict[str, Any] = {}

    def _set_dot(data: dict[str, Any], key: str, value: Any) -> None:
        parts = key.split(".")
        current = data
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    for dotted_key, env_name in _ENV_MAP.items():
        field_name = env_name.removeprefix("KNOX_").lower()
        value = getattr(settings, field_name, None)
        if value is not None:
            if isinstance(value, str):
                value = value.strip()
            if env_name in _ENV_LIST_VARS and isinstance(value, str):
                value = [v.strip() for v in value.split(",") if v.strip()]
            _set_dot(result, dotted_key, value)
    return result
