"""Cross-key coherence: validate the merged config through :class:`KompendConfig`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kompend.core.config.engine._schema import DEFAULTS
from kompend.core.config.models.root import KompendConfig
from kompend.logging import get_logger

logger = get_logger("kompend.core.config.engine._check")


def collect_validation_errors(raw: dict[str, Any], user_keys: set[str]) -> dict[str, str]:
    """Validate ``raw`` on top of the defaults and index errors by user key."""
    merged: dict[str, Any] = overlay_merge(safe_dump_defaults(), raw)
    validation_error_keys: dict[str, str] = {}
    try:
        KompendConfig.model_validate(merged)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            if loc and loc in user_keys:
                validation_error_keys.setdefault(loc, err.get("msg", "validation error"))
    except Exception as exc:
        logger.debug("Unexpected config validation exception: %s", exc)
    return validation_error_keys


def overlay_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` on top of ``base`` (last-wins per leaf)."""
    result: dict[str, Any] = {}
    for key, base_val in base.items():
        if key in overlay and isinstance(base_val, dict) and isinstance(overlay[key], dict):
            result[key] = overlay_merge(base_val, overlay[key])
        else:
            result[key] = base_val
    for key, overlay_val in overlay.items():
        if key not in base or not (isinstance(base[key], dict) and isinstance(overlay_val, dict)):
            result[key] = overlay_val
    return result


def safe_dump_defaults() -> dict[str, Any]:
    """Make a JSON-safe copy of DEFAULTS suitable for ``model_validate``.

    ``DEFAULTS`` comes from :func:`KompendConfig().model_dump()` but recursive
    pydantic validation may trip on intrinsic scalars pydantic re-emits as
    Path; ``Path`` is fine for validation but we also pass the user's dict
    in. Defensive copy normalises Path → str (matches :func:`_coerce_paths_to_str`
    used elsewhere in the engine).
    """

    def _walk(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _walk(inner) for k, inner in v.items()}
        if isinstance(v, list):
            return [_walk(x) for x in v]
        if isinstance(v, Path):
            return str(v)
        return v

    return _walk(DEFAULTS)
