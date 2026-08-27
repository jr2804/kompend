"""Diff a user-supplied config TOML against the built-in :data:`DEFAULTS`.

Used by the ``knox config check`` CLI command. For each provided file:

- Walks every leaf key in the live :class:`KnoxConfig` schema (so nested
  model fields like ``compile.llm.rlm.enable`` are reachable even when the
  parent defaults to ``None``).
- For every key the user actually set in the TOML, report:
    * ``ok``         — value matches the schema default
    * ``modified``   — value is set and differs from the default
    * ``error``      — value fails pydantic validation
- Reports ``ok`` for every schema key the user did *not* set (the resolved
  value equals the default).
- Adds one ``error`` row per leaf key present in the TOML but unknown to
  the schema (catches typos and removed fields).
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from knox.core.config._configs import KnoxConfig
from knox.core.config.engine._loading import load_toml
from knox.core.config.engine._schema import DEFAULTS, REMOVED_KEYS
from knox.core.utils.dicts import flat_keys, get_dot
from knox.logging import get_logger
from knox.types import JSONDict, JSONValue

# ── Status kind ─────────────────────────────────────────────────────────────

STATUS_OK = "ok"
STATUS_MODIFIED = "modified"
STATUS_ERROR = "error"
STATUS_REMOVED = "removed"  # key present in TOML but removed from schema

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class ConfigCheckEntry:
    """One row in a config-check report.

    `current` is the user's value when the key is set in the TOML, or
    ``None`` when the user did not set it and we are reporting the
    default-resolved value.
    """

    key: str
    status: str  # one of STATUS_OK / STATUS_MODIFIED / STATUS_ERROR
    current: JSONValue  # None when not set in user TOML
    default: JSONValue  # value in DEFAULTS (None for keys unknown to schema)
    set_in_toml: bool = False  # False → "not set; default resolves here"
    message: str | None = None  # populated only for error rows

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "key": self.key,
            "status": self.status,
            "default": self.default,
        }
        if self.set_in_toml or self.status != STATUS_OK:
            out["current"] = self.current
        if self.message is not None:
            out["message"] = self.message
        return out


@dataclasses.dataclass(frozen=True)
class ConfigCheckReport:
    """Full result for one config file (or merged set)."""

    path: Path | None  # None when the report covers multiple files merged
    entries: tuple[ConfigCheckEntry, ...]
    files_scanned: tuple[Path, ...] = ()

    @property
    def ok_count(self) -> int:
        return sum(1 for e in self.entries if e.status == STATUS_OK)

    @property
    def modified_count(self) -> int:
        return sum(1 for e in self.entries if e.status == STATUS_MODIFIED)

    @property
    def error_count(self) -> int:
        return sum(1 for e in self.entries if e.status == STATUS_ERROR)

    @property
    def removed_count(self) -> int:
        return sum(1 for e in self.entries if e.status == STATUS_REMOVED)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def to_dict(self) -> JSONDict:
        out: JSONDict = {
            "ok": self.ok_count,
            "modified": self.modified_count,
            "error": self.error_count,
            "removed": self.removed_count,
            "entries": [e.to_dict() for e in self.entries],
        }
        if self.path is not None:
            out["path"] = str(self.path)
        if self.files_scanned:
            out["files"] = [str(p) for p in self.files_scanned]
        return out


# ── Public API ──────────────────────────────────────────────────────────────


def check_config_file(path: Path) -> ConfigCheckReport:
    """Check one config file against the live :class:`KnoxConfig` schema.

    A row is emitted for every leaf key known to the schema; unknown keys
    present in the file surface as additional error rows. Keys the user
    did not set are reported as ``ok`` (default).
    """
    if not path.exists():
        entries = _all_defaults_as_errors(path, "file does not exist")
        return ConfigCheckReport(path=path, entries=entries)
    raw = load_toml(path)
    return _check_loaded(raw=raw, path=path, files_scanned=(path,))


def check_config_files(paths: list[Path]) -> ConfigCheckReport:
    """Check multiple files as a merged overlay (later wins).

    Useful for ``knox config check ./base.toml ./local.toml`` — the local
    file overrides the base, and the diff is against the resulting merged
    config.
    """
    raw: dict[str, object] = {}
    files_scanned: list[Path] = []
    for path in paths:
        if not path.exists():
            entries = _all_defaults_as_errors(path, "file does not exist")
            return ConfigCheckReport(path=None, entries=entries, files_scanned=tuple(paths))
        raw = _overlay_merge(raw, load_toml(path))
        files_scanned.append(path)
    return _check_loaded(
        raw=raw,
        path=None,
        files_scanned=tuple(files_scanned),
    )


def _all_defaults_as_errors(path: Path, message: str) -> tuple[ConfigCheckEntry, ...]:
    """Return one ``error`` entry per default key when the file can't be loaded."""
    return tuple(
        ConfigCheckEntry(
            key=k,
            status=STATUS_ERROR,
            current=None,
            default=get_dot(DEFAULTS, k),
            message=f"{path}: {message}",
        )
        for k in sorted(_schema_leaf_keys())
    )


def _check_loaded(
    *,
    raw: dict[str, Any],
    path: Path | None,
    files_scanned: tuple[Path, ...],
) -> ConfigCheckReport:
    """Diff ``raw`` against schema defaults and validate via :class:`KnoxConfig`."""
    schema_keys = _schema_leaf_keys()
    schema_default_values = _schema_defaults()
    user_keys = set(flat_keys(raw))
    removed_leaf_keys = _removed_leaf_keys()
    unknown_keys = sorted((user_keys - schema_keys) - removed_leaf_keys)
    removed_keys = sorted(user_keys & removed_leaf_keys)

    # Validation pass via pydantic to surface type/value errors.
    merged: dict[str, Any] = _overlay_merge(_safe_dump_defaults(), raw)
    validation_error_keys: dict[str, str] = {}
    try:
        KnoxConfig.model_validate(merged)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            if loc and loc in user_keys:
                validation_error_keys.setdefault(loc, err.get("msg", "validation error"))
    except Exception as exc:
        logger.debug("Unexpected config validation exception: %s", exc)

    entries: list[ConfigCheckEntry] = []
    for key in sorted(schema_keys):
        default_value = schema_default_values.get(key, get_dot(DEFAULTS, key))
        in_user_toml = key in user_keys
        user_value = get_dot(raw, key) if in_user_toml else None

        # Leaves under an optional nested model (e.g. ``compile.llm.rlm.enable``)
        # are only relevant when the user actually wrote through the
        # parent. If neither the parent nor the leaf is in the user's TOML
        # we silently skip them — we are not going to manufacture rows for
        # ``ok`` (= default = False) under a parent the user did not touch.
        if not in_user_toml and not _parent_key_present(raw, key):
            continue

        if key in validation_error_keys:
            entries.append(
                ConfigCheckEntry(
                    key=key,
                    status=STATUS_ERROR,
                    current=user_value,
                    default=default_value,
                    set_in_toml=in_user_toml,
                    message=validation_error_keys[key],
                ),
            )
            continue

        if in_user_toml and not _values_equal(user_value, default_value):
            entries.append(
                ConfigCheckEntry(
                    key=key,
                    status=STATUS_MODIFIED,
                    current=user_value,
                    default=default_value,
                    set_in_toml=True,
                ),
            )
            continue

        entries.append(
            ConfigCheckEntry(
                key=key,
                status=STATUS_OK,
                current=user_value,
                default=default_value,
                set_in_toml=in_user_toml,
            ),
        )

    # Add error rows for unknown keys the user wrote.
    for key in unknown_keys:
        entries.append(
            ConfigCheckEntry(
                key=key,
                status=STATUS_ERROR,
                current=get_dot(raw, key),
                default=None,
                set_in_toml=True,
                message="unknown config key (not in schema)",
            ),
        )

    # Add deprecation rows for keys removed from the schema (older vaults).
    for key in sorted(removed_keys):
        entries.append(
            ConfigCheckEntry(
                key=key,
                status=STATUS_REMOVED,
                current=get_dot(raw, key),
                default=None,
                set_in_toml=True,
                message="key removed from schema; value is ignored at runtime — please delete it from your config",
            ),
        )

    return ConfigCheckReport(
        path=path,
        entries=tuple(entries),
        files_scanned=files_scanned,
    )


def _removed_leaf_keys() -> set[str]:
    """Flatten ``REMOVED_KEYS`` (section -> keys) into full dotted leaf keys."""
    out: set[str] = set()
    for section, keys in REMOVED_KEYS.items():
        out.update(f"{section}.{k}" for k in keys)
    return out


# ── Internals ───────────────────────────────────────────────────────────────


def _schema_leaf_keys() -> set[str]:
    """Walk the live :class:`KnoxConfig` model and return every leaf dotted key.

    Recurses into nested :class:`BaseModel` types, and only stops at fields
    whose annotation is not itself a model. This catches nested-model fields
    that the DEFAULTS dict may show as ``None`` because their parent is
    optional (e.g. ``LLMConfig.rlm: RlmConfig | None`` → DEFAULTS has
    ``None`` but the schema still exposes ``compile.llm.rlm.enable``).
    """

    def _walk(cls: type[BaseModel], prefix: str) -> set[str]:
        out: set[str] = set()
        for name, info in cls.model_fields.items():
            dotted = f"{prefix}.{name}" if prefix else name
            nested = _nested_model_type(info, name)
            if nested is not None:
                out.update(_walk(nested, dotted))
            else:
                out.add(dotted)
        return out

    return _walk(KnoxConfig, "")


def _schema_defaults() -> dict[str, Any]:
    """Walk the live :class:`KnoxConfig` model and return every leaf field's default.

    Mirrors :func:`_schema_leaf_keys` but instead of keys returns
    ``{key: pydantic FieldInfo.default}``. Used to compare user-set values
    against the authoritative schema default when the DEFAULTS dump holds
    ``None`` for the parent (e.g. optional nested model fields).

    Falls back to ``info.default_factory()`` (or an instantiated empty model
    for nested fields) when ``info.default`` is :data:`PydanticUndefined`.
    """

    def _walk(cls: type[BaseModel], prefix: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, info in cls.model_fields.items():
            dotted = f"{prefix}.{name}" if prefix else name
            nested = _nested_model_type(info, name)
            if nested is not None:
                out.update(_walk(nested, dotted))
            else:
                value = info.default
                if value is PydanticUndefined and info.default_factory is not None:
                    try:
                        value = info.default_factory()
                    except Exception:
                        value = PydanticUndefined
                out[dotted] = value
        return out

    return _walk(KnoxConfig, "")


def _nested_model_type(info: FieldInfo, name: str) -> type[BaseModel] | None:
    """Return the nested model class for *info* (field *name*), if any.

    Unwraps ``Optional[Model]``, ``Model | None``, and direct ``Model``
    annotations. Other unions and concrete scalars return ``None``.
    """
    ann = info.annotation
    if ann is None:
        return None
    # Resolve string annotations / forward refs by re-reading field metadata.
    if isinstance(ann, str):
        try:
            ann = typing.get_type_hints(KnoxConfig, include_extras=True).get(
                name,
                ann,
            )
        except Exception:  # pragma: no cover — defensive
            return None
    origin = typing.get_origin(ann)
    # Unwrap ``Optional[Model]`` (``Union``), ``Model | None`` (PEP 604), or
    # direct ``Model``. ``X | None`` uses ``types.UnionType`` on Python 3.10+.
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        for arg in args:
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
        return None
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return ann
    return None


def _overlay_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` on top of ``base`` (last-wins per leaf)."""
    result: dict[str, Any] = {}
    for key, base_val in base.items():
        if key in overlay and isinstance(base_val, dict) and isinstance(overlay[key], dict):
            result[key] = _overlay_merge(base_val, overlay[key])
        else:
            result[key] = base_val
    for key, overlay_val in overlay.items():
        if key not in base or not (isinstance(base[key], dict) and isinstance(overlay_val, dict)):
            result[key] = overlay_val
    return result


def _parent_key_present(raw: dict[str, Any], key: str) -> bool:
    """Return ``True`` if every section path from the root to *key*'s
    immediate parent is present in *raw*.

    For ``compile.llm.rlm.enable`` the chain is
    ``["compile", "compile.llm", "compile.llm.rlm"]`` — the parent of
    the leaf is ``compile.llm.rlm``. We only surface nested-model leaves
    when the user actually opened those sections in the TOML.
    """
    parts = key.split(".")
    # All parent prefixes (length 1..len-1 inclusive); we want every one present.
    for i in range(1, len(parts)):
        section = parts[:i]
        cur: Any = raw
        present = True
        for part in section:
            if not isinstance(cur, dict) or part not in cur:
                present = False
                break
            cur = cur[part]
        if not present:
            return False
    return True


def _safe_dump_defaults() -> dict[str, Any]:
    """Make a JSON-safe copy of DEFAULTS suitable for ``model_validate``.

    ``DEFAULTS`` comes from :func:`KnoxConfig().model_dump()` but recursive
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


def _values_equal(a: Any, b: Any) -> bool:
    """Structural equality, agnostic to list ordering.

    Lists are sorted first so reordering doesn't change the diff result;
    nested dicts are JSON-serialized with sorted keys to keep order
    deterministic.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, bool) != isinstance(b, bool):
        return False

    def _walk(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _walk(inner) for k, inner in sorted(v.items())}
        if isinstance(v, list):
            return sorted(_walk(x) for x in v)
        if isinstance(v, Path):
            return str(v)
        return v

    try:
        return bool(
            json.dumps(_walk(a), sort_keys=True, default=str) == json.dumps(_walk(b), sort_keys=True, default=str),
        )
    except Exception:
        return a == b
