"""Diff a user-supplied config TOML against the built-in :data:`DEFAULTS`.

Used by the ``kompend config check`` CLI command. For each provided file:

- Walks every leaf key in the live :class:`KompendConfig` schema (so nested
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

Fields whose keys the *user* names are namespaces, not typos: ``providers``,
``compile.skills`` and ``providers.<name>.models`` hold models keyed by name,
and ``categories``/``frontmatter``/``hooks`` are free-form tables. A key under
one of those is matched segment-wise against the schema instead of compared to a
fixed list, so ``providers.ollama.base_url`` is a normal row while
``providers.ollama.base_uri`` is still an error.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import functools
import json
import re
import types
import typing
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from kompend.core.config._configs import KompendConfig
from kompend.core.config.engine._loading import load_toml
from kompend.core.config.engine._schema import DEFAULTS, DEPRECATED_ALIASES, REMOVED_KEYS
from kompend.core.utils.dicts import flat_keys, get_dot, resolve_dotted
from kompend.logging import get_logger
from kompend.types import JSONDict, JSONList, JSONValue

# ── Status kind ─────────────────────────────────────────────────────────────

STATUS_OK = "ok"
STATUS_MODIFIED = "modified"
STATUS_ERROR = "error"
STATUS_REMOVED = "removed"  # key present in TOML but removed from schema
STATUS_DEPRECATED = "deprecated"  # renamed key, still migrated at resolve time

#: Schema segment standing for a name the *user* chooses (provider, model,
#: skill). Matching allows one ``*`` to span several dotted segments, because a
#: provider or model name may itself contain dots
#: (``[providers.ollama.models."hf.co/qwen3.6"]``).
_DYNAMIC = "*"


_KIND_LEAF = "leaf"
_KIND_OPEN = "open"

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class _SchemaLeaf:
    """One node of the config schema walk (:func:`_schema_walk`)."""

    kind: str
    default: Any = None


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

    def to_dict(self) -> JSONDict:
        out: JSONDict = {
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
    def deprecated_count(self) -> int:
        return sum(1 for e in self.entries if e.status == STATUS_DEPRECATED)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def to_dict(self) -> JSONDict:
        entries: JSONList = [e.to_dict() for e in self.entries]
        out: JSONDict = {
            "ok": self.ok_count,
            "modified": self.modified_count,
            "error": self.error_count,
            "removed": self.removed_count,
            "entries": entries,
        }
        if self.path is not None:
            out["path"] = str(self.path)
        if self.files_scanned:
            out["files"] = [str(p) for p in self.files_scanned]
        return out


# ── Public API ──────────────────────────────────────────────────────────────


def check_config_file(path: Path) -> ConfigCheckReport:
    """Check one config file against the live :class:`KompendConfig` schema.

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

    Useful for ``kompend config check ./base.toml ./local.toml`` — the local
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
        if _DYNAMIC not in k
    )


def _check_loaded(
    *,
    raw: dict[str, Any],
    path: Path | None,
    files_scanned: tuple[Path, ...],
) -> ConfigCheckReport:
    """Diff ``raw`` against schema defaults and validate via :class:`KompendConfig`."""
    all_schema_keys = _schema_leaf_keys()
    # Keys with a ``*`` segment describe a user-named namespace; they are matched
    # segment-wise, never compared for equality.
    schema_patterns = tuple(sorted(k for k in all_schema_keys if _DYNAMIC in k))
    schema_keys = {k for k in all_schema_keys if _DYNAMIC not in k}
    schema_default_values = _schema_defaults()
    user_keys = set(flat_keys(raw))
    removed_leaf_keys = _removed_leaf_keys()

    def _known_schema_key(key: str) -> bool:
        """True when *key* is a schema leaf, in a namespace, or under a table."""
        if key in schema_keys:
            return True
        for pattern in schema_patterns:
            if _matches_schema_pattern(key, pattern):
                return True
        # A free-form table's keys are the user's data (``categories.foo``).
        return any(key.startswith(f"{leaf}.") for leaf in _open_namespace_leaves())

    def _removed_ancestor(key: str) -> str | None:
        """Return the retired key that *key* equals or sits under, if any.

        Table-valued retired keys (e.g. ``compile.llm.role_models``) flatten to
        nested leaves like ``compile.llm.role_models.<role>``, so matching must be
        prefix-aware, not exact.
        """
        for removed in removed_leaf_keys:
            if key == removed or key.startswith(f"{removed}."):
                return removed
        return None

    unknown_keys = sorted(k for k in user_keys if not _known_schema_key(k) and _removed_ancestor(k) is None and k not in DEPRECATED_ALIASES)
    removed_keys = sorted({a for k in user_keys if k not in schema_keys and (a := _removed_ancestor(k)) is not None})
    deprecated_keys = sorted(k for k in user_keys if k in DEPRECATED_ALIASES)

    # Validation pass via pydantic to surface type/value errors.
    merged: dict[str, Any] = _overlay_merge(_safe_dump_defaults(), raw)
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

    # Rows for keys the user set inside a user-named namespace. They are not in
    # ``schema_keys`` (that set is pattern-free), so without this pass a valid
    # ``providers.ollama.base_url`` would silently vanish from the report.
    for key in sorted(k for k in user_keys if k not in schema_keys and _known_schema_key(k)):
        found, default_value = _schema_lookup(key, schema_default_values, schema_patterns)
        if not found:
            continue
        found_value, user_value = resolve_dotted(raw, key)
        user_value = user_value if found_value else None
        status = STATUS_OK
        message = None
        if key in validation_error_keys:
            status, message = STATUS_ERROR, validation_error_keys[key]
        elif not _values_equal(user_value, default_value):
            status = STATUS_MODIFIED
        entries.append(
            ConfigCheckEntry(
                key=key,
                status=status,
                current=user_value,
                default=default_value,
                set_in_toml=True,
                message=message,
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

    # Renamed keys: still migrated at resolve time, so report the migration
    # rather than an unknown-key error.
    entries.extend(
        ConfigCheckEntry(
            key=key,
            status=STATUS_DEPRECATED,
            current=get_dot(raw, key),
            default=get_dot(raw, DEPRECATED_ALIASES[key]),
            set_in_toml=True,
            message=f"deprecated — use '{DEPRECATED_ALIASES[key]}' instead; the value is migrated automatically",
        )
        for key in deprecated_keys
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
    """Every key the config schema accepts, as dotted paths.

    A ``*`` segment marks a **user-named namespace** (a ``dict[str, Model]``
    field): the user picks the segment, the schema validates what follows it.
    Callers that need a literal key filter those out — a pattern is not a key.
    """
    return set(_schema_walk())


def _schema_defaults() -> dict[str, Any]:
    """The authoritative default for every :func:`_schema_leaf_keys` entry.

    Read from the live model rather than the :data:`DEFAULTS` dump, which holds
    ``None`` for optional parents (e.g. ``LLMConfig.rlm``) and would hide the
    real default of a leaf underneath.
    """
    return {key: leaf.default for key, leaf in _schema_walk().items()}


def _open_namespace_leaves() -> set[str]:
    """Leaves that are free-form tables — their keys are the user's data.

    ``categories`` and ``frontmatter`` are ``dict[str, str]``/``dict[str, Any]``,
    so ``categories.foo`` is a value the user chose, not a typo to report.
    """
    return {key for key, leaf in _schema_walk().items() if leaf.kind == _KIND_OPEN}


@functools.cache
def _schema_walk() -> dict[str, _SchemaLeaf]:
    """Classify every leaf of :class:`KompendConfig` in one traversal.

    One walk feeds all three views (keys, defaults, open namespaces) so they
    cannot drift apart — the failure mode this replaced: the keys walker and the
    defaults walker each had to learn about dynamic namespaces separately. The
    result is cached: the schema is static for the life of the process, and the
    dynamic-key matcher consults the open-namespace view per user key.

    Three kinds, because a key is only "unknown" when nothing can accept it:

    * ``leaf``    — ordinary field, matched by exact key
    * ``*`` key   — leaf under a user-named namespace with a schema
      (``providers.<any>.base_url``), matched segment-wise
    * ``open``    — free-form table, so any key under it is user data
    """

    def _walk(cls: type[BaseModel], prefix: str) -> dict[str, _SchemaLeaf]:
        out: dict[str, _SchemaLeaf] = {}
        for name, info in cls.model_fields.items():
            dotted = f"{prefix}.{name}" if prefix else name
            nested = _nested_model_type(info, name)
            if nested is not None:
                out.update(_walk(nested, dotted))
                continue
            value_type = _mapping_value_type(info, name)
            if value_type is not None:
                if isinstance(value_type, type) and issubclass(value_type, BaseModel):
                    out.update(_walk(value_type, f"{dotted}.{_DYNAMIC}"))
                else:
                    out[dotted] = _SchemaLeaf(kind=_KIND_OPEN)
                continue
            out[dotted] = _SchemaLeaf(kind=_KIND_LEAF, default=_field_default(info))
        return out

    return _walk(KompendConfig, "")


def _field_default(info: FieldInfo) -> Any:
    """The field's default, materializing a factory that needs no arguments.

    pydantic types ``default_factory`` as either a zero-argument callable or one
    taking validated data; only the former can be evaluated here, so the call is
    narrowed explicitly instead of guessing at the union.
    """
    value = info.default
    factory = cast("collections.abc.Callable[[], Any]", info.default_factory) if info.default_factory is not None else None
    if value is PydanticUndefined and factory is not None:
        try:
            value = factory()
        except Exception:
            value = PydanticUndefined
    return value


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
            ann = typing.get_type_hints(KompendConfig, include_extras=True).get(
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


def _mapping_value_type(info: FieldInfo, name: str) -> Any | None:
    """Return the value type of a ``dict[str, X]`` / ``Mapping[str, X]`` field.

    ``None`` means the field is not a string-keyed mapping, so it is an ordinary
    leaf. A returned type tells the caller which namespace kind it is: a
    :class:`BaseModel` subclass is a user-named namespace *with* a schema of its
    own (validated), anything else is a free-form table (the keys are data).
    """
    ann = _resolved_annotation(info, name)
    if ann is None or typing.get_origin(ann) not in (dict, collections.abc.Mapping):
        return None
    args = typing.get_args(ann)
    if len(args) != 2 or args[0] is not str:
        return None
    return args[1]


def _resolved_annotation(info: FieldInfo, name: str) -> Any | None:
    """The field's annotation, resolving forward references where possible."""
    ann = info.annotation
    if isinstance(ann, str):
        try:
            ann = typing.get_type_hints(KompendConfig, include_extras=True).get(name, ann)
        except Exception:  # pragma: no cover — defensive
            return None
    return ann


def _schema_lookup(key: str, table: dict[str, Any], patterns: tuple[str, ...]) -> tuple[bool, Any]:
    """Resolve *key* through *table*, falling back to dynamic patterns.

    Returns ``(found, value)``. The exact table wins; a pattern provides the
    schema default for a user-named namespace (``providers.*.base_url`` → the
    field default for any provider).
    """
    if key in table:
        return True, table[key]
    for pattern in patterns:
        if _matches_schema_pattern(key, pattern):
            return True, table[pattern]
    return False, None


def _matches_schema_pattern(key: str, pattern: str) -> bool:
    """True when *key* is a valid instance of the dynamic schema *pattern*."""
    return re.match(_pattern_to_regex(pattern), key) is not None


def _pattern_to_regex(pattern: str) -> str:
    """Compile a schema key into a regex, ``*`` spanning one or more segments."""
    parts = [".+" if part == _DYNAMIC else re.escape(part) for part in pattern.split(".")]
    return "^" + r"\.".join(parts) + "$"


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
