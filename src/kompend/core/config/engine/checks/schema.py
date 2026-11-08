"""Walk the live :class:`KompendConfig` schema into dotted keys and defaults."""

from __future__ import annotations

import collections.abc
import dataclasses
import functools
import re
import types
import typing
from typing import Any, cast

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from kompend.core.config.models.root import KompendConfig

#: Schema segment standing for a name the *user* chooses (provider, model,
#: skill). Matching allows one ``*`` to span several dotted segments, because a
#: provider or model name may itself contain dots
#: (``[providers.ollama.models."hf.co/qwen3.6"]``).
_DYNAMIC = "*"


_KIND_LEAF = "leaf"
_KIND_OPEN = "open"


@dataclasses.dataclass(frozen=True)
class _SchemaLeaf:
    """One node of the config schema walk (:func:`_schema_walk`)."""

    kind: str
    default: Any = None


def schema_leaf_keys() -> set[str]:
    """Every key the config schema accepts, as dotted paths.

    A ``*`` segment marks a **user-named namespace** (a ``dict[str, Model]``
    field): the user picks the segment, the schema validates what follows it.
    Callers that need a literal key filter those out — a pattern is not a key.
    """
    return set(_schema_walk())


def schema_defaults() -> dict[str, Any]:
    """The authoritative default for every :func:`schema_leaf_keys` entry.

    Read from the live model rather than the :data:`DEFAULTS` dump, which holds
    ``None`` for optional parents (e.g. ``LLMConfig.rlm``) and would hide the
    real default of a leaf underneath.
    """
    return {key: leaf.default for key, leaf in _schema_walk().items()}


def open_namespace_leaves() -> set[str]:
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


def schema_lookup(key: str, table: dict[str, Any], patterns: tuple[str, ...]) -> tuple[bool, Any]:
    """Resolve *key* through *table*, falling back to dynamic patterns.

    Returns ``(found, value)``. The exact table wins; a pattern provides the
    schema default for a user-named namespace (``providers.*.base_url`` → the
    field default for any provider).
    """
    if key in table:
        return True, table[key]
    for pattern in patterns:
        if matches_schema_pattern(key, pattern):
            return True, table[pattern]
    return False, None


def matches_schema_pattern(key: str, pattern: str) -> bool:
    """True when *key* is a valid instance of the dynamic schema *pattern*."""
    return re.match(_pattern_to_regex(pattern), key) is not None


def _pattern_to_regex(pattern: str) -> str:
    """Compile a schema key into a regex, ``*`` spanning one or more segments."""
    parts = [".+" if part == _DYNAMIC else re.escape(part) for part in pattern.split(".")]
    return "^" + r"\.".join(parts) + "$"
