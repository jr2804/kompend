"""LLM communication format utilities.

Supports JSON (default), TOON, and ISON as compact alternatives for LLM
prompt input context (serialize structured data before injection into
templates). TOON and ISON reduce prompt token count by 30–70 %, leaving
more context window budget for model output and reducing truncation-related
malformed responses with small local models.

LLM *output* always uses JSON — the ``response_format: json_object`` API
constraint is kept so providers can enforce valid JSON at the protocol
level. TOON and ISON are not yet widely understood by models, and attaching
a format specification to every call would be counterproductive.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import ison_parser as _ison
import json_repair
import toon_format as _toon
from pydantic import BaseModel

from knox.core.config._configs import LlmDataFormat
from knox.types import JSON, JSONDict, TypeType

# ── Reasoning strip ─────────────────────────────────────────────────────


_REASONING_TAGS: frozenset[str] = frozenset(
    {
        "think",
        "thinking",
        "reason",
        "reasoning",
        "thought",
        "thoughts",
        "reflection",
        "analysis",
        "scratchpad",
    }
)


def _build_reasoning_strip_pattern() -> re.Pattern[str]:
    """Compile one regex that strips any known reasoning tag."""
    alt = "|".join(re.escape(t) for t in sorted(_REASONING_TAGS))
    return re.compile(rf"<(?:{alt})>.*?</(?:{alt})>|<(?:{alt})>.*", re.DOTALL | re.IGNORECASE)


_REASONING_RE: re.Pattern[str] = _build_reasoning_strip_pattern()


def salvage_json(
    content: str,
    schema: type[BaseModel] | dict[str, object] | None = None,
    *,
    unwrap_list: bool = False,
) -> JSON | None:
    """Best-effort parse of JSON that small LLMs may truncate or malform.

    **json_repair is the single source of truth** for parsing and schema-guided
    repair.  This function is a thin wrapper that adds three small post-fixes
    for gaps json_repair doesn't cover:

    1. **Double-encoded JSON unwrapping** — models sometimes wrap JSON in
       extra quotes (``'"{"key": 1}"'``).  ``json_repair`` doesn't detect this.
    2. **List-of-dicts → object coercion** — models sometimes return
       ``[{"a": 1}, {"b": 2}]`` instead of ``{"a": 1, "b": 2}``.
       ``json_repair`` schema mode raises ``ValueError`` on this.
    3. **Empty-string filtering** — ``json_repair`` schema mode converts
       ``"field": ""`` to ``"field": [""]`` for array fields.  The empty
       string is stripped post-parse for data quality.

    When *schema* is given (a pydantic v2 model or JSON Schema dict), the
    ``salvage`` schema-repair mode runs first: it completes truncated JSON,
    fills missing fields, filters garbled items, and coerces types toward
    the schema.  That mode raises ``ValueError`` on wrong field names, so on
    any error it falls back to plain schema-less repair — callers never see
    a raised error.

    When *unwrap_list* is ``True`` (default) and the result is a single-element
    list containing a dict, the dict is returned instead. Small models
    frequently wrap expected JSON objects in an array: ``[{"score": 5}]``
    instead of ``{"score": 5}``. Pass ``unwrap_list=False`` when the caller
    legitimately expects a list.

    Pathological deeply-nested input raises ``RecursionError`` from BOTH
    stdlib and ``json_repair``'s recursive parsers, so it is caught here;
    total parse failure likewise degrades to ``None``.

    Returns the parsed value or ``None`` if nothing could be salvaged.
    """
    if not content or not content.strip():
        return None

    content = content.strip()

    # Post-fix 1: Detect double-encoded JSON — the model wrapped a JSON
    # object in extra quotes, producing a JSON string literal like
    # '{"summary":"..."}'.  json_repair doesn't detect this.
    if content.startswith('"') and content.endswith('"') and len(content) > 2 and ("{" in content or "[" in content):
        try:
            inner = json.loads(content)  # Parse the outer string literal
            if isinstance(inner, str) and inner.strip().startswith(("{", "[")):
                content = inner.strip()  # Use the unwrapped JSON
        except (json.JSONDecodeError, ValueError):
            pass  # Not a string literal — proceed with original content

    # Primary path: json_repair with schema-guided repair.
    if schema is not None:
        try:
            parsed = json_repair.loads(content, schema=schema, schema_repair_mode="salvage")
            if parsed != "":
                return _coerce_nulls_for_schema(_strip_empty_strings(parsed), schema)
        except (json.JSONDecodeError, ValueError, RecursionError):
            logging.getLogger(__name__).debug("salvage_json schema-mode failed", exc_info=True)

    # Fallback: schema-less repair.
    try:
        parsed = json_repair.loads(content)
    except (json.JSONDecodeError, RecursionError):
        return None
    # json_repair returns "" for prose-only input; normalize to None so
    # callers' `is None` checks behave as before.
    if parsed == "":
        return None

    # Post-fix 2: Coerce list-of-dicts to dict when schema expects an object.
    # Small models sometimes return [{"summary": "..."}, {"entities": []}]
    # instead of {"summary": "...", "entities": []}.  json_repair schema mode
    # raises ValueError on this; the schemaless fallback returns the raw list.
    if schema is not None and isinstance(parsed, list):
        coerced = _coerce_list_to_object(parsed, schema)
        if coerced is not None:
            return _coerce_nulls_for_schema(_strip_empty_strings(coerced), schema)

    # Unwrap single-element list to dict when caller expects an object.
    # Small models frequently wrap JSON objects in an array:
    # [{"score": 5, "feedback": "..."}] instead of {"score": 5, ...}.
    if unwrap_list and isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        return _coerce_nulls_for_schema(_strip_empty_strings(parsed[0]), schema) if schema is not None else _strip_empty_strings(parsed[0])

    return _coerce_nulls_for_schema(_strip_empty_strings(parsed), schema) if schema is not None else _strip_empty_strings(parsed)


# ── JSON salvage (json_repair is the single source of truth) ─────────────


def _strip_empty_strings(data: JSON) -> JSON:
    """Recursively remove empty-string items from lists in parsed JSON.

    ``json_repair`` schema mode converts ``"related": ""`` to
    ``"related": [""]`` (an array with one empty string).  While pydantic
    accepts this, empty strings in lists are never useful LLM output.
    This one-pass cleanup improves data quality without adding a full
    custom normalizer on top of json_repair.
    """
    if isinstance(data, dict):
        return {k: _strip_empty_strings(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_strip_empty_strings(item) for item in data if item != ""]
    return data


def _coerce_nulls_for_schema(data: JSON, schema: type[BaseModel] | dict[str, object]) -> JSON:
    """Replace ``None`` with type-appropriate defaults for known schema fields.

    ``json_repair`` schema mode raises ``ValueError`` when it encounters
    ``null`` for a ``string`` field (e.g. ``"slug": null``), so
    :func:`salvage_json` falls back to schema-less repair — which preserves
    the ``null``, causing pydantic to reject the entire item and trigger a
    model retry.  Small models frequently emit ``null`` where a string or
    list is expected, and retrying produces worse output.

    This function walks the parsed data alongside the pydantic schema and
    coerces ``None`` to the field's default: ``""`` for ``str``, ``[]`` for
    ``list``.  It only touches keys that exist in the schema — unknown keys
    are passed through unchanged.  This is the minimal post-fix for the one
    gap json_repair leaves open.
    """
    if not hasattr(schema, "model_fields"):
        return data

    if isinstance(data, dict):
        result = dict(data)
        for key, value in data.items():
            field_info = schema.model_fields.get(key)
            if field_info is None:
                continue
            annotation = field_info.annotation
            if value is None:
                result[key] = _default_for_annotation(annotation)
            else:
                # Recurse into nested model fields.
                inner = _inner_model_type(annotation)
                if inner is not None and isinstance(value, dict):
                    result[key] = _coerce_nulls_for_schema(value, inner)
                # Recurse into list-of-model fields.
                item_model = _list_inner_model_type(annotation)
                if item_model is not None and isinstance(value, list):
                    result[key] = [_coerce_nulls_for_schema(item, item_model) if isinstance(item, dict) else item for item in value]
        return result

    return data


def _default_for_annotation(annotation: Any) -> JSON:
    """Return the default value for a field annotation (``""`` or ``[]``)."""
    origin = getattr(annotation, "__origin__", None)
    if origin is list or str(annotation).startswith("list["):
        return []
    return ""  # str and anything else


def _list_inner_model_type(annotation: Any) -> TypeType:
    """If *annotation* is ``list[Model]``, return the Model type; else None."""
    args = getattr(annotation, "__args__", None)
    if args and len(args) == 1:
        return _inner_model_type(args[0])
    return None


def _inner_model_type(annotation: Any) -> TypeType:
    """If *annotation* is a pydantic BaseModel subclass, return it; else None."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _coerce_list_to_object(items: list[JSON], schema: type[BaseModel] | dict[str, object]) -> JSONDict | None:
    """Merge a list of single-key dicts into one dict matching the schema.

    Small models sometimes split a single object across multiple items:
    ``[{"summary": "..."}, {"entities": []}, {"concepts": [...]}]``.
    When the schema expects a Pydantic model (object), merge these into
    one dict. Returns ``None`` if the list can't be coerced to the schema.
    """
    if not items:
        return None

    # Get expected field names from the schema.
    expected_fields: set[str] | None = None
    if hasattr(schema, "model_fields"):
        expected_fields = set(schema.model_fields.keys())
    elif isinstance(schema, dict):
        props = schema.get("properties", {})
        expected_fields = set(props.keys()) if props else None

    if expected_fields is None:
        return None  # Can't determine schema shape

    merged: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict):
            return None  # Non-dict item — can't merge
        for key, val in item.items():
            if key in expected_fields:
                # Don't overwrite a non-empty value with an empty one.
                if key not in merged or (not merged[key] and val):
                    merged[key] = val
            else:
                return None  # Unknown field — this isn't a split-object pattern

    # Must have at least one expected field.
    if not any(k in expected_fields for k in merged):
        return None
    return merged


# ── Format encode / decode ────────────────────────────────────────────────


def encode_for_llm(data: JSON, fmt: LlmDataFormat) -> str:
    """Serialize *data* to the LLM communication format *fmt*.

    Use this to serialize structured context data before injecting into
    prompt templates. Compact formats reduce input token count, leaving
    more context window available for model output.

    Parameters
    ----------
    data:
        Python dict or list to serialize.
    fmt:
        :class:`LlmDataFormat` value or equivalent string:
        ``"json"`` (default), ``"toon"``, ``"ison"``, or ``"text"``.
        For ``"text"``/``"markdown"`` the data is returned as-is
        (no-op) since it is already human-readable prose.
    """
    # MARKDOWN and TEXT share value "text" — compare by value.
    if fmt.value == "text":
        return str(data) if not isinstance(data, str) else data
    if fmt == LlmDataFormat.JSON:
        return json.dumps(data, ensure_ascii=False)
    elif fmt == LlmDataFormat.TOON:
        return _toon.encode(data)
    elif fmt == LlmDataFormat.ISON:
        doc = _ison.from_dict(data if isinstance(data, dict) else {"data": data})
        return _ison.dumps(doc)
    else:
        raise ValueError(f"Unknown data_format: {fmt!r}. Valid: {[f.value for f in LlmDataFormat]}")


def strip_reasoning(content: str) -> str:
    """Remove inline reasoning/thinking blocks from model *content*."""
    if not content:
        return content
    stripped = _REASONING_RE.sub("", content)
    return stripped.strip() if stripped != content else content.strip()
