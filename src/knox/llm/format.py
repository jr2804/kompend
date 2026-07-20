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

from knox.core.config._configs import LlmDataFormat

# ── JSON salvage (re-exported; callers import salvage_json from here) ────


def _coerce_list_to_object(items: list[Any], schema: Any) -> dict[str, Any] | None:
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


def salvage_json(content: str, schema: Any | None = None) -> Any | None:
    """Best-effort parse of JSON that small LLMs may truncate or malform.

    Delegates to ``json_repair`` (stdlib ``json.loads`` fast path, then a
    forgiving repair parser that closes truncated brackets, strips prose and
    markdown fences, and tolerates raw control chars inside strings).

    When *schema* is given (a pydantic v2 model or JSON Schema dict), the
    ``salvage`` schema-repair mode runs first: it completes truncated JSON and
    fills missing fields toward the schema. That mode raises ``ValueError`` on
    wrong field names (treated as a missing required key), so on any error it
    falls back to plain schema-less repair — callers never see a raised error.

    Pathological deeply-nested input raises ``RecursionError`` from BOTH
    stdlib and ``json_repair``'s recursive parsers, so it is caught here;
    total parse failure likewise degrades to ``None``.

    Returns the parsed value or ``None`` if nothing could be salvaged.
    """
    if not content or not content.strip():
        return None

    content = content.strip()

    # Detect double-encoded JSON: the model wrapped a JSON object in extra
    # quotes, producing a JSON string literal like '{"summary":"..."}'.
    # This happens when small models confuse "return JSON" with "return a
    # string containing JSON". Unwrap the outer string before parsing.
    if content.startswith('"') and content.endswith('"') and len(content) > 2 and ("{" in content or "[" in content):
        try:
            inner = json.loads(content)  # Parse the outer string literal
            if isinstance(inner, str) and inner.strip().startswith(("{", "[")):
                content = inner.strip()  # Use the unwrapped JSON
        except (json.JSONDecodeError, ValueError):
            pass  # Not a string literal — proceed with original content

    if schema is not None:
        try:
            parsed = json_repair.loads(content, schema=schema, schema_repair_mode="salvage")
            if parsed != "":
                return parsed
        except (json.JSONDecodeError, ValueError, RecursionError):
            logging.getLogger(__name__).debug("salvage_json schema-mode failed", exc_info=True)

    # try repairing without schema; this is the fallback path for any error above
    try:
        parsed = json_repair.loads(content)
    except (json.JSONDecodeError, RecursionError):
        return None
    # json_repair returns "" for prose-only input; normalize to None so
    # callers' `is None` checks behave as before.
    if parsed == "":
        return None

    # Coerce list-of-dicts to dict when schema expects an object.
    # Small models sometimes return [{"summary": "..."}, {"entities": []}]
    # instead of {"summary": "...", "entities": []}. Merge them.
    if schema is not None and isinstance(parsed, list):
        coerced = _coerce_list_to_object(parsed, schema)
        if coerced is not None:
            return coerced

    return parsed


# ── Format encode / decode ────────────────────────────────────────────────


def encode_for_llm(data: Any, fmt: LlmDataFormat) -> str:
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


def strip_reasoning(content: str) -> str:
    """Remove inline reasoning/thinking blocks from model *content*."""
    if not content:
        return content
    stripped = _REASONING_RE.sub("", content)
    return stripped.strip() if stripped != content else content.strip()
