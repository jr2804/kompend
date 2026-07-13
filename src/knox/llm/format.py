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
from typing import Any

import ison_parser as _ison
import json_repair
import toon_format as _toon

from knox.core.config._configs import LlmDataFormat

# ── JSON salvage (re-exported; callers import salvage_json from here) ────


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

    if schema is not None:
        try:
            parsed = json_repair.loads(content, schema=schema, schema_repair_mode="salvage")
            if parsed != "":
                return parsed
        except (json.JSONDecodeError, ValueError, RecursionError):
            pass

    # try repairing without schema; this is the fallback path for any error above
    try:
        parsed = json_repair.loads(content)
    except (json.JSONDecodeError, RecursionError):
        return None
    # json_repair returns "" for prose-only input; normalize to None so
    # callers' `is None` checks behave as before.
    return None if parsed == "" else parsed


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
        ``"json"`` (default), ``"toon"``, or ``"ison"``.
    """
    if fmt == LlmDataFormat.JSON:
        return json.dumps(data, ensure_ascii=False)
    elif fmt == LlmDataFormat.TOON:
        return _toon.encode(data)
    elif fmt == LlmDataFormat.ISON:
        doc = _ison.from_dict(data if isinstance(data, dict) else {"data": data})
        return _ison.dumps(doc)
    else:
        raise ValueError(f"Unknown data_format: {fmt!r}. Valid: {[f.value for f in LlmDataFormat]}")
