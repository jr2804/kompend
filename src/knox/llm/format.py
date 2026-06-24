"""LLM communication format utilities.

Supports JSON (default), TOON, and ISON as compact alternatives for both
LLM prompt input context and LLM response parsing.

Input benefit
-------------
Serializing structured context data (entity lists, concept batches) in
TOON or ISON reduces prompt token count by 30–70 %, leaving more context
window budget for the model's output. This directly addresses the most
common source of malformed/truncated responses with small local models.

Output benefit
--------------
Asking the model to respond in TOON or ISON (via a system-message hint)
uses simpler syntax — no nested braces, minimal punctuation — which reduces
generation errors. Trade-off: the API-level ``response_format: json_object``
guarantee is lost; reliability depends on model quality. Works best with
frontier models. For small/local models the input-only benefit is safer.

Optional packages (only required when the corresponding format is configured)
------------------------------------------------------------------------------
TOON: ``uv add toon_format``   (PyPI: toon_format >= 0.9.0)
ISON: ``uv add ison-py``       (PyPI: ison-py >= 1.0.0)

The try/except ImportError guards below are intentional Optional-feature
probes (see src/knox/AGENTS.md — Error handling). The packages are declared
under [project.optional-dependencies] in pyproject.toml. Importing at module
level (not inline) satisfies PLC0415; graceful fallback satisfies the
optional-dep antipattern rules.
"""

from __future__ import annotations

import json
from typing import Any

# ── Optional dependency probes ───────────────────────────────────────────
# Module-level try/except so imports fire once, not on every call.

try:
    import toon_format as _toon  # type: ignore[import-untyped]

    _HAS_TOON = True
except ImportError:
    _toon = None  # type: ignore[assignment]
    _HAS_TOON = False

try:
    import ison_py as _ison  # type: ignore[import-untyped]

    _HAS_ISON = True
except ImportError:
    _ison = None  # type: ignore[assignment]
    _HAS_ISON = False


# ── JSON salvage (moved from client.py, re-exported there) ───────────────


def salvage_json(content: str) -> Any | None:
    """Best-effort parse of JSON that small LLMs may truncate or malform.

    Tries, in order:
    1. Direct ``json.loads``.
    2. Strip markdown code fences.
    3. Close unclosed ``{ }`` / ``[ ]`` (respecting string literals).
    4. ``raw_decode`` to salvage the first valid value from extra data.

    Returns the parsed value or ``None`` if nothing could be salvaged.
    """
    if not content or not content.strip():
        return None
    content = content.strip()

    # 1. Direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences
    if content.startswith("```"):
        inner = content.split("\n", 1)[-1] if "\n" in content else content[3:]
        inner = inner.strip()
        if inner.endswith("```"):
            inner = inner[:-3].strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            content = inner

    # 3. Close unclosed brackets/braces (truncated output)
    closed = _close_truncated_json(content)
    if closed is not None:
        try:
            return json.loads(closed)
        except json.JSONDecodeError:
            pass

    # 4. Salvage the first valid JSON value (handles extra data)
    try:
        decoder = json.JSONDecoder()
        obj, _end = decoder.raw_decode(content)
        return obj
    except json.JSONDecodeError:
        pass

    return None


def _close_truncated_json(s: str) -> str | None:
    """Close any unclosed ``{``/``[`` in *s*, respecting string literals.

    Returns the closed string, or ``None`` if brackets are already balanced.
    """
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    if not stack and not in_string:
        return None
    s = s.rstrip()
    if s.endswith(","):
        s = s[:-1]
    # Close an unterminated string literal before closing brackets.
    if in_string:
        s += '"'
    return s + "".join(reversed(stack))


# ── Format encode / decode ────────────────────────────────────────────────


def encode_for_llm(data: Any, fmt: str) -> str:
    """Serialize *data* to the LLM communication format *fmt*.

    Use this to serialize structured context data before injecting into
    prompt templates. Compact formats reduce input token count, leaving
    more context window available for model output.

    Parameters
    ----------
    data:
        Python dict or list to serialize.
    fmt:
        ``"json"`` (default), ``"toon"``, or ``"ison"``.
    """
    if fmt == "json":
        return json.dumps(data, ensure_ascii=False)
    if fmt == "toon":
        if not _HAS_TOON:
            raise ImportError("toon_format is required when data_format='toon'. Install with: uv add toon_format")
        return _toon.encode(data)
    if fmt == "ison":
        if not _HAS_ISON:
            raise ImportError("ison-py is required when data_format='ison'. Install with: uv add ison-py")
        doc = _ison.from_dict(data if isinstance(data, dict) else {"data": data})
        return _ison.dumps(doc)
    raise ValueError(f"Unknown data_format: {fmt!r}. Valid: 'json', 'toon', 'ison'")


def decode_from_llm(text: str, fmt: str) -> Any | None:
    """Parse an LLM response in format *fmt* to a Python object.

    Returns the parsed object or ``None`` if parsing fails entirely.
    """
    if fmt == "json":
        return salvage_json(text)
    if fmt == "toon":
        if not _HAS_TOON:
            raise ImportError("toon_format is required when data_format='toon'. Install with: uv add toon_format")
        try:
            return _toon.decode(text)
        except Exception:
            # Try non-strict mode for slightly malformed TOON
            try:
                return _toon.decode(text, {"strict": False})
            except Exception:
                return None
    if fmt == "ison":
        if not _HAS_ISON:
            raise ImportError("ison-py is required when data_format='ison'. Install with: uv add ison-py")
        try:
            doc = _ison.parse(text)
            return json.loads(doc.to_json())
        except Exception:
            return None
    raise ValueError(f"Unknown data_format: {fmt!r}. Valid: 'json', 'toon', 'ison'")


# ── System message helpers ────────────────────────────────────────────────


def format_system_hint(fmt: str) -> str:
    """Return a system-message addition instructing the model to use *fmt*.

    Returns an empty string for ``"json"`` (default, no hint needed).
    Appended to the system message when ``data_format != 'json'``.
    """
    if fmt == "toon":
        return (
            "\n\nIMPORTANT — use TOON format (Token-Oriented Object Notation) "
            "instead of JSON for all structured responses.\n"
            "TOON syntax quick reference:\n"
            "  Simple object:    key: value\n"
            "  Primitive array:  name[N]: val1,val2,val3\n"
            "  Tabular array:    name[N]{field1,field2}:\n"
            "                      row1val1,row1val2\n"
            "                      row2val1,row2val2\n"
            "Do NOT use JSON braces { }, brackets [ ], or colons in values "
            "unless the value itself contains them."
        )
    if fmt == "ison":
        return (
            "\n\nIMPORTANT — use ISON format (Interchange Simple Object Notation) "
            "instead of JSON for all structured responses.\n"
            "ISON syntax quick reference:\n"
            "  Table block:  table.name\n"
            "                field1 field2 field3\n"
            "                val1   val2   val3\n"
            "  Object block: object.name\n"
            "                key value\n"
            "                k1  v1\n"
            "Do NOT use JSON braces, brackets, or comma separators."
        )
    return ""  # "json": no hint needed


def inject_format_hint(messages: list[dict[str, Any]], hint: str) -> list[dict[str, Any]]:
    """Return a copy of *messages* with *hint* appended to the system message.

    If no system message exists, prepends one. Does not mutate the input list.
    """
    if not hint:
        return messages
    result = list(messages)
    if result and result[0].get("role") == "system":
        result[0] = dict(result[0])
        result[0]["content"] = result[0]["content"] + hint
    else:
        result.insert(0, {"role": "system", "content": hint.strip()})
    return result
