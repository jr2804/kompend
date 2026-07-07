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
import toon_format as _toon

from knox.core.config._configs import LlmDataFormat

# ── JSON salvage (re-exported; callers import salvage_json from here) ────


def salvage_json(content: str) -> Any | None:  # noqa: PLR0911
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

    try:
        return json.loads(content)
    except (json.JSONDecodeError, RecursionError):
        # RecursionError: pathological deeply-nested content blows stdlib
        # json's recursive-descent parser — treat as unparseable, fall through.
        pass

    # Small models often emit raw control chars (newlines, tabs) inside JSON
    # string values, which is invalid JSON. Sanitize those (only inside string
    # literals — structural whitespace is preserved) and retry before the
    # heavier truncation-recovery strategies.
    sanitized = _sanitize_string_control_chars(content)
    if sanitized != content:
        try:
            return json.loads(sanitized)
        except (json.JSONDecodeError, RecursionError):
            content = sanitized

    # Leading prose before the JSON (e.g. "Sure, here is the JSON: {...}"):
    # parse from the first object/array token.
    first_obj = _find_first_json_token(content)
    if first_obj > 0:
        tail = content[first_obj:]
        salvaged = salvage_json(tail)
        if salvaged is not None:
            return salvaged

    if content.startswith("```"):
        inner = content.split("\n", 1)[-1] if "\n" in content else content[3:]
        inner = inner.strip()
        if inner.endswith("```"):
            inner = inner[:-3].strip()
        try:
            return json.loads(inner)
        except (json.JSONDecodeError, RecursionError):
            content = inner

    closed = _close_truncated_json(content)
    if closed is not None:
        try:
            return json.loads(closed)
        except (json.JSONDecodeError, RecursionError):
            pass

    try:
        decoder = json.JSONDecoder()
        obj, _end = decoder.raw_decode(content)
        return obj
    except (json.JSONDecodeError, RecursionError):
        pass

    return None


def _find_first_json_token(s: str) -> int:
    """Index of the first ``{`` or ``[`` in *s*, or ``-1`` if none.

    Used to skip leading prose (``"Here is the JSON: {...}"``). Returns 0 when
    *s* already starts with a JSON token so callers can guard against no-op
    recursion.
    """
    brace = s.find("{")
    bracket = s.find("[")
    candidates = [i for i in (brace, bracket) if i >= 0]
    return min(candidates) if candidates else -1


def _sanitize_string_control_chars(s: str) -> str:
    r"""Escape raw control chars (0x00–0x1F) that appear *inside* JSON strings.

    Structural whitespace between tokens is preserved. Small models often
    emit literal newlines/tabs inside string values, which is invalid JSON;
    this rewrites them to their escaped forms (``\n``, ``\t``, …) so
    ``json.loads`` accepts them.
    """
    _ESCAPE = {0x08: "\\b", 0x09: "\\t", 0x0A: "\\n", 0x0C: "\\f", 0x0D: "\\r"}
    out: list[str] = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string:
            o = ord(ch)
            if o < 0x20:
                out.append(_ESCAPE.get(o, f"\\u{o:04x}"))
                continue
        out.append(ch)
    return "".join(out)


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
    if escape:
        # Truncated mid-escape (e.g. trailing backslash): drop the incomplete
        # escape so the appended close-quote isn't turned into an escaped quote.
        s = s[:-1]
    if in_string:
        s += '"'
    return s + "".join(reversed(stack))


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
    if fmt == LlmDataFormat.TOON:
        return _toon.encode(data)
    if fmt == LlmDataFormat.ISON:
        doc = _ison.from_dict(data if isinstance(data, dict) else {"data": data})
        return _ison.dumps(doc)
    raise ValueError(f"Unknown data_format: {fmt!r}. Valid: {[f.value for f in LlmDataFormat]}")


def decode_from_llm(text: str, fmt: LlmDataFormat) -> Any | None:
    """Parse an LLM response in format *fmt* to a Python object.

    Returns the parsed object or ``None`` if parsing fails entirely.
    LLM output is always JSON in practice; this exists for completeness
    and potential future use.
    """
    if fmt == LlmDataFormat.JSON:
        return salvage_json(text)
    if fmt == LlmDataFormat.TOON:
        try:
            return _toon.decode(text)
        except Exception:
            try:
                return _toon.decode(text, {"strict": False})
            except Exception:
                return None
    if fmt == LlmDataFormat.ISON:
        try:
            return json.loads(_ison.loads(text).to_json())
        except Exception:
            return None
    raise ValueError(f"Unknown data_format: {fmt!r}. Valid: {[f.value for f in LlmDataFormat]}")


# ── System message helper ─────────────────────────────────────────────────


def structured_return_instruction(fmt: LlmDataFormat) -> str:
    """Return the format-appropriate instruction for system messages."""
    return f"Return only valid {str(fmt).upper()}."
