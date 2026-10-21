"""Tolerant handling of duplicated TOML keys.

``tomllib`` rejects any document in which a key is assigned twice
("Cannot overwrite a value"), which throws away an otherwise valid config
because of one stale leftover line. The scanner in this module rewrites
such a document by dropping **later** re-assignments and keeping the first
occurrence, so the config loads with a warning instead of failing outright.

It is only invoked after tomllib has already rejected the text; any
residual parse error after cleaning falls back to the original failure
path. Appending ``[[array]]`` elements is legal TOML and never treated
as a duplicate; an *adjacent* re-declaration of the same ``[table]``
header is merged into the first block, while non-adjacent re-opens are
left untouched (their key block cannot be relocated safely) and fall
back to the hard failure path.
"""

from __future__ import annotations

import re

_TRIPLE_DELIMS: tuple[str, ...] = ('"""', "'''")
_ARRAY_TABLE_RE = re.compile(r"^\s*\[\[(?P<name>[^\]]+)\]\]\s*(?:#.*)?$")
_TABLE_RE = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*(?:#.*)?$")
_KEY_VALUE_RE = re.compile(r"""^\s*(?P<key>(?:"[^"]*"|'[^']*'|[^=#[\]])+?)\s*=(?!=)(?P<value>.*)$""")


def strip_duplicate_keys(text: str) -> tuple[str, list[tuple[str, int]]]:
    """Drop later re-assignments of duplicate keys, keeping the first.

    Line-oriented scanner that tracks the active table, multi-line
    arrays/inline tables, and triple-quoted strings, so key-looking lines
    inside multi-line values are not mistaken for assignments. Returns the
    cleaned text plus ``(dotted.key, line number)`` for every dropped entry.
    """
    lines = text.splitlines()
    kept: list[str] = []
    dropped: list[tuple[str, int]] = []
    seen: set[tuple[str, ...]] = set()
    declared: set[tuple[str, ...]] = set()
    last_header: tuple[str, ...] | None = None
    table: tuple[str, ...] = ()
    in_triple: str | None = None
    depth = 0
    dropping = False

    for lineno, line in enumerate(lines, start=1):
        if in_triple is not None:
            if line.count(in_triple) % 2 == 1:
                after = line.rfind(in_triple) + len(in_triple)
                depth = max(0, depth + bracket_delta(line[after:]))
                in_triple = None
            if not dropping:
                kept.append(line)
            if in_triple is None and depth == 0:
                dropping = False
            continue
        if depth > 0:
            depth = max(0, depth + bracket_delta(line))
            if not dropping:
                kept.append(line)
            if depth == 0:
                dropping = False
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            kept.append(line)
            continue
        array_header = _ARRAY_TABLE_RE.match(stripped)
        header = array_header or _TABLE_RE.match(stripped)
        if header:
            identity = tuple(_split_dotted(header.group("name")))
            if array_header is None and identity in declared and identity == last_header:
                # Adjacent re-declaration of the same table — merge into the
                # first block instead of failing the whole document.
                dropped.append((".".join(_unquote(seg) for seg in identity) + " (header)", lineno))
                table = identity
                continue
            table = identity
            last_header = identity
            if array_header is None:
                declared.add(identity)
            kept.append(line)
            continue
        kv = _KEY_VALUE_RE.match(line)
        if kv is None:
            kept.append(line)
            continue

        identity = table + tuple(_split_dotted(kv.group("key").strip()))
        rest, opens_triple = _cut_triple_spans(kv.group("value"))
        depth = max(0, depth + bracket_delta(rest))
        if identity in seen:
            dropped.append((".".join(_unquote(seg) for seg in identity), lineno))
            dropping = True
        else:
            seen.add(identity)
            kept.append(line)
        if opens_triple is not None:
            in_triple = opens_triple
        if in_triple is None and depth == 0:
            dropping = False

    cleaned = "\n".join(kept)
    if text.endswith("\n"):
        cleaned += "\n"
    return cleaned, dropped


def _split_dotted(expr: str) -> list[str]:
    """Split a TOML dotted key or header expression on unquoted dots."""
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for char in expr:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            current.append(char)
        elif char == ".":
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return parts


def _unquote(part: str) -> str:
    """Strip one layer of matching quotes from a key segment."""
    if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'":
        return part[1:-1]
    return part


def _cut_triple_spans(text: str) -> tuple[str, str | None]:
    """Remove complete triple-quoted spans from *text*.

    Returns the remaining text and the delimiter that opens an unterminated
    (multi-line) span, or ``None`` when all spans are closed.
    """
    while True:
        positions = [(text.find(delim), delim) for delim in _TRIPLE_DELIMS]
        found = [(pos, delim) for pos, delim in positions if pos != -1]
        if not found:
            return text, None
        pos, delim = min(found)
        end = text.find(delim, pos + 3)
        if end == -1:
            return text[:pos], delim
        text = text[:pos] + text[end + 3 :]


def bracket_delta(text: str) -> int:
    """Net ``[]``/``{}`` delta in *text*, ignoring quoted spans and comments."""
    depth = 0
    quote: str | None = None
    i = 0
    while i < len(text):
        char = text[i]
        if quote is not None:
            if quote == '"' and char == "\\":
                i += 2
                continue
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#":
            break
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        i += 1
    return depth
