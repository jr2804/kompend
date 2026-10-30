r"""Prose-pipe escaping for emitted markdown (generation contract).

A literal ``|`` in markdown prose can be misparsed as GFM table structure —
the exact input class that sent docling's markdown backend into an infinite
CPU spin (unescaped pipes in list-item paragraph continuations). Escaping is
meaning-preserving: ``\\|`` renders as ``|`` in every CommonMark renderer, so
an escaped body round-trips to the source text.

wenmode's renderer already escapes ``|`` in AST text nodes (everything built
through ``PageBuilder``). This module closes the gap for verbatim string
emission: raw bodies, LLM output, user notes, exported element text.

Protected regions are left untouched:

- fenced code blocks (```` ``` ```` / ```` ~~~ ````),
- genuine GFM tables (header row + matching delimiter row + pipe rows),
- inline code spans, inline/block math (``$…$``, ``$$…$$``),
- ``[[wikilink|alias]]`` spans,
- pipes already preceded by an odd number of backslashes.

:func:`escape_prose_pipes` verifies its own output via
:func:`assert_pipes_escaped` before returning — emission never continues with
an unverified body.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

__all__ = ["PipeEscapeError", "assert_pipes_escaped", "escape_prose_pipes"]


class PipeEscapeError(ValueError):
    """A markdown body contains an unescaped prose pipe."""


# Inline spans whose content must never gain a backslash: code spans (with
# matching tick runs), [[wikilink|alias]] syntax, and math delimiters.
_PROTECTED_SPAN_RE = re.compile(
    r"(`+)(.+?)\1"
    r"|\[\[[^\[\]\n]*\]\]"
    r"|\$\$.+?\$\$"
    r"|\$[^$\n]+?\$"
)

_FENCE_OPEN_RE = re.compile(r"^ {0,3}(?P<marker>`{3,}|~{3,})")
_DELIM_CELL = r":?-+:?"
# A GFM delimiter row: optional pipes around, cells of hyphens/colons.
_DELIM_ROW_RE = re.compile(rf"^\s*\|?\s*{_DELIM_CELL}\s*(?:\|\s*{_DELIM_CELL}\s*)*\|?\s*$")
# A pipe with its preceding backslash run (parity decides escaped or bare).
_PIPE_RUN_RE = re.compile(r"(\\*)\|")

_PROSE = "prose"
_FENCED = "fenced"
_TABLE = "table"


def _bare_pipe_count(line: str) -> int:
    """Count pipes that are *not* backslash-escaped (even backslash run)."""
    return sum(1 for m in _PIPE_RUN_RE.finditer(line) if len(m.group(1)) % 2 == 0)


def _cells(line: str) -> list[str]:
    """Split *line* on bare pipes into GFM cells (outer empty cells dropped)."""
    masked = _PIPE_RUN_RE.sub(
        lambda m: (m.group(1) + "\x00") if len(m.group(1)) % 2 == 0 else m.group(1),
        line,
    )
    cells = masked.split("\x00")
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return cells


def _is_table_start(header: str, delimiter: str) -> bool:
    """GFM table start: pipe header row + delimiter row with equal cell count."""
    if not _bare_pipe_count(header) or not _DELIM_ROW_RE.match(delimiter):
        return False
    return len(_cells(header)) == len(_cells(delimiter)) >= 1


def _fence_close_re(marker: str) -> re.Pattern[str]:
    return re.compile(rf"^ {{0,3}}{marker[0]}{{{len(marker)},}}\s*$")


def _classify(lines: list[str]) -> Iterator[tuple[int, str, str]]:
    """Yield ``(index, line, kind)`` with kind in prose/fenced/table.

    Fenced and table lines are protected verbatim; only prose lines may be
    rewritten. Shared by the escaper and the verifier so the two can never
    drift apart.
    """
    fence_marker: str | None = None
    fence_close: re.Pattern[str] | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if fence_marker is not None:
            yield i, line, _FENCED
            if fence_close is not None and fence_close.match(line):
                fence_marker = None
                fence_close = None
            i += 1
            continue
        opened = _FENCE_OPEN_RE.match(line)
        if opened:
            fence_marker = opened.group("marker")
            fence_close = _fence_close_re(fence_marker)
            yield i, line, _FENCED
            i += 1
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if "|" in line and "|" in nxt and _is_table_start(line, nxt):
            yield i, line, _TABLE
            yield i + 1, nxt, _TABLE
            i += 2
            while i < len(lines) and _bare_pipe_count(lines[i]):
                yield i, lines[i], _TABLE
                i += 1
            continue
        yield i, line, _PROSE
        i += 1


def _escape_gap(segment: str) -> str:
    """Escape bare pipes; leave already-escaped (odd backslash run) pipes."""

    def _repl(m: re.Match[str]) -> str:
        return m.group(0) if len(m.group(1)) % 2 else m.group(1) + "\\|"

    return _PIPE_RUN_RE.sub(_repl, segment)


def _escape_prose_line(line: str) -> str:
    """Escape pipes in *line* outside protected inline spans."""
    out: list[str] = []
    pos = 0
    for m in _PROTECTED_SPAN_RE.finditer(line):
        out.append(_escape_gap(line[pos : m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(_escape_gap(line[pos:]))
    return "".join(out)


def escape_prose_pipes(text: str) -> str:
    """Return *text* with every prose pipe escaped; verify before returning.

    Idempotent: an already-escaped body passes through unchanged.
    """
    if "|" not in text:
        return text
    lines = text.split("\n")
    escaped = [_escape_prose_line(line) if kind == _PROSE else line for _, line, kind in _classify(lines)]
    result = "\n".join(escaped)
    assert_pipes_escaped(result)
    return result


def assert_pipes_escaped(text: str) -> None:
    """Verify every prose pipe in *text* is backslash-escaped.

    Raises :class:`PipeEscapeError` naming the offending line. Protected
    regions (fences, tables, code spans, math, wikilinks) are exempt.
    """
    lines = text.split("\n")
    for idx, line, kind in _classify(lines):
        if kind != _PROSE or "|" not in line:
            continue
        pos = 0
        for m in _PROTECTED_SPAN_RE.finditer(line):
            _assert_gap(line, line[pos : m.start()], idx)
            pos = m.end()
        _assert_gap(line, line[pos:], idx)


def _assert_gap(line: str, gap: str, idx: int) -> None:
    for m in _PIPE_RUN_RE.finditer(gap):
        if len(m.group(1)) % 2 == 0:
            raise PipeEscapeError(f"unescaped prose pipe on line {idx + 1}: {line!r}")
