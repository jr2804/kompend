"""Recognising a context-window overflow reported by the provider.

A provider that refuses an oversized request says so in its own words, and
— crucially — it usually names *its* limit: ``request (33145 tokens)
exceeds the available context size (32768 tokens), try increasing it``.
That number is authoritative: it is what the server enforces, whatever the
configuration or the metadata claims.

This module owns two questions:

- :func:`is_context_overflow` — is this failure the window, or something else?
- :func:`stated_context_size` — what limit did the provider say it enforces?

Together they let a stage learn the server's real window instead of dying on
a repeated 400 (see the ``knox-60pg`` bug: a server started with
``-c 32768`` while the configuration asked for 64000).
"""

from __future__ import annotations

import re
from typing import Final

#: Error-text markers of a context-window overflow across providers
#: (Ollama, OpenAI-compatible gateways). Matched case-insensitively.
OVERFLOW_MARKERS: Final[tuple[str, ...]] = (
    "exceeds the available context size",
    "context_length_exceeded",
    "maximum context length",
    "reduce the length of the messages",
    "reduce the length of your input",
)

#: Patterns that carry the provider's own context limit. The first group
#: matches "exceeds the available context size (32768 tokens)"; the second
#: matches OpenAI-style "maximum context length is 32768 tokens"; the third
#: matches a bare "context window is 32768".
_STATED_SIZE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"exceeds the available context size[^0-9]{0,40}?(\d[\d_,]*)"),
    re.compile(r"maximum context length[^0-9]{0,40}?(\d[\d_,]*)"),
    re.compile(r"context (?:window|length)[^0-9]{0,20}?(?:is|of)[^0-9]{0,20}?(\d[\d_,]*)"),
)


def is_context_overflow(exc: BaseException) -> bool:
    """Whether *exc* is a context-window overflow reported by the provider.

    Walks the exception chain (gateway errors arrive wrapped) and matches
    the markers providers use for "the request does not fit the window".
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        blob = f"{type(current).__name__}: {current}".lower()
        if any(marker in blob for marker in OVERFLOW_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def stated_context_size(exc: BaseException) -> int | None:
    """The context limit the provider named in *exc*, if it named one.

    Returns the token count the server says it enforces (``32768`` for
    ``request (33145 tokens) exceeds the available context size (32768
    tokens)``), or ``None`` when the message carries no number or the number
    cannot be trusted (the requested size rather than the limit).

    The distinction matters: the *request* size is what kompend sent, the
    *limit* is what the server allows. Only the limit may lower the window.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        blob = f"{type(current).__name__}: {current}"
        for pattern in _STATED_SIZE_PATTERNS:
            match = pattern.search(blob)
            if match:
                size = int(match.group(1).replace(",", "").replace("_", ""))
                if size > 0:
                    return size
        current = current.__cause__ or current.__context__
    return None
