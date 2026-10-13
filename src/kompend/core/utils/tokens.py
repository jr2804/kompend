"""Coarse token-count heuristic shared across stages.

A single char-based estimate (~4 chars/token) keeps budget math consistent
between chunking and compilation without pulling in a tokenizer.
"""

from __future__ import annotations

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token; never less than 1)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)
