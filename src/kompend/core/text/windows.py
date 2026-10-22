"""Text windowing — split long content into bounded windows without losing any of it.

A budget may size one request; it may never shorten the content that is
processed. The helper in this module turns "this content does not fit into
one call" into "this content is processed in several calls" — the caller
pays in speed, not in coverage.
"""

from __future__ import annotations


def split_text_windows(text: str, limit: int) -> list[str]:
    """Split *text* into windows of at most *limit* characters.

    Nothing is dropped: content that exceeds the limit becomes several
    windows, which costs extra calls (speed) but never loses the tail.
    Windows break on line boundaries; a single line longer than *limit* is
    hard-split.

    Args:
        text: The content to window.
        limit: Maximum characters per window. A limit of 0 or less means
            "no limit" — the text is returned as a single window.

    Returns:
        The windows, in order. An empty text yields no windows.
    """
    if limit <= 0 or len(text) <= limit:
        return [text] if text else []

    windows: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if len(line) > limit:
            if current:
                windows.append("".join(current))
                current = []
                current_len = 0
            for start in range(0, len(line), limit):
                windows.append(line[start : start + limit])
            continue
        if current and current_len + len(line) > limit:
            windows.append("".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line)
    if current:
        windows.append("".join(current))
    return windows
