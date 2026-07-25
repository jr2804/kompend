"""Reference-section detection and excision.

Given the full text of a source document, detect and separate the
bibliographic reference section from the body content. This lets the
pipeline send only the body to the extraction LLM (saving tokens for
substantive content) and send the reference section to the
:class:`~knox.agents.reference_extraction.ReferenceExtractionAgent`
for structured parsing.

Strategy (two-stage):

1. **Heading regex** — detect Markdown/ATX headings or numbered
   sections whose title matches common reference-section names
   ("References", "Bibliography", "Works Cited", etc.). Everything
   from that heading to the next same-or-higher-level heading is the
   reference section.
2. **Fallback** — if no heading is found, return the text unchanged
   (the ``ReferenceExtractionAgent`` can still scan the tail as a
   last resort, but for Phase 1 we rely on explicit headings).
"""

from __future__ import annotations

import re

from knox.logging import get_logger

logger = get_logger(__name__)

__all__ = ["excise_references", "is_likely_reference_heading"]

# Common reference-section heading titles (case-insensitive).
_REFERENCE_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s+(references|bibliography|works\s+cited|literature\s+cited"
    r"|sources|cited\s+references|references\s+and\s+notes)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Numbered variant: "9 References", "10. Bibliography", "8. Literature Cited"
_NUMBERED_REF_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s+)?\d+\.?\s+"
    r"(references|bibliography|works\s+cited|literature\s+cited"
    r"|references\s+and\s+notes)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Heading line in general (ATX or Setext underline)
_HEADING_RE = re.compile(
    r"^(?:\s{0,3}#{1,6}\s+.+|.+[ \t]*\n=+|.+[ \t]*\n-+)$",
    re.MULTILINE,
)


def is_likely_reference_heading(line: str) -> bool:
    """Return ``True`` if *line* looks like a reference-section heading."""
    return bool(
        _REFERENCE_HEADING_RE.match(line) or _NUMBERED_REF_RE.match(line),
    )


def excise_references(text: str) -> tuple[str, str]:
    """Split *text* into ``(body, references)``.

    Finds the first reference-section heading and treats everything from
    that point to the end of the document as the reference section
    (bibliographies are always terminal). The heading itself is included
    in the references part.

    If no reference section is detected, returns ``(text, "")``.

    Parameters
    ----------
    text:
        Full source text (typically the concatenation of all chunks
        for a single source file).

    Returns:
    -------
    tuple of ``(body, references)``
    """
    # Find the first heading match (ATX or numbered)
    match = _REFERENCE_HEADING_RE.search(text)
    if not match:
        match = _NUMBERED_REF_RE.search(text)

    if not match:
        return text, ""

    split_point = match.start()
    body = text[:split_point].rstrip()
    references = text[split_point:].strip()

    logger.debug(
        "Excised references: body=%d chars, refs=%d chars",
        len(body),
        len(references),
    )
    return body, references
