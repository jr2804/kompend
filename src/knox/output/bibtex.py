"""BibTeX export for :class:`~knox.core.models.references.Reference`.

Converts the CSL JSON-aligned :class:`Reference` objects to BibTeX format.
BibTeX generation is implemented natively — no external dependency required.

CSL type → BibTeX entry-type mapping follows the `citeproc-py` /
`pandoc-citeproc` conventions.
"""

from __future__ import annotations

import re
from io import StringIO

from knox.core.models.references import _YEAR_DIGITS, Reference
from knox.logging import get_logger

logger = get_logger(__name__)

#: Maximum characters of an author family name used in a citation key.
_AUTHOR_KEY_MAX = 20

__all__ = [
    "reference_to_bibtex_entry",
    "references_to_bibtex",
]

#: CSL type → BibTeX entry-type.
#: Unmapped CSL types fall back to ``misc``.
_CSL_TO_BIBTEX_TYPE: dict[str, str] = {
    "article-journal": "article",
    "article-magazine": "article",
    "article-newspaper": "article",
    "book": "book",
    "chapter": "incollection",
    "paper-conference": "inproceedings",
    "report": "techreport",
    "thesis": "phdthesis",
    "webpage": "misc",
    "manuscript": "unpublished",
    "patent": "misc",
    "interview": "misc",
    "speech": "misc",
    "personal-communication": "misc",
    "map": "misc",
    "broadcast": "misc",
    "software": "misc",
    "dataset": "misc",
    "entry-dictionary": "misc",
    "entry-encyclopedia": "misc",
    "figure": "misc",
    "graphic": "misc",
    "legal-case": "misc",
    "legislation": "misc",
    "motion_picture": "misc",
    "musical_score": "misc",
    "pamphlet": "booklet",
    "post": "misc",
    "post-weblog": "misc",
    "review": "article",
    "review-book": "misc",
    "song": "misc",
    "treaty": "misc",
}

#: Characters that need escaping in BibTeX field values.
_BIBTEX_SPECIAL = str.maketrans(
    {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
)


def _make_citation_key(ref: Reference, index: int) -> str:
    """Generate a BibTeX citation key.

    Format: ``FirstAuthorYear`` or ``ref<N>`` as fallback.
    """
    first_author = ""
    if ref.author:
        a0 = ref.author[0]
        name = a0.literal or a0.family
        if name:
            # Keep only letters, collapse spaces
            first_author = re.sub(r"[^A-Za-z]", "", name).lower()
            first_author = first_author[:_AUTHOR_KEY_MAX]

    year = ""
    if ref.issued:
        digits = "".join(c for c in ref.issued if c.isdigit())
        year = digits[:_YEAR_DIGITS]

    if first_author and year:
        key = f"{first_author}{year}"
    elif ref.id:
        key = re.sub(r"[^A-Za-z0-9]", "", ref.id).lower()
    else:
        key = f"ref{index}"

    # Ensure non-empty and alphanumeric
    return key or f"ref{index}"


def _escape_value(value: str) -> str:
    """Escape special characters in a BibTeX field value."""
    if not value:
        return ""
    return value.translate(_BIBTEX_SPECIAL)


def _format_bibtex_authors(authors: list) -> str:
    """Format author list as BibTeX ``" and "``-separated string.

    Each author is rendered as ``Family, Given`` or ``literal`` form.
    """
    parts: list[str] = []
    for a in authors:
        if a.literal:
            parts.append(a.literal)
        elif a.family or a.given:
            if a.family and a.given:
                parts.append(f"{a.family}, {a.given}")
            elif a.family:
                parts.append(a.family)
            else:
                parts.append(a.given)
    return " and ".join(parts)


def _extract_year(issued: str) -> str:
    """Extract the 4-digit year from the ``issued`` field."""
    digits = "".join(c for c in issued if c.isdigit())
    return digits[:_YEAR_DIGITS] if len(digits) >= _YEAR_DIGITS else issued


def reference_to_bibtex_entry(ref: Reference, index: int = 0) -> str:
    """Format a single :class:`Reference` as a BibTeX entry.

    Parameters
    ----------
    ref:
        The reference to convert.
    index:
        1-indexed position for fallback citation key generation.

    Returns:
    -------
    A BibTeX entry string like::

        @article{smith2020,
          author = {Smith, John and Doe, Jane},
          title = {Deep Learning for NLP},
          ...
        }
    """
    entry_type = _CSL_TO_BIBTEX_TYPE.get(ref.type, "misc")
    cite_key = _make_citation_key(ref, index)
    lines: list[str] = [f"@{entry_type}{{{cite_key},"]

    # Fields are added in conventional BibTeX order.
    authors = _format_bibtex_authors(ref.author)
    if authors:
        lines.append(f"  author    = {{{_escape_value(authors)}}},")

    if ref.title:
        lines.append(f"  title     = {{{_escape_value(ref.title)}}},")

    year = _extract_year(ref.issued)
    if year:
        lines.append(f"  year      = {{{year}}},")

    if ref.container_title:
        field = "booktitle" if entry_type in ("inproceedings", "incollection") else "journal"
        lines.append(f"  {field:<9} = {{{_escape_value(ref.container_title)}}},")

    if ref.volume:
        lines.append(f"  volume    = {{{ref.volume}}},")

    if ref.number:
        lines.append(f"  number    = {{{ref.number}}},")

    if ref.pages:
        # BibTeX uses "--" for page ranges
        pages = ref.pages.replace("-", "--").replace("–", "--")
        lines.append(f"  pages     = {{{pages}}},")

    if ref.publisher:
        lines.append(f"  publisher = {{{_escape_value(ref.publisher)}}},")

    if ref.doi:
        lines.append(f"  doi       = {{{ref.doi}}},")

    if ref.url:
        lines.append(f"  url       = {{{ref.url}}},")

    # Remove trailing comma from the last field.
    if len(lines) > 1:
        lines[-1] = lines[-1].rstrip(",")

    lines.append("}")
    return "\n".join(lines)


def references_to_bibtex(refs: list[Reference]) -> str:
    """Convert a list of references to a complete ``.bib`` document.

    Entries are separated by blank lines. An empty list returns ``""``.
    """
    if not refs:
        return ""
    buf = StringIO()
    for i, ref in enumerate(refs, 1):
        if i > 1:
            buf.write("\n\n")
        buf.write(reference_to_bibtex_entry(ref, index=i))
    buf.write("\n")
    return buf.getvalue()
