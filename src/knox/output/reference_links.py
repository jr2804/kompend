"""Rewrite in-text citation markers to reference links.

Given a :class:`~knox.core.pipeline.references.ReferenceStore`, this module
finds bibliographic citation markers in wiki page content and rewrites
them to links pointing at a ``references`` page.

Supported citation marker patterns:

- **Bracketed numeric**: ``[1]``, ``[2, 3]``, ``[1-3]``
- **Bracketed author-year**: ``[Smith2020]``, ``[Smith2020, p. 42]``
- **Parenthetical author-year**: ``(Smith et al., 2020)``, ``(Smith & Doe, 2019)``
- **Superscript-numeric** (as plain digits): ``¹``, ``²³`` (best-effort)

The rewriter is conservative: it only rewrites markers that match a known
reference in the store. Unknown markers are left untouched.
"""

from __future__ import annotations

import re

from knox.core.models.references import _YEAR_DIGITS, Reference
from knox.core.pipeline.references import ReferenceStore
from knox.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "REFERENCE_PAGE_SLUG",
    "REFERENCE_PAGE_TITLE",
    "build_reference_key_index",
    "format_bibliography_entry",
    "rewrite_citation_links",
]

#: Slug and title for the generated references page.
REFERENCE_PAGE_SLUG = "references"
REFERENCE_PAGE_TITLE = "References"

#: Link format for a single reference on the references page.
#: Uses an anchor fragment so links point to the specific entry.
_REF_LINK_FORMAT = "[{label}](#{anchor})"

#: Regex patterns for in-text citation markers.
# Bracketed numeric: [1], [2, 3], [1-3], [1, 3-5]
_BRACKET_NUMERIC_RE = re.compile(
    r"\[((?:\d+(?:\s*[-–,]\s*\d+)*))\]",
)
# Bracketed key: [Smith2020], [smith2020, p. 42]
_BRACKET_KEY_RE = re.compile(
    r"\[([A-Za-z][A-Za-z0-9_]+(?:\s*,\s*p\.\s*\d+)?)\]",
)
# Parenthetical author-year: (Smith, 2020), (Smith et al., 2020), (Smith & Doe, 2019)
_PAREN_AUTHOR_YEAR_RE = re.compile(
    r"\(([A-Z][A-Za-z]+(?:\s+(?:et al\.?|&|and)\s+[A-Z][A-Za-z]+)?)"
    r",?\s+(\d{4}[a-z]?)\)",
)

#: Maximum characters of a title used for anchor generation.
_ANCHOR_TITLE_MAX = 60


def _slugify_ref(ref: Reference) -> str:
    """Generate an HTML-safe anchor slug for a reference."""
    label = ref.short_label() or ref.id or ref.title[:_ANCHOR_TITLE_MAX]
    slug = label.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug or "ref"


def build_reference_key_index(
    refs: list[Reference],
) -> dict[str, Reference]:
    """Build a lookup index mapping citation keys to references.

    The index includes:

    - ``ref.id`` (lowercase) — the canonical slug/id
    - ``ref.short_label()`` (lowercase, no spaces) — e.g. ``"smith2020"``
    - ``"author+year"`` — e.g. ``"smith2020"``, ``"smith"`` (first author family + year)
    - ``"doi:..."`` — for DOI-based lookups
    """
    index: dict[str, Reference] = {}
    for ref in refs:
        if ref.id:
            index[ref.id.lower()] = ref
        label = ref.short_label()
        if label:
            key = label.lower().replace(" ", "")
            index.setdefault(key, ref)
        # author+year from structured fields
        if ref.author and ref.issued:
            a0 = ref.author[0]
            family = (a0.literal or a0.family or "").lower()
            year_digits = "".join(c for c in ref.issued if c.isdigit())[:_YEAR_DIGITS]
            if family and year_digits:
                index.setdefault(f"{family}{year_digits}", ref)
        if ref.doi:
            index[f"doi:{ref.doi.lower().strip()}"] = ref
    return index


def _format_ref_link(ref: Reference) -> str:
    """Format a citation link to the references page."""
    anchor = _slugify_ref(ref)
    label = ref.short_label()
    return _REF_LINK_FORMAT.format(label=label, anchor=anchor)


def _expand_numeric_range(spec: str) -> list[int]:
    """Expand a numeric citation spec like ``"1, 3-5"`` to ``[1, 3, 4, 5]``."""
    nums: list[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        range_match = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            nums.extend(range(start, end + 1))
        elif part.isdigit():
            nums.append(int(part))
    return nums


def _find_numeric_ref(num: int, refs: list[Reference]) -> Reference | None:
    """Find the Nth reference in reference-list order (1-indexed)."""
    if 1 <= num <= len(refs):
        return refs[num - 1]
    return None


def rewrite_citation_links(
    text: str,
    refs: list[Reference],
) -> str:
    r"""Rewrite in-text citation markers in *text* to reference links.

    Parameters
    ----------
    text:
        Wiki page body text (Markdown).
    refs:
        Ordered list of references for this source.

    Returns:
    -------
    The text with citation markers replaced by ``[label](#anchor)`` links.
    Unknown markers are left untouched.
    """
    if not refs or not text:
        return text

    key_index = build_reference_key_index(refs)

    # ── Pass 1: bracketed numeric [1], [2, 3], [1-3] ──────────────────
    def _replace_bracket_numeric(m: re.Match[str]) -> str:
        spec = m.group(1)
        nums = _expand_numeric_range(spec)
        if not nums:
            return m.group(0)
        links: list[str] = []
        all_found = True
        for num in nums:
            ref = _find_numeric_ref(num, refs)
            if ref is None:
                all_found = False
                break
            links.append(_format_ref_link(ref))
        if not all_found or not links:
            return m.group(0)
        return "[" + ", ".join(links) + "]"

    text = _BRACKET_NUMERIC_RE.sub(_replace_bracket_numeric, text)

    # ── Pass 2: bracketed key [Smith2020], [smith2020, p. 42] ─────────
    def _replace_bracket_key(m: re.Match[str]) -> str:
        raw = m.group(1)
        # Strip ", p. N" page suffix
        key = re.sub(r"\s*,\s*p\.\s*\d+", "", raw).lower()
        ref = key_index.get(key)
        if ref is None:
            return m.group(0)
        return _format_ref_link(ref)

    text = _BRACKET_KEY_RE.sub(_replace_bracket_key, text)

    # ── Pass 3: parenthetical author-year (Smith, 2020) ───────────────
    def _replace_paren_author_year(m: re.Match[str]) -> str:
        author_part = m.group(1)
        year = m.group(2)
        # Build candidate keys: "smith2020", "smith" + year
        first_word = author_part.split()[0].lower() if author_part.split() else ""
        first_word = re.sub(r"[^a-z]", "", first_word)
        for candidate in (f"{first_word}{year}", first_word):
            ref = key_index.get(candidate)
            if ref is not None:
                return _format_ref_link(ref)
        return m.group(0)

    text = _PAREN_AUTHOR_YEAR_RE.sub(_replace_paren_author_year, text)

    return text


def format_bibliography_entry(ref: Reference, number: int = 0) -> str:
    r"""Format a single reference as a bibliography entry.

    Produces a Markdown list item with an anchor for link targets::

        ### smith2020

        Smith, J. (2020). Deep Learning for NLP. *Journal of AI*, 12(3), 45–67. doi:10.1/xyz

    Parameters
    ----------
    ref:
        The reference to format.
    number:
        Optional 1-indexed position in the bibliography. ``0`` = no number.

    Returns:
    -------
    Formatted Markdown string.
    """
    anchor = _slugify_ref(ref)
    prefix = f"{number}. " if number else ""
    parts: list[str] = []

    authors = ref.display_authors()
    if authors:
        parts.append(authors + ".")
    if ref.issued:
        parts.append(f"({ref.issued}).")
    if ref.title:
        parts.append(f"*{ref.title}.*")
    if ref.container_title:
        parts.append(f"*{ref.container_title}*,")
    if ref.volume and ref.number:
        parts.append(f"{ref.volume}({ref.number}),")
    elif ref.volume:
        parts.append(f"{ref.volume},")
    if ref.pages:
        parts.append(f"{ref.pages}.")
    if ref.publisher:
        parts.append(ref.publisher + ".")
    if ref.doi:
        parts.append(f"doi:[{ref.doi}](https://doi.org/{ref.doi})")
    if ref.url and not ref.doi:
        parts.append(f"[{ref.url}]({ref.url})")

    body = " ".join(parts) if parts else ref.raw or ref.id

    return f'<a id="{anchor}"></a>{prefix}{body}\n'


def rewrite_source_references(
    body: str,
    source: str,
    store: ReferenceStore,
) -> str:
    """Rewrite citation markers in *body* using references for *source*.

    Convenience wrapper: loads references for a specific source from the
    store, then calls :func:`rewrite_citation_links`.
    """
    refs = store.get_source(source)
    if not refs:
        return body
    return rewrite_citation_links(body, refs)
