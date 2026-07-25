"""Bibliographic reference model (CSL JSON-aligned).

A :class:`Reference` represents a single bibliographic citation extracted
from a source document's reference section. The field names follow
CSL JSON (Citation Style Language) conventions so the data can be
rendered by ``citeproc-py`` (Phase 2+) without remapping.

This is distinct from :class:`knox.core.models.citations.Citation`,
which represents in-text provenance markers like ``[source:42-58]``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "Author",
    "Reference",
    "ReferenceExtractionOutput",
    "ReferenceResolutionOutput",
]

#: Maximum length of a short display label title fallback.
_SHORT_LABEL_TITLE_MAX = 40
#: Number of digits in a year (used for label extraction from issued date).
_YEAR_DIGITS = 4


class Author(BaseModel):
    """A single author in CSL structured form.

    Either ``literal`` (for org / single-name entities) or the
    ``given`` + ``family`` pair is populated.
    """

    family: str = ""
    given: str = ""
    literal: str = ""

    def to_csl(self) -> dict[str, str]:
        """Return the CSL-compatible dict form."""
        if self.literal:
            return {"literal": self.literal}
        d: dict[str, str] = {}
        if self.family:
            d["family"] = self.family
        if self.given:
            d["given"] = self.given
        return d


class Reference(BaseModel):
    """A single bibliographic reference, CSL JSON-aligned.

    Fields follow the CSL 1.0.2 variable names (mapped to snake_case)
    so the data round-trips through ``citeproc-py`` with minimal
    transformation. Extra ``raw``, ``citing_source``, and
    ``in_text_keys`` fields are KNOX-specific metadata that do not
    belong in CSL output.
    """

    # ── CSL fields ───────────────────────────────────────────────────
    id: str = ""
    type: str = ""  # article-journal, book, chapter, paper-conference, webpage, report, thesis…
    title: str = ""
    author: list[Author] = Field(default_factory=list)
    issued: str = ""  # year or "date-parts" representation
    container_title: str = ""
    volume: str = ""
    number: str = ""  # CSL "issue"
    pages: str = ""
    publisher: str = ""
    doi: str = ""
    url: str = ""

    # ── KNOX-specific metadata ───────────────────────────────────────
    raw: str = ""  # original unparsed citation string
    citing_source: str = ""  # source file that cited this reference
    in_text_keys: list[str] = Field(default_factory=list)

    def to_csl(self) -> dict[str, Any]:
        """Return a CSL-JSON-compatible dict (excludes KNOX-only fields)."""
        d: dict[str, Any] = {}
        if self.id:
            d["id"] = self.id
        if self.type:
            d["type"] = self.type
        if self.title:
            d["title"] = self.title
        if self.author:
            authors = [a.to_csl() for a in self.author]
            d["author"] = authors
        if self.issued:
            d["issued"] = {"raw": self.issued}
        if self.container_title:
            d["container-title"] = self.container_title
        if self.volume:
            d["volume"] = self.volume
        if self.number:
            d["issue"] = self.number
        if self.pages:
            d["page"] = self.pages
        if self.publisher:
            d["publisher"] = self.publisher
        if self.doi:
            d["DOI"] = self.doi
        if self.url:
            d["URL"] = self.url
        return d

    def display_authors(self) -> str:
        """Return a human-readable author string."""
        if not self.author:
            return ""
        names: list[str] = []
        for a in self.author:
            if a.literal:
                names.append(a.literal)
            elif a.family:
                names.append(a.given + " " + a.family if a.given else a.family)
        return ", ".join(names)

    def short_label(self) -> str:
        """Return a compact label for dedup / display (Author Year)."""
        first_author = ""
        if self.author:
            a0 = self.author[0]
            first_author = a0.literal or a0.family or ""
        year = ""
        if self.issued:
            # Extract just the year digits for the label
            digits = "".join(c for c in self.issued if c.isdigit())
            year = digits[:_YEAR_DIGITS]
        parts = [p for p in (first_author, year) if p]
        if parts:
            return " ".join(parts)
        return self.title[:_SHORT_LABEL_TITLE_MAX] if self.title else self.id


class ReferenceExtractionOutput(BaseModel):
    """LLM output schema for ``compilation/reference_extraction_prompt.md.j2``.

    The agent returns a list of parsed :class:`Reference` objects.
    """

    references: list[Reference] = Field(default_factory=list)


class ReferenceResolutionOutput(BaseModel):
    """LLM output schema for ``compilation/reference_resolver_prompt.md.j2``.

    The resolver agent takes a single reference's raw text and attempts to
    recover a canonical ``doi`` and/or ``url`` when they are missing. It is
    intentionally permissive: ``resolved`` is ``False`` when nothing
    confident could be found, so callers can skip without corrupting data.
    """

    resolved: bool = False
    doi: str = ""
    url: str = ""
    note: str = ""
