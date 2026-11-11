"""The CSL type → BibTeX entry-type table, as data.

BibTeX has fourteen entry types; CSL-JSON (the citation vocabulary a
:class:`~kompend.core.models.references.Reference` carries) has forty-one. The
mapping below is the `citeproc-py` / `pandoc-citeproc` convention for the
difference, and it is a foreign specification, not kompend's vocabulary: the
keys are CSL types kompend does not declare, and the values are BibTeX's own
entry names. It lives here, read-only, so a reader can diff it against the CSL
schema without walking past the writer that consumes it.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

__all__ = ["CSL_TO_BIBTEX_TYPE", "DEFAULT_BIBTEX_ENTRY"]

#: Entry type for a CSL type the table does not name. ``misc`` is BibTeX's
#: catch-all, so an unknown CSL type still produces a valid entry.
DEFAULT_BIBTEX_ENTRY: Final[str] = "misc"

#: CSL type → BibTeX entry-type, following `citeproc-py`/`pandoc-citeproc`.
CSL_TO_BIBTEX_TYPE: Final[MappingProxyType[str, str]] = MappingProxyType(
    {
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
)
