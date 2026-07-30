"""Shared concept-key helpers used by every compiler backend.

Both :class:`~knox.compilation.graph.GraphCompiler` (the ``wicer`` base)
and :class:`~knox.compilation.two_phase.TwoPhaseCompiler` need to decide
whether a candidate name is a real concept or a structural artifact pulled
out of tables/spreadsheets. Putting the normalization + structural filter
here keeps a single source of truth so the two backends never drift apart.

The module is deliberately dependency-free (stdlib only) so that importing
it from :mod:`graph` cannot create a circular import: ``graph → base →
_two_phase_core`` would otherwise loop back to ``graph``.
"""

from __future__ import annotations

import re

# Strips a trailing parenthetical acronym/qualifier, e.g. "Foo (Bar)" → "Foo".
_ACRONYM_RE = re.compile(r"\s*\([^)\d][^)]*\)\s*$")


# Structural artifacts from spreadsheet/table-heavy sources that the LLM
# frequently mis-extracts as concepts (e.g. "Table Data", "data_id value",
# "Name", "Source Filename"). Matched against the normalized concept key
# (lowercase, hyphens → spaces, stripped of parentheticals). Underscores
# are kept by the normalizer so the regex uses [._ ] to cover both.
_STRUCTURAL_CONCEPT_RE = re.compile(
    r"^(?:"
    r"table[._ ]?data(?:[._ ]?(?:sources?|definitions?|sources? definitions?|definition schema))?s?"
    r"|data[._ ]?(?:id|source|definition)s?"
    r"|data[._ ]?id[._ ]?value(?:[._ ]?source[._ ]?filename)?"
    r"|data"
    r"|name"
    r"|source(?:[._ ]?filename)?"
    r"|value"
    r"|definition"
    r"|source[._ ]?table"
    r")$"
)


def is_structural_concept(name: str) -> bool:
    """Return True if *name* is a table/spreadsheet structural artifact.

    These are column headers, field labels, and metadata labels that
    small models extract from table-heavy sources but which never
    warrant a wiki page. The check uses the normalized key so that
    casing, punctuation, and parenthetical acronyms don't bypass it.
    """
    key = normalize_concept_key(name)
    return bool(_STRUCTURAL_CONCEPT_RE.match(key))


def normalize_concept_key(name: str) -> str:
    """Normalize a concept name for merge-key deduplication.

    Strips parenthetical acronyms, normalizes case/hyphens/whitespace,
    and collapses English plurals on the last word (heuristic).

    Examples::

        "Deep Neural Networks (DNNs)"  → "deep neural network"
        "Short-Time Objective Intelligibility (STOI)" → "short time objective intelligibility"
        "Deep neural network"  → "deep neural network"
        "Phoneme posteriorgrams" → "phoneme posteriorgram"
    """
    text = _ACRONYM_RE.sub("", name)
    text = text.replace("-", " ").lower().strip()
    if not text:
        return ""
    words = text.split()
    last = words[-1]
    if len(last) > 3:
        if last.endswith("ies"):
            words[-1] = last[:-3] + "y"
        elif last.endswith(("ches", "shes", "xes", "zes", "oes")):
            words[-1] = last[:-2]
        elif last.endswith("s") and not last.endswith("ss"):
            words[-1] = last[:-1]
    return " ".join(words).strip()
