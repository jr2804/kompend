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

from knox.core.text.matching import best_match

# Strips a trailing parenthetical acronym/qualifier, e.g. "Foo (Bar)" → "Foo".
_ACRONYM_RE = re.compile(r"\s*\([^)\d][^)]*\)\s*$")

#: Parenthetical parts inside a link target, e.g. "MOS Listening Quality
#: Objective (MOS-LQO)" → "MOS-LQO".
_PAREN_RE = re.compile(r"\(([^)]+)\)")


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


# ── Wikilink alias resolution ─────────────────────────────────────────

#: Acceptance threshold for the rapidfuzz similarity behind alias
#: resolution (same 0–100 scale as
#: :data:`knox.core.text.matching.DEFAULT_SIMILARITY_THRESHOLD`).
WIKILINK_ALIAS_MIN_SIMILARITY = 85.0


def is_structural_concept(name: str) -> bool:
    """Return True if *name* is a table/spreadsheet structural artifact.

    These are column headers, field labels, and metadata labels that
    small models extract from table-heavy sources but which never
    warrant a wiki page. The check uses the normalized key so that
    casing, punctuation, and parenthetical acronyms don't bypass it.
    """
    key = normalize_concept_key(name)
    return bool(_STRUCTURAL_CONCEPT_RE.match(key))


def nearest_alias(target: str, alias_index: dict[str, str]) -> str | None:
    """Return the page id most likely intended by *target*.

    Only triggered when *target* has no exact match in *alias_index*.
    Scoring routes through :func:`knox.core.text.matching.similarity`
    (rapidfuzz — the single source of truth for string similarity), so
    token order, punctuation variants (``P.10/G.100`` ≡ ``P 10 G 100``),
    and parenthetical extras all match without bespoke logic.

    Special case kept here (domain logic, not similarity math):
    parenthetical acronyms in the target ("Full Name (MOS-LQO)") resolve
    directly against page ids/titles — the expansion's tokens are too
    dissimilar from the acronym page for token similarity to connect.

    Returns the matched ``node.id``, or ``None`` when no alias candidate
    is strong enough.
    """
    target_norm = normalize_concept_key(target)
    if not target_norm:
        return None

    for acro in parenthetical_acronyms(target):
        direct = alias_index.get(acro.lower()) or alias_index.get(normalize_concept_key(acro))
        if direct is not None:
            return direct

    matched = best_match(target_norm, set(alias_index), threshold=WIKILINK_ALIAS_MIN_SIMILARITY)
    return alias_index[matched[0]] if matched else None


def parenthetical_acronyms(target: str) -> list[str]:
    """Acronym-like parenthetical parts of a wikilink target, in order.

    ``"MOS Listening Quality Objective (MOS-LQO)"`` → ``["MOS-LQO"]``.
    Lowercase/generic parentheticals (``"Echo (in telephony)"``) are
    excluded — only uppercase-dominant, short parts qualify as acronyms.
    """
    out: list[str] = []
    for m in _PAREN_RE.finditer(target):
        part = m.group(1).strip()
        letters = [c for c in part if c.isalpha()]
        if not letters or len(part) > 12:
            continue
        upper_ratio = sum(c.isupper() for c in letters) / len(letters)
        if upper_ratio >= 0.6:
            out.append(part)
    return out


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
