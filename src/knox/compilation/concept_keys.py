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

WIKILINK_ALIAS_MIN_JACCARD = 0.5
#: Score assigned to substring matches (e.g. ``P.863`` ⊆ candidate title).
#: Above :data:`WIKILINK_ALIAS_MIN_JACCARD` so substring wins on ties, but
#: below 1.0 so a higher Jaccard can still override.
WIKILINK_ALIAS_SUBSTRING_SCORE = 0.75


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

    Only triggered when *target* has no exact match in *alias_index*. Uses
    token-overlap (Jaccard) against known titles to flag "the LLM meant
    this page but used the wrong short alias" cases. Conservative by
    design — returns ``None`` (not an alias) unless the match is clear.

    Special case: short identifiers like ``"P.863"`` rarely overlap with
    long titles on tokens alone. We also accept matches where the target
    is a **contiguous substring** of the candidate title (case-insensitive)
    with a length ≥4 chars — this catches ``P.863`` → ``Full-scale
    database (P.863 evaluation)``, ``ITU-T P.863`` → ``ITU-T P.800.1``
    prefix matches, etc., without false-positiving on trivial substrings.

    Returns the matched ``node.id``, or ``None`` when no alias candidate
    is strong enough.
    """
    target_norm = normalize_concept_key(target)
    if not target_norm:
        return None

    # Parenthetical acronyms in the target ("Full Name (MOS-LQO)") resolve
    # directly against page ids/titles — the expansion's tokens are too
    # dissimilar from the acronym page for Jaccard to connect them.
    for acro in parenthetical_acronyms(target):
        direct = alias_index.get(acro.lower()) or alias_index.get(normalize_concept_key(acro))
        if direct is not None:
            return direct
    # Compare against period-stripped forms so ``p.863`` (short alias) matches
    # ``p 863`` (slug-normalized, period replaced by hyphen-then-space).
    target_no_punct = target_norm.replace(".", "")
    target_norm_tokens = {t for t in target_norm.split() if len(t) > 2}

    best_id: str | None = None
    best_score = WIKILINK_ALIAS_MIN_JACCARD

    for key, page_id in alias_index.items():
        if len(key) <= 2:
            continue
        key_tokens = set(key.split())
        if not key_tokens:
            continue
        # Token Jaccard (catches MOS-LQO ↔ MOS-LQO Fullband).
        if target_norm_tokens:
            overlap = len(target_norm_tokens & key_tokens)
            if overlap:
                union = target_norm_tokens | key_tokens
                score = overlap / len(union)
                if score > best_score:
                    best_score = score
                    best_id = page_id
        # Substring containment for short references like "P.863".
        # Compare against the key with hyphens, periods, and spaces
        # collapsed — so ``p.863`` (short alias) matches
        # ``full-scale-database-p-863-evaluation`` (slug).
        if len(target_no_punct) >= 4:
            key_normalized = key.replace("-", "").replace(".", "").replace(" ", "")
            if (target_no_punct in key_normalized) and best_score < WIKILINK_ALIAS_SUBSTRING_SCORE:
                best_score = WIKILINK_ALIAS_SUBSTRING_SCORE
                best_id = page_id

    return best_id


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
