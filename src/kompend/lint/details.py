"""The documented shape of a lint finding's ``details`` payload.

The registry lives in its own module because a dataclass default names it:
``field(default_factory=LintDetails)`` evaluates the name while the class body
runs, and the formatter sorts plain classes BELOW the dataclasses. An import
binds the name before any class body runs, so the default stays valid however
the module is sorted.
"""

from __future__ import annotations

from typing import Any, TypedDict

# ``from`` is a Python keyword, so the heading-skip keys need the functional form.
_HeadingSkipDetails = TypedDict("_HeadingSkipDetails", {"from": int, "to": int}, total=False)


class LintDetails(_HeadingSkipDetails, total=False):
    """The documented shape of a finding's ``details`` payload.

    Every key a lint rule writes is declared here, so a typo in a key is a
    type error instead of a silently different record in the lint history.
    A rule owns the keys it writes; a fix path may add ``error`` to a
    finding that already carries ``path``.
    """

    # attachments.py — the file a finding is about, the pages the scan could
    # not read, the files it kept because a page links them, and the error a
    # failed fix records on the finding.
    path: str
    unreadable_pages: list[str]
    kept_linked: int
    error: str
    # images.py, links.py — the page and the link target it points at.
    target: str
    alias_of: str
    alias_similarity: float
    # markdown.py — one rumdl finding. rumdl leaves the position and the rule
    # name empty for some findings, so those keys may be None.
    file: str
    line: int | None
    column: int | None
    rumdl_rule: str | None
    fixable: bool
    # metadata.py — the frontmatter fields a page is missing.
    missing_fields: list[str]
    # references.py — the duplicate reference groups.
    duplicates: list[dict[str, Any]]
    near_duplicates: list[dict[str, Any]]
    # structure.py — the pages that share a title.
    duplicate_ids: list[str]
    # llm_rules.py — an inconsistent claim.
    claim: str
    reason: str
    # llm_rules.py — the clarity score of a page.
    score: float
    feedback: str
    # llm_rules.py — a coverage gap.
    concept: str
    aliases: list[str]
    # llm_rules.py — two pages that contradict each other.
    page_a_id: str
    page_a_title: str
    page_b_id: str
    page_b_title: str
    claim_a: str
    claim_b: str
    explanation: str
    confidence: float
    # _remediation_gaps.py — the pages a remediation action worked on.
    sources: list[str]
    matched_concept: str
    id: str
