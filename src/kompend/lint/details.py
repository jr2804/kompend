"""The documented shape of a lint finding's ``details`` payload.

The registry lives in its own module because a dataclass default names it:
``field(default_factory=LintDetails)`` evaluates the name while the class body
runs, and the formatter sorts plain classes BELOW the dataclasses. An import
binds the name before any class body runs, so the default stays valid however
the module is sorted.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypedDict


class LintField(StrEnum):
    """Keys of a lint finding's ``details`` payload — the single owner.

    Every rule writes and reads ``details`` through these members; a bare
    string is a field-name literal (``.agents/CODE.md``, Antipatterns).
    ``tests/unit/test_lint_details.py`` pins this enum and
    :class:`LintDetails` to the same keys, so the two cannot drift.
    """

    # attachments.py — the file a finding is about, the pages the scan could
    # not read, the files it kept because a page links them, and the error a
    # failed fix records on the finding.
    PATH = "path"
    UNREADABLE_PAGES = "unreadable_pages"
    KEPT_LINKED = "kept_linked"
    ERROR = "error"
    # images.py, links.py — the page and the link target it points at.
    TARGET = "target"
    ALIAS_OF = "alias_of"
    ALIAS_SIMILARITY = "alias_similarity"
    # markdown.py — one rumdl finding. rumdl leaves the position and the rule
    # name empty for some findings, so those keys may be None.
    FILE = "file"
    LINE = "line"
    COLUMN = "column"
    RUMDL_RULE = "rumdl_rule"
    FIXABLE = "fixable"
    # metadata.py — the frontmatter fields a page is missing.
    MISSING_FIELDS = "missing_fields"
    # references.py — the duplicate reference groups.
    DUPLICATES = "duplicates"
    NEAR_DUPLICATES = "near_duplicates"
    # structure.py — the pages that share a title, and the heading levels a
    # page skips over.
    DUPLICATE_IDS = "duplicate_ids"
    START = "start"
    STOP = "stop"
    # llm_rules.py — an inconsistent claim.
    CLAIM = "claim"
    REASON = "reason"
    # llm_rules.py — the clarity score of a page.
    SCORE = "score"
    FEEDBACK = "feedback"
    # llm_rules.py — a coverage gap.
    CONCEPT = "concept"
    ALIASES = "aliases"
    # llm_rules.py — two pages that contradict each other.
    PAGE_A_ID = "page_a_id"
    PAGE_A_TITLE = "page_a_title"
    PAGE_B_ID = "page_b_id"
    PAGE_B_TITLE = "page_b_title"
    CLAIM_A = "claim_a"
    CLAIM_B = "claim_b"
    EXPLANATION = "explanation"
    CONFIDENCE = "confidence"
    # _remediation_gaps.py — the pages a remediation action worked on.
    SOURCES = "sources"
    MATCHED_CONCEPT = "matched_concept"
    ID = "id"


class LintDetails(TypedDict, total=False):
    """The documented shape of a finding's ``details`` payload.

    Every key a lint rule writes is declared here, so a typo in a key is a
    type error instead of a silently different record in the lint history.
    A rule owns the keys it writes; a fix path may add ``error`` to a finding
    that already carries ``path``.
    """

    # attachments.py
    path: str
    unreadable_pages: list[str]
    kept_linked: int
    error: str
    # images.py, links.py
    target: str
    alias_of: str
    alias_similarity: float
    # markdown.py
    file: str
    line: int | None
    column: int | None
    rumdl_rule: str | None
    fixable: bool
    # metadata.py
    missing_fields: list[str]
    # references.py
    duplicates: list[dict[str, Any]]
    near_duplicates: list[dict[str, Any]]
    # structure.py
    duplicate_ids: list[str]
    start: int
    stop: int
    # llm_rules.py
    claim: str
    reason: str
    score: float
    feedback: str
    concept: str
    aliases: list[str]
    page_a_id: str
    page_a_title: str
    page_b_id: str
    page_b_title: str
    claim_a: str
    claim_b: str
    explanation: str
    confidence: float
    # _remediation_gaps.py
    sources: list[str]
    matched_concept: str
    id: str
