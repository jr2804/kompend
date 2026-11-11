"""The documented shape of a lint finding's ``details`` payload.

The registry lives in its own module because a dataclass default names it:
``field(default_factory=LintDetails)`` evaluates the name while the class body
runs, and the formatter sorts plain classes BELOW the dataclasses. An import
binds the name before any class body runs, so the default stays valid however
the module is sorted.

A key that names a file holds a :class:`pathlib.Path`, never a string, and its
name says so: ``attachment_file``, not ``path``; ``markdown_file``, not
``file``. A bare ``path``/``file``/``target`` names nothing a reader can act
on. The payload is relative while it lives in memory and a relative POSIX
string once it is persisted — :func:`details_for_storage` and
:func:`details_from_storage` are the only two conversions, so a vault stays
relocatable and no absolute path reaches ``.kompend/kompend.db``.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Final, TypedDict, cast

from kompend.types import JSONDict, JSONValue


class LintField(StrEnum):
    """Keys of a lint finding's ``details`` payload — the single owner.

    Every rule writes and reads ``details`` through these members; a bare
    string is a field-name literal (``.agents/CODE.md``, Antipatterns).
    ``tests/unit/test_lint_details.py`` pins this enum and
    :class:`LintDetails` to the same keys, so the two cannot drift.
    """

    # attachments.py — the attachment file a finding is about (vault-relative),
    # the pages the scan could not read, the files it kept because a page links
    # them, and the error a failed fix records on the finding.
    ATTACHMENT_FILE = "attachment_file"
    # sources.py — the source file a page claims but the vault does not have.
    # This key names the finding's payload, not ``WikiNode.metadata``: the same
    # words name two different contracts, and this enum owns the payload one.
    SOURCE_FILE = "source_file"
    UNREADABLE_PAGES = "unreadable_pages"
    KEPT_LINKED = "kept_linked"
    ERROR = "error"
    # images.py, links.py — a wikilink target (a page name, not a file), and
    # the page title a short-form alias was resolved to.
    TARGET = "target"
    ALIAS_OF = "alias_of"
    ALIAS_SIMILARITY = "alias_similarity"
    # markdown.py — one rumdl finding. The file is vault-relative, so a moved
    # vault keeps its findings; rumdl leaves the position and the rule name
    # empty for some findings, so those keys may be None.
    MARKDOWN_FILE = "markdown_file"
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
    # _remediation_gaps.py — the sources a remediation action drew on, the
    # concept it matched, and the page it wrote.
    SOURCES = "sources"
    MATCHED_CONCEPT = "matched_concept"
    ID = "id"


class DuplicateReferenceDetail(TypedDict, total=False):
    """One ``duplicates`` entry — a reference another source cites too.

    ``type`` says which identity matched the pair (``doi``, ``title-author``,
    ``raw``); the remaining keys are the evidence for that identity, so each is
    absent when the match did not need it.
    """

    type: str
    doi: str
    author: str
    title: str
    raw: str
    sources: list[str]
    refs: list[str]


class NearDuplicateReferenceDetail(TypedDict, total=False):
    """One ``near_duplicates`` entry — the fuzzy-merge count for one threshold."""

    type: str
    removed: int
    threshold: float
    remaining: int


#: The members whose value is a filesystem path. They hold a ``Path`` in
#: memory and a relative POSIX string in the store; :data:`LintDetails` declares
#: the ``Path``, and ``tests/unit/test_lint_details.py`` pins this set to those
#: annotations so a new path key cannot be added as a bare string.
_PATH_KEYS: Final[frozenset[str]] = frozenset({LintField.ATTACHMENT_FILE, LintField.MARKDOWN_FILE, LintField.SOURCE_FILE})


class LintDetails(TypedDict, total=False):
    """The documented shape of a finding's ``details`` payload.

    Every key a lint rule writes is declared here, so a typo in a key is a
    type error instead of a silently different record in the lint history.
    A rule owns the keys it writes; a fix path may add ``error`` to a finding
    that already carries ``attachment_file``.
    """

    # attachments.py
    attachment_file: Path
    unreadable_pages: list[str]
    kept_linked: int
    error: str
    # images.py, links.py
    target: str
    alias_of: str
    alias_similarity: float
    # markdown.py
    markdown_file: Path
    line: int | None
    column: int | None
    rumdl_rule: str | None
    fixable: bool
    # sources.py
    source_file: Path
    # metadata.py
    missing_fields: list[str]
    # references.py
    duplicates: list[DuplicateReferenceDetail]
    near_duplicates: list[NearDuplicateReferenceDetail]
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


def details_for_storage(details: LintDetails) -> JSONDict:
    """Convert an in-memory payload to its persisted form.

    A path key becomes its relative POSIX string, which is what every
    ``.kompend/`` artifact stores (Antipattern 6). An absolute path is refused
    rather than persisted: the store is the boundary where a vault stops being
    relocatable, so this is where the mistake has to fail.
    """
    stored: JSONDict = {}
    for key, value in details.items():
        if key in _PATH_KEYS:
            stored[key] = _stored_path(key, value)
            continue
        stored[key] = _json_value(key, value)
    return stored


def _json_value(key: str, value: object) -> JSONValue:
    """Narrow one declared value to its JSON form, refusing what JSON cannot hold.

    A ``TypedDict`` field is read as ``object`` here: the payload is walked key
    by key, so the declared per-key type is gone by construction time. A value
    that is not JSON-shaped is a bug in the rule that wrote it, and saying so
    is the whole point of the conversion.
    """
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list | tuple):
        return [_json_value(key, item) for item in value]
    if isinstance(value, Mapping):
        return {str(name): _json_value(key, item) for name, item in value.items()}
    msg = f"details[{key!r}] cannot be persisted: {type(value).__name__} is not JSON-shaped"
    raise TypeError(msg)


def _stored_path(key: str, value: object) -> str:
    """The vault-relative POSIX string a path key persists as."""
    if not isinstance(value, Path):
        msg = f"details[{key!r}] must be a pathlib.Path, not {type(value).__name__}: {value!r}"
        raise TypeError(msg)
    if value.is_absolute():
        msg = f"details[{key!r}] must be vault-relative, not absolute: {value}"
        raise ValueError(msg)
    return value.as_posix()


def details_from_storage(details: Mapping[str, JSONValue]) -> LintDetails:
    """Convert a persisted payload back to its in-memory form.

    A path key becomes a :class:`pathlib.Path`; every other key is kept as the
    store holds it, because a row written by an older build is data, not a bug
    to reject — including a key this module no longer declares.
    """
    restored: dict[str, JSONValue | Path] = dict(details)
    for key in _PATH_KEYS:
        value = details.get(key)
        if isinstance(value, str):
            restored[key] = Path(value)
    return cast("LintDetails", restored)
