"""Base types for kompend lint rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, TypedDict

from kompend.core.models import CompiledWiki


class LintRuleOutcome(StrEnum):
    """Result of applying a single lint rule to a page."""

    PASS = "pass"  # noqa: S105
    FAIL = "fail"  # noqa: S105
    SKIP = "skip"  # noqa: S105


class RemediationStatus(StrEnum):
    APPLIED = "applied"
    SKIPPED = "skipped"
    RECOMMENDED = "recommended"


class Severity(StrEnum):
    """Lint finding severity."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


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


@dataclass
class LintResult:
    """A single finding from a lint rule.

    Attributes:
    ----------
    rule:
        Machine-readable rule name (e.g. ``"broken-wikilinks"``).
    outcome:
        Whether the check passed, failed, or was skipped.
    severity:
        How serious a failure is.
    message:
        Human-readable description of the finding.
    page_id:
        Wiki node ``id`` that the finding applies to (empty for wiki-level).
    fix_applied:
        Whether ``--fix`` resolved this finding automatically.
    details:
        Extra structured data, shaped by :class:`LintDetails` (e.g. the
        broken link targets).
    """

    rule: str
    outcome: LintRuleOutcome
    severity: Severity = Severity.WARNING
    message: str = ""
    page_id: str = ""
    fix_applied: bool = False
    details: LintDetails = field(default_factory=LintDetails)


@dataclass
class LintReport:
    """Typed wrapper around a lint report's ``results`` list.

    Single source of truth for the lint-report shape: a consumer that imports a
    narrower copy crashes in ``runner.report_to_payload``. Summary fields default to
    0/0.0; callers that can compute them (runner, flow) fill them in.

    ``results`` holds :class:`LintResult` — the same finding type the rules
    return. History entries are parsed back into ``LintResult`` by
    :func:`kompend.lint.history.report_from_dict`; there is no second, narrower
    result type, so a reader never has to guess which of the two it holds.
    """

    results: list[LintResult]
    total_checks: int = 0
    failures: int = 0
    passes: int = 0
    skips: int = 0
    health_score: float = 0.0


@dataclass
class RemediationAction:
    """A single remediation action applied, skipped, or recommended."""

    category: str
    description: str
    status: RemediationStatus
    details: LintDetails = field(default_factory=LintDetails)


class BaseLintRule(ABC):
    """Abstract base class that every lint rule must implement.

    Subclasses must set :attr:`name`, :attr:`description` (class attributes)
    and implement :meth:`check`. The :meth:`fix` method is optional — the
    default implementation returns results unchanged (no auto-fix).
    """

    name: ClassVar[str]
    description: ClassVar[str]

    #: When True, :meth:`fix` deletes or irreversibly modifies files. The
    #: lint runner skips destructive fixes unless the caller explicitly
    #: passes ``fix_destructive=True`` — the post-compile lint and default
    #: ``kompend lint check`` do NOT, so destructive rules only report there.
    destructive: ClassVar[bool] = False

    @abstractmethod
    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]: ...

    def fix(  # noqa: PLR6301 — the base method every rule overrides
        self,
        wiki: CompiledWiki,
        results: list[LintResult],
        *,
        sources_dir: str | None = None,
    ) -> list[LintResult]:
        """Attempt auto-fix and return updated results.

        The runner passes only the failing findings and expects one updated
        finding per input, in the same order; it splices them back into the
        rule's full result list. Returning fewer findings than you were given
        loses a finding — the runner logs that as an error and keeps it.

        *sources_dir* is the directory the scan looked at, as the runner
        received it. A rule that acts on files takes it from here, never from a
        directory remembered by an earlier call.

        Default implementation returns results unchanged.
        """
        return list(results)
