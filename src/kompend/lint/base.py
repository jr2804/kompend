"""Base types for kompend lint rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

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
        Extra structured data (e.g. list of broken link targets).
    """

    rule: str
    outcome: LintRuleOutcome
    severity: Severity = Severity.WARNING
    message: str = ""
    page_id: str = ""
    fix_applied: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class LintReport:
    """Typed wrapper around a lint report's ``results`` list.

    Single source of truth for the lint-report shape: a consumer that imports a
    narrower copy crashes in ``runner.report_to_dict``. Summary fields default to
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
    details: dict[str, Any] = field(default_factory=dict)


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

    @staticmethod
    def fix(wiki: CompiledWiki, results: list[LintResult]) -> list[LintResult]:
        """Attempt auto-fix and return updated results.

        Default implementation returns results unchanged.
        """
        return list(results)
