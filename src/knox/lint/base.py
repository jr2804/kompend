"""Base types for KNOX lint rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from knox.core.models import CompiledWiki


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
class LintReportResult:
    """A single lint finding, as stored in ``lint_history.jsonl``.

    Used by the remediation layer, which reads the serialized report dict
    rather than in-memory :class:`LintResult` objects.
    """

    rule: str
    outcome: str
    details: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    page_id: str = ""


@dataclass
class LintReport:
    """Typed wrapper around a lint report's ``results`` list.

    Replaces the previous ``report_dict: dict[str, Any]`` parameter with a
    named, strictly-typed model.
    """

    results: list[LintReportResult]


@dataclass
class RemediationAction:
    """A single remediation action applied, skipped, or recommended."""

    category: str
    description: str
    status: RemediationStatus
    details: dict[str, Any] = field(default_factory=dict)


class BaseLintRule(ABC):
    """Abstract base class that every lint rule must implement.

    Subclasses must define :attr:`name`, :attr:`description` and
    :meth:`check`. The :meth:`fix` method is optional — the default
    implementation returns results unchanged (no auto-fix).
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]: ...

    @staticmethod
    def fix(wiki: CompiledWiki, results: list[LintResult]) -> list[LintResult]:
        """Attempt auto-fix and return updated results.

        Default implementation returns results unchanged.
        """
        return list(results)
