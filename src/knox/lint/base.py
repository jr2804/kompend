"""Base types for KNOX lint rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class Severity(StrEnum):
    """Lint finding severity."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class LintRuleOutcome(StrEnum):
    """Result of applying a single lint rule to a page."""

    PASS = "pass"  # noqa: S105
    FAIL = "fail"  # noqa: S105
    SKIP = "skip"  # noqa: S105


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


class LintRule(Protocol):
    """Protocol that every lint rule must satisfy.

    Implementations are plain callables (functions or classes with
    ``__call__``). Each receives the full :class:`CompiledWiki` and
    returns a list of :class:`LintResult`.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def check(self, wiki: Any, *, sources_dir: str | None = None) -> list[LintResult]: ...

    def fix(self, wiki: Any, results: list[LintResult]) -> list[LintResult]:
        """Attempt auto-fix and return updated results.

        Default implementation marks all results as ``fix_applied=False``.
        """
        ...
