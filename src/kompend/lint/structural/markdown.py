"""Structural lint family: markdown style via the external ``rumdl`` CLI.

``rumdl`` is owned by an external project — this module only marshals files
to it, parses its JSON diagnostics (:class:`RumdlFinding`), and converts
them to lint results. rumdl owns its schema; unknown JSON keys are ignored
so a rumdl upgrade cannot break the lint run.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Final

from pydantic import BaseModel, TypeAdapter, ValidationError

from kompend.core.models import CompiledWiki
from kompend.core.paths import ATTACHMENTS_DIR_NAME, KOMP_DIR
from kompend.lint.base import BaseLintRule, LintResult, LintRuleOutcome, Severity
from kompend.logging import get_logger

#: Leading argv entries of the rumdl command shown in a failure warning —
#: enough to identify the invocation without echoing every path.
_CMD_PREVIEW_ARGS = 4

#: Vault-root directories a markdown style check must never descend into:
#: tool state and user attachments are neither prose nor authored pages.
_EXCLUDED_ROOT_DIRS: Final[frozenset[str]] = frozenset({KOMP_DIR, ATTACHMENTS_DIR_NAME})

logger = get_logger(__name__)


class RumdlFinding(BaseModel, extra="ignore"):
    """One diagnostic from ``rumdl check --output-format json``.

    rumdl owns this schema — unknown keys are ignored, so a rumdl upgrade
    that adds fields cannot break the lint run. Every field we read is
    optional: a diagnostic that omits one still converts.
    """

    file: str | None = None
    line: int | None = None
    column: int | None = None
    rule: str | None = None
    message: str | None = None
    severity: str | None = None
    fixable: bool = False


_RUMDL_FINDINGS = TypeAdapter(list[RumdlFinding])

#: Log-line bound for the unexpected-JSON warning below. The payload itself
#: is discarded by contract (empty list on failure) — this only keeps the
#: warning readable, it truncates no data that flows downstream.
_UNEXPECTED_JSON_PREVIEW_CHARS = 200


class RumdlRule(BaseLintRule):
    """Markdown style via ``rumdl check`` on all vault folders except sources.

    Scans every ``*.md`` file under the vault root (derived from
    *sources_dir*'s parent), excluding ``sources/``, ``.kompend/``, and
    ``attachments/``. Sources are never read or modified. Files are passed
    explicitly to bypass rumdl's ignore-file filtering — self-written notes
    that are gitignored still get checked. rumdl auto-discovers a
    ``.rumdl.toml`` in the vault.
    """

    name = "rumdl"
    description = "Markdown style issues found by rumdl (all vault folders except sources)"

    #: Explicit paths bypass rumdl's ignore-file filtering, but the Windows
    #: command line caps at ~32k chars — batch to stay well under it.
    _BATCH_SIZE = 100

    def __init__(self, rumdl_config: Path | None = None) -> None:
        self.rumdl_config = rumdl_config

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        exe = shutil.which("rumdl")
        if not exe:
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="rumdl is not installed")]
        if not sources_dir:
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="Sources directory unknown — cannot derive vault root")]
        files = self._md_files(sources_dir)
        if not files:
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="No markdown files to check")]

        findings: list[RumdlFinding] = []
        for batch in _batched(files, self._BATCH_SIZE):
            findings.extend(self._run_rumdl(exe, ["check", *self._config_args(), "--output-format", "json", *batch]))
        if not findings:
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="All markdown files pass rumdl checks")]
        return [self._to_result(f) for f in findings]

    def fix(self, wiki: CompiledWiki, results: list[LintResult], *, sources_dir: str | None = None) -> list[LintResult]:
        """Apply ``rumdl check --fix`` to the files with fixable findings."""
        exe = shutil.which("rumdl")
        if not exe:
            return results
        fixable_files = sorted(
            {r.details.get("file", "") for r in results if r.outcome == LintRuleOutcome.FAIL and r.details.get("fixable") and r.details.get("file")}
        )
        for batch in _batched(fixable_files, self._BATCH_SIZE):
            self._run_rumdl(exe, ["check", "--fix", *self._config_args(), "--output-format", "json", *batch])
        for r in results:
            if r.outcome == LintRuleOutcome.FAIL and r.details.get("fixable"):
                r.fix_applied = True
        return results

    def _config_args(self) -> list[str]:
        """``--config`` args when a custom rumdl config is set (else auto-discovery)."""
        return ["--config", str(self.rumdl_config)] if self.rumdl_config else []

    def _to_result(self, finding: RumdlFinding) -> LintResult:
        severity = Severity.ERROR if finding.severity == "error" else Severity.WARNING
        return LintResult(
            rule=self.name,
            outcome=LintRuleOutcome.FAIL,
            severity=severity,
            message=f"{finding.file}:{finding.line} {finding.rule}: {finding.message}",
            page_id="",
            details={
                "file": finding.file or "",
                "line": finding.line,
                "column": finding.column,
                "rumdl_rule": finding.rule,
                "fixable": finding.fixable,
            },
        )

    @staticmethod
    def _md_files(sources_dir: str) -> list[str]:
        """All ``*.md`` files under the vault root, excluding sources/, .kompend/ and attachments/."""
        base = Path(sources_dir).parent
        sources_path = Path(sources_dir)
        files: list[str] = []
        for path in base.rglob("*.md"):
            rel = path.relative_to(base)
            if sources_path in path.parents:
                continue
            if any(part in _EXCLUDED_ROOT_DIRS for part in rel.parts[:-1]):
                continue
            files.append(str(path))
        return sorted(files)

    @staticmethod
    def _run_rumdl(exe: str, cmd: list[str]) -> list[RumdlFinding]:
        """Run rumdl and return its JSON diagnostics (empty list on failure)."""
        try:
            result = subprocess.run(  # noqa: S603 — internally constructed cmd list
                [exe, *cmd],
                capture_output=True,
                encoding="utf-8",
                timeout=120,  # nocheck: rumdl subprocess timeout, unrelated to any model request timeout
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            logger.warning("rumdl invocation failed: %s", cmd[:_CMD_PREVIEW_ARGS])
            return []
        try:
            return _RUMDL_FINDINGS.validate_json(result.stdout or "[]")
        except ValidationError:
            logger.warning("rumdl emitted unexpected JSON: %s", (result.stdout or "")[:_UNEXPECTED_JSON_PREVIEW_CHARS])
            return []


def _batched(items: list[str], size: int) -> list[list[str]]:
    """Split *items* into consecutive batches of at most *size*."""
    return [items[i : i + size] for i in range(0, len(items), size)]
