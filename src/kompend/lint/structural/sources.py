"""Structural lint family: the source files behind the pages.

Owns the two rules that look at the vault's ``sources/`` directory instead
of the compiled wiki: ``stale-page`` (the source file behind a page is gone
or the page was superseded in the memory store) and ``source-not-found``
(the page cites a source file that no longer exists on disk).
"""

from __future__ import annotations

from pathlib import Path

from kompend.core.models import CompiledWiki, MetadataField
from kompend.core.models.metadata import meta_str
from kompend.lint.base import BaseLintRule, LintResult, LintRuleOutcome, Severity
from kompend.lint.details import LintField


class StalePageRule(BaseLintRule):
    """Detect pages whose source files have been modified since last compile.

    Also checks the memory lifecycle store for superseded and decayed (stale)
    nodes when memory metadata is present in ``wiki.metadata``.
    """

    name = "stale-page"
    description = "Pages whose source files changed since last compile"

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []

        # ── Source file existence check ──
        if sources_dir is not None:
            sources_root = Path(sources_dir)
            if sources_root.exists():
                for node in wiki.nodes:
                    source_file_str = meta_str(node.metadata.get(MetadataField.SOURCE_FILE))
                    if not source_file_str:
                        continue
                    source_file = _resolve_source_file(source_file_str, sources_root)
                    if not (sources_root.parent / source_file).exists():
                        results.append(
                            LintResult(
                                rule=self.name,
                                outcome=LintRuleOutcome.FAIL,
                                severity=Severity.WARNING,
                                message=f"Source file not found for page '{node.title}': {source_file}",
                                page_id=node.id,
                                details={LintField.SOURCE_FILE: source_file},
                            )
                        )

        # ── Memory lifecycle checks ──
        memory_meta = wiki.metadata.get("memory", {})
        superseded_ids = set(memory_meta.get("superseded", []))
        stale_ids = set(memory_meta.get("stale", []))

        for node in wiki.nodes:
            if node.id in superseded_ids:
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.WARNING,
                        message=f"Page '{node.title}' has been superseded by a newer node",
                        page_id=node.id,
                        details={LintField.REASON: "superseded"},
                    )
                )
            elif node.id in stale_ids:
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.INFO,
                        message=f"Page '{node.title}' has decayed below retention threshold",
                        page_id=node.id,
                        details={LintField.REASON: "decayed"},
                    )
                )

        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="All source files intact and memory fresh"))
        return results


class SourceNotFoundRule(BaseLintRule):
    """Detect pages that reference source files not present on disk."""

    name = "source-not-found"
    description = "Pages referencing source files that don't exist"

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []
        if sources_dir is None:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No sources directory provided"))
            return results

        sources_root = Path(sources_dir)
        for node in wiki.nodes:
            source_chunks = node.source_chunks or []
            if not source_chunks:
                continue
            # source_chunks are chunk IDs, not file paths. Check if
            # metadata references source_file.
            source_file_str = meta_str(node.metadata.get(MetadataField.SOURCE_FILE))
            if not source_file_str:
                continue
            source_file = _resolve_source_file(source_file_str, sources_root)
            if not (sources_root.parent / source_file).exists():
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.ERROR,
                        message=f"Source file missing for page '{node.title}': {source_file}",
                        page_id=node.id,
                        details={LintField.SOURCE_FILE: source_file},
                    )
                )
        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="All referenced sources exist"))
        return results


#: The rules this family contributes to the structural registry.
SOURCE_RULES: tuple[BaseLintRule, ...] = (StalePageRule(), SourceNotFoundRule())


def _resolve_source_file(source_file_str: str, sources_root: Path) -> Path:
    """The source file a page names, as a path relative to the vault.

    Tries the base-dir-relative form first (``sources/paper.pdf``), then the
    sources-dir-relative form (``paper.pdf``), and returns the form that exists.
    When neither exists it returns the path the page claims, unchanged — a
    finding that the vault does not resolve must still print the file the page
    names, and it must stay relative so the persisted payload does not carry a
    machine-specific absolute path.
    """
    vault_root = sources_root.parent
    claimed = Path(source_file_str)
    if claimed.is_absolute():
        # A vault-relative payload is the contract; an absolute path inside the
        # vault is relativized, one outside it is left alone so the persistence
        # boundary refuses it loudly instead of writing a machine path.
        return claimed.relative_to(vault_root) if claimed.is_relative_to(vault_root) else claimed
    for root in (vault_root, sources_root):
        if (root / claimed).exists():
            return (root / claimed).relative_to(vault_root)
    return claimed
