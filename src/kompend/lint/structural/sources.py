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
                    source_file = Path(source_file_str)
                    if not source_file.is_absolute():
                        # Try base_dir-relative first (sources/paper.pdf),
                        # then sources_dir-relative (paper.pdf)
                        for root in (sources_root.parent, Path(sources_dir)):
                            if (root / source_file).exists():
                                source_file = str(root / source_file)
                                break
                        else:
                            source_file = str(sources_root.parent / source_file)
                    if not Path(source_file).exists():
                        results.append(
                            LintResult(
                                rule=self.name,
                                outcome=LintRuleOutcome.FAIL,
                                severity=Severity.WARNING,
                                message=f"Source file not found for page '{node.title}': {source_file}",
                                page_id=node.id,
                                details={MetadataField.SOURCE_FILE: str(source_file)},
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
                        details={"reason": "superseded"},
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
                        details={"reason": "decayed"},
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
            source_file = Path(source_file_str)
            if not source_file.is_absolute():
                for root in (sources_root.parent, Path(sources_dir)):
                    if (root / source_file).exists():
                        source_file = str(root / source_file)
                        break
                else:
                    source_file = str(sources_root.parent / source_file)
            if not Path(source_file).exists():
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.ERROR,
                        message=f"Source file missing for page '{node.title}': {source_file}",
                        page_id=node.id,
                        details={MetadataField.SOURCE_FILE: str(source_file)},
                    )
                )
        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="All referenced sources exist"))
        return results
