"""Structural lint family: page metadata.

Owns the ``missing-frontmatter`` rule — the check that a written page
carries the frontmatter fields the wiki format requires.
"""

from __future__ import annotations

from pathlib import Path

from kompend.core.models import CompiledWiki, MetadataField
from kompend.lint.base import BaseLintRule, LintResult, LintRuleOutcome, Severity


class MissingFrontmatterRule(BaseLintRule):
    """Detect pages missing required frontmatter fields."""

    name = "missing-frontmatter"
    description = "Pages missing required frontmatter fields"

    REQUIRED_FIELDS: tuple[str, ...] = ("title",)

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []
        for node in wiki.nodes:
            fm = node.metadata.get(MetadataField.FRONTMATTER, {})
            if not isinstance(fm, dict):
                fm = {}
            missing = [f for f in self.REQUIRED_FIELDS if f not in fm]
            if missing:
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.WARNING,
                        message=f"Page '{node.title}' missing frontmatter: {', '.join(missing)}",
                        page_id=node.id,
                        details={"missing_fields": missing},
                    )
                )
        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="All pages have frontmatter"))
        return results

    def fix(self, wiki: CompiledWiki, results: list[LintResult], *, sources_dir: Path | None = None) -> list[LintResult]:  # noqa: PLR6301
        """Add minimal frontmatter to pages that lack it."""
        fixed = list(results)
        for r in fixed:
            if r.outcome != LintRuleOutcome.FAIL:
                continue
            node = next((n for n in wiki.nodes if n.id == r.page_id), None)
            if node is None:
                continue
            fm = node.metadata.get(MetadataField.FRONTMATTER, {})
            if not isinstance(fm, dict):
                fm = {}
            for field_name in r.details.get("missing_fields", []):
                if field_name == "title":
                    fm["title"] = node.title
            node.metadata[MetadataField.FRONTMATTER] = fm
            r.fix_applied = True
        return fixed


#: The rules this family contributes to the structural registry.
METADATA_RULES: tuple[BaseLintRule, ...] = (MissingFrontmatterRule(),)
