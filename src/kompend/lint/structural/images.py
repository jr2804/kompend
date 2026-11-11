"""Structural lint family: image embeds.

Owns the image vocabulary — which suffixes count as an image, and what a
missing ``!`` embed prefix looks like — plus the ``image-embed`` rule that
flags ``[[img.png]]`` written as a plain link instead of an embed.
"""

from __future__ import annotations

import re
from pathlib import Path

from kompend.core.models import CompiledWiki
from kompend.lint.base import BaseLintRule, LintResult, LintRuleOutcome, Severity

#: Image file extensions — ``[[img.png]]`` is an embed-syntax error, not a
#: broken wikilink; :class:`ImageEmbedRule` handles those.
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".bmp",
        ".tiff",
    }
)

#: ``[[img.png]]`` without the ``!`` embed prefix.
_IMAGE_EMBED_RE = re.compile(
    r"(?<!\!)\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg|webp|bmp|tiff))\]\]",
    re.IGNORECASE,
)


class ImageEmbedRule(BaseLintRule):
    """Detect image wikilinks written without the ``!`` embed prefix."""

    name = "image-embed"
    description = "Image links that should use ![[...]] embed syntax"

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []
        for node in wiki.nodes:
            for target in _IMAGE_EMBED_RE.findall(node.content or ""):
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.WARNING,
                        message=f"Image link [[{target}]] in page '{node.title}' should be ![[{target}]]",
                        page_id=node.id,
                        details={"target": target},
                    )
                )
        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="All image links use embed syntax"))
        return results

    def fix(self, wiki: CompiledWiki, results: list[LintResult], *, sources_dir: Path | None = None) -> list[LintResult]:  # noqa: PLR6301
        """Rewrite ``[[img.png]]`` → ``![[img.png]]`` in page content."""
        fixed = list(results)
        for node in wiki.nodes:
            content = node.content or ""
            if not _IMAGE_EMBED_RE.search(content):
                continue
            node.content = _IMAGE_EMBED_RE.sub(lambda m: f"![[{m.group(1)}]]", content)
            for r in fixed:
                if r.page_id == node.id and r.outcome == LintRuleOutcome.FAIL:
                    r.fix_applied = True
        return fixed


#: The rules this family contributes to the structural registry.
IMAGE_RULES: tuple[BaseLintRule, ...] = (ImageEmbedRule(),)
