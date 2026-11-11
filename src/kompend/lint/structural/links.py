"""Structural lint family: link targets inside page content.

Owns the wikilink vocabulary (how a ``[[target]]`` is extracted and how a
target is resolved against the compiled pages) plus the two rules that read
link targets: ``broken-wikilinks`` (a target that resolves to nothing) and
``kompend-reference`` (a target that points back into the ``.kompend/``
state dir, which breaks portability).
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from kompend.compilation.concept_keys import normalize_concept_key
from kompend.core.models import KIND_FOLDERS, CompiledWiki, MetadataField
from kompend.core.models.metadata import meta_dict_list, meta_str
from kompend.core.paths import KOMP_DIR
from kompend.core.text.matching import best_match
from kompend.lint.base import BaseLintRule, LintDetails, LintResult, LintRuleOutcome, Severity
from kompend.lint.structural.images import _IMAGE_EXTENSIONS

#: Markdown link or embed target pointing into the ``.kompend/`` state dir
#: (``.kompend/…``, ``./.kompend/…``, ``../.kompend/…``, ``/.kompend/…``).
_KOMP_PATH_RE = re.compile(rf"\]\(((?:\.\./)*(?:\./)?/?{re.escape(KOMP_DIR)}/[^)\s]+)\)")

#: Threshold above which a broken wikilink is treated as a SHORT-FORM alias
#: (severity WARNING, the suggested alias is attached) instead of a hard
#: failure (severity ERROR). Below the default best_match threshold (0.85)
#: because most "broken" wikilinks are intentional short-form aliases emitted by
#: small models -- surface these as guidance, not as a blocker.
_WIKILINK_ALIAS_WARN_THRESHOLD: float = 60.0

#: Lower detection bound — anything below this is treated as a genuine
#: missing topic (no alias suggestion, ERROR severity). Avoids noisy
#: ``alias_of=foo (similarity=9)`` markers when the target has no real
#: relationship to any page.
_WIKILINK_ALIAS_MIN_DETECT: float = 25.0

_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


class BrokenWikilinksRule(BaseLintRule):
    """Detect ``[[wikilinks]]`` that point to non-existent pages."""

    name = "broken-wikilinks"
    description = "Wikilinks pointing to pages that do not exist"

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        # Lookup index covers: page title, page slug, and acronym-stripped
        # short alias (e.g. ``[[P.863]]`` resolves to the page titled
        # ``POLQA (Perceptual Objective Listening Quality Analysis)``).
        alias_index = _node_lookup_index(wiki)
        results: list[LintResult] = []
        for node in wiki.nodes:
            for target_title in _extract_wikilinks(node.content):
                if Path(target_title).suffix.lower() in _IMAGE_EXTENSIONS:
                    # Image embeds are handled by ImageEmbedRule — not page links.
                    continue
                # Source-file citations (sources/foo.pdf) are provenance, not page links.
                if target_title.startswith("sources/"):
                    continue
                # Alias syntax [[Page Title|display text]]: resolve the page ref.
                page_ref = target_title.split("|", 1)[0].strip()
                if page_ref.lower() not in alias_index:
                    target_norm = normalize_concept_key(page_ref)
                    alias_of: str | None = None
                    similarity_score: float = 0.0
                    if target_norm:
                        best = best_match(target_norm, alias_index, threshold=0.0)
                        if best and best[1] >= _WIKILINK_ALIAS_MIN_DETECT:
                            alias_of = alias_index[best[0]]
                            similarity_score = best[1]
                    details: LintDetails = {"target": target_title}
                    if alias_of:
                        details["alias_of"] = alias_of
                        details["alias_similarity"] = round(similarity_score, 1)
                    severity = Severity.WARNING if similarity_score >= _WIKILINK_ALIAS_WARN_THRESHOLD else Severity.ERROR
                    message = (
                        f"Wikilink [[{target_title}]] in page '{node.title}' resolves to '{alias_of}' via short-form alias (similarity={similarity_score:.0f})"
                        if alias_of
                        else f"Broken wikilink [[{target_title}]] in page '{node.title}'"
                    )
                    results.append(
                        LintResult(
                            rule=self.name,
                            outcome=LintRuleOutcome.FAIL,
                            severity=severity,
                            message=message,
                            page_id=node.id,
                            details=details,
                        )
                    )
        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="All wikilinks resolve"))
        return results

    def fix(self, wiki: CompiledWiki, results: list[LintResult], *, sources_dir: str | None = None) -> list[LintResult]:  # noqa: PLR6301 — the fix repairs links in page content; it reads no rule state
        """Repair broken wikilinks in page content.

        Alias-resolvable links (the LLM used a short form of an existing
        page's title) are rewritten to real markdown links pointing at the
        resolved page — matching what the renderer's ``convert_wikilinks``
        produces. Unresolvable links are stripped to their text.
        """
        node_by_id = {n.id: n for n in wiki.nodes}
        page_edits: dict[str, list[tuple[str, str | None]]] = {}
        for r in results:
            if r.outcome != LintRuleOutcome.FAIL or not r.details.get("target"):
                continue
            page_edits.setdefault(r.page_id, []).append((r.details["target"], r.details.get("alias_of")))

        fixed = list(results)
        for node in wiki.nodes:
            edits = page_edits.get(node.id)
            if not edits:
                continue
            for target, alias_of in edits:
                inner = target
                if "|" in inner:
                    page_ref, alias_text = inner.split("|", 1)
                else:
                    alias_text = inner
                replacement: str
                target_node = node_by_id.get(alias_of or "")
                if alias_of and target_node is not None:
                    # Rewrite to the resolved page link (same path the
                    # renderer's convert_wikilinks produces).
                    folder = KIND_FOLDERS.get(target_node.node_type.value, "concepts")
                    replacement = f"[{alias_text.strip()}]({folder}/{alias_of}.md)"
                else:
                    # Genuinely missing target — keep the text, drop the brackets.
                    replacement = alias_text.strip()
                node.content = node.content.replace(f"[[{target}]]", replacement)
            for r in fixed:
                if r.page_id == node.id and r.outcome == LintRuleOutcome.FAIL:
                    r.fix_applied = True
        return fixed


class KompendReferenceRule(BaseLintRule):
    """Detect page links/embeds that resolve into ``.kompend/``.

    Markdown-only invariant: ``.kompend/`` is derived state and may be absent
    from a shared vault — a page referencing it breaks portability.
    """

    name = "kompend-reference"
    description = "Links or embeds resolving into the .kompend/ state directory"

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []
        for node in wiki.nodes:
            content = node.content or ""
            for target in _KOMP_PATH_RE.findall(content):
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.ERROR,
                        message=(f"Page '{node.title}' references '{target}' inside {KOMP_DIR}/ — copy the file into attachments/ and link relatively"),
                        page_id=node.id,
                        details={"target": target},
                    )
                )
            for img in meta_dict_list(node.metadata.get(MetadataField.IMAGES)):
                path = meta_str(img.get("path"))
                if path and KOMP_DIR in PurePosixPath(path).parts:
                    results.append(
                        LintResult(
                            rule=self.name,
                            outcome=LintRuleOutcome.FAIL,
                            severity=Severity.ERROR,
                            message=(
                                f"Page '{node.title}' embeds image '{path}' from inside {KOMP_DIR}/ — "
                                "the writer must copy it into attachments/ (portablize_images)"
                            ),
                            page_id=node.id,
                            details={"target": path},
                        )
                    )
        if not results:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message=f"No page references {KOMP_DIR}/"))
        return results


#: The rules this family contributes to the structural registry.
LINK_RULES: tuple[BaseLintRule, ...] = (BrokenWikilinksRule(), KompendReferenceRule())


def _extract_wikilinks(content: str) -> list[str]:
    """Return all ``[[target]]`` strings from *content*."""
    return _WIKILINK_RE.findall(content)


def _node_lookup_index(wiki: CompiledWiki) -> dict[str, str]:
    """Build a wikilink lookup that also accepts slugs and acronym aliases.

    The returned ``alias_index`` maps any of ``node.title.lower()``,
    ``node.id``, or the acronym-stripped title (e.g. ``"polqa"`` extracted
    from ``"POLQA (Perceptual Objective Listening Quality Analysis)"``)
    to ``node.id``.

    Also indexes the filesystem-path forms ``concepts/<id>`` and
    ``entities/<id>`` (entities get a sub-folder per type, e.g.
    ``entities/standard/foo``), so a wikilink emitted by the model that
    includes the subdirectory prefix (``[[concepts/foo]]``) still
    resolves.

    The acronym extraction lives in :mod:`kompend.compilation.concept_keys`
    so the same normalization is used for concept-merge and wikilink
    resolution — one source of truth.
    """
    alias_index: dict[str, str] = {}
    for n in wiki.nodes:
        # Lower-case first entry wins; existing index already preferred title
        alias_index.setdefault(n.title.lower(), n.id)
        alias_index.setdefault(n.id, n.id)
        stripped = normalize_concept_key(n.title)
        if stripped:
            alias_index.setdefault(stripped, n.id)
        # Also index the bare-acronym form so wikilinks like ``[[MOS-LQO]]``
        # resolve to the page titled ``MOS-LQO (Mean Opinion Score - ...)``.
        # ``normalize_concept_key`` strips the parenthetical, so ``stripped``
        # already gives us this form for hyphen-having prefixes.
        title_before_paren = n.title.split("(", 1)[0].strip().lower()
        if title_before_paren and title_before_paren != n.title.lower():
            alias_index.setdefault(title_before_paren, n.id)
        # Index filesystem-path forms so wikilinks with subdirectory
        # prefix (``[[concepts/foo]]``) resolve. The vault writer places
        # nodes under KIND_FOLDERS[<node_type>]/<id>.md; entities get an
        # extra <type>/ level per _folder_for_kind. We can't know the
        # entity_subtype here without inspecting metadata, so we register
        # both forms (``entities/<id>`` for unknown-type lookup and let
        # nearest_alias pick the closer one).
        folder = KIND_FOLDERS.get(n.node_type.value, "concepts")
        alias_index.setdefault(f"{folder}/{n.id}", n.id)
    return alias_index


def _node_titles_index(wiki: CompiledWiki) -> dict[str, str]:
    """Map ``node.title.lower()`` → ``node.id`` (legacy alias for callers)."""
    return {n.title.lower(): n.id for n in wiki.nodes}
