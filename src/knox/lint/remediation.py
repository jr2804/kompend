"""Remediate lint findings — auto-fix, LLM-generate, or recommend.

Three tiers of remediation:

**Tier 1 — Auto-fixable (no LLM):** Deterministic fixes applied directly to
wiki content. Currently handles image embed syntax errors (``[[x.png]]`` →
``![[x.png]]``).

**Tier 2 — LLM-remediable:** Generates new content using cached extractions
as source material. Currently handles coverage gaps by generating concept
pages from phase-1 extraction data.

**Tier 3 — Recommendations:** Issues that need new sources or manual
intervention. Reported but not actioned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from slugify import slugify

from knox.compilation.two_phase import Extraction, ExtractionSet
from knox.core.models import CompiledWiki, WikiNode
from knox.core.template_loader import get_template_loader
from knox.lint.gap_analysis import analyze_gaps
from knox.llm.roles import LLMRole
from knox.logging import get_logger
from knox.organize.categorize import load_all_extractions
from knox.output.renderer import render_frontmatter

logger = get_logger(__name__)

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

_TIER3_CATEGORIES: frozenset[str] = frozenset(
    {
        "Missing topic",
        "Unresolved contradiction",
        "Empty page",
        "Poor content quality",
        "Isolated topic",
        "Coverage gap",
    }
)


class RemediationStatus(StrEnum):
    APPLIED = "applied"
    SKIPPED = "skipped"
    RECOMMENDED = "recommended"


@dataclass
class RemediationAction:
    """A single remediation action applied, skipped, or recommended."""

    tier: int  # 1, 2, or 3
    category: str
    description: str
    status: RemediationStatus
    details: dict[str, Any] = field(default_factory=dict)


def remediate(
    wiki: CompiledWiki,
    report_dict: dict[str, Any],
    *,
    base_dir: Path | None = None,
    sources_dir: Path | None = None,
    output_dir: Path | None = None,
    llm_client: Any | None = None,
    dry_run: bool = False,
) -> list[RemediationAction]:
    """Remediate lint findings from a report dict.

    Args:
        wiki: The compiled wiki to modify in-place (tier 1/2).
        report_dict: The ``report`` sub-dict from ``lint_history.jsonl``.
        base_dir: Project root. Required for tier 2.
        sources_dir: Sources directory. Required for tier 2.
        output_dir: Wiki output directory. Tier-2 generated concept pages are
            written here as ``.md`` files so they are visible to the user and
            survive a re-organize. When ``None``, pages only update the cache.
        llm_client: LLM client for tier 2. If ``None``, tier 2 is skipped.
        dry_run: If ``True``, report what would be done without modifying.

    Returns a list of :class:`RemediationAction` sorted by tier.
    """
    results: list[dict[str, Any]] = report_dict.get("results", [])
    actions: list[RemediationAction] = []

    # ── Tier 1: Auto-fixable ──
    actions.extend(_fix_image_wikilinks(wiki, results, dry_run=dry_run))

    # ── Tier 2: LLM-remediable ──
    handled_concepts: set[str] = set()
    if llm_client is not None and base_dir is not None and sources_dir is not None:
        tier2_actions = _remediate_coverage_gaps(
            wiki,
            results,
            llm_client=llm_client,
            base_dir=base_dir,
            sources_dir=sources_dir,
            output_dir=output_dir,
            dry_run=dry_run,
        )
        actions.extend(tier2_actions)
        # Track which concepts tier 2 looked at (any status) to exclude from tier 3
        for a in tier2_actions:
            concept = a.details.get("concept")
            if concept:
                handled_concepts.add(concept.lower().strip())
            matched = a.details.get("matched_concept")
            if matched:
                handled_concepts.add(matched.lower().strip())

    # ── Tier 3: Recommendations ──
    # Collect targets already handled by tier 1 to avoid duplicate recommendations
    handled_targets: set[str] = set()
    for r in results:
        if r.get("rule") != "broken-wikilinks" or r.get("outcome") != "fail":
            continue
        target = r.get("details", {}).get("target", "")
        if target and Path(target).suffix.lower() in _IMAGE_EXTENSIONS:
            handled_targets.add(target)

    for rec in analyze_gaps(report_dict):
        if rec.category not in _TIER3_CATEGORIES:
            continue
        # Skip image-extension false positives already handled in tier 1
        finding_lower = rec.finding.lower()
        if any(ext in finding_lower for ext in _IMAGE_EXTENSIONS):
            continue
        # Skip coverage gaps already handled by tier 2
        if rec.category == "Coverage gap" and any(hc in finding_lower for hc in handled_concepts):
            continue
        actions.append(
            RemediationAction(
                tier=3,
                category=rec.category,
                description=rec.finding,
                status="recommended",
                details={"action": rec.action, "priority": rec.priority},
            )
        )

    actions.sort(key=lambda a: a.tier)
    return actions


# ── Tier 1: Auto-fixable ───────────────────────────────────────────────


def _fix_image_wikilinks(
    wiki: CompiledWiki,
    results: list[dict[str, Any]],
    *,
    dry_run: bool = False,
) -> list[RemediationAction]:
    """Fix ``[[image.png]]`` → ``![[image.png]]`` (wrong embed syntax)."""
    image_targets: dict[str, list[str]] = {}
    for r in results:
        if r.get("rule") != "broken-wikilinks" or r.get("outcome") != "fail":
            continue
        target = r.get("details", {}).get("target", "")
        if target and Path(target).suffix.lower() in _IMAGE_EXTENSIONS:
            image_targets.setdefault(r.get("page_id", ""), []).append(target)

    if not image_targets:
        return []

    actions: list[RemediationAction] = []
    for node in wiki.nodes:
        targets = image_targets.get(node.id)
        if not targets:
            continue
        fixed = 0
        for target in targets:
            old_syntax = f"[[{target}]]"
            if old_syntax in (node.content or ""):
                if not dry_run:
                    node.content = node.content.replace(old_syntax, f"![[{target}]]")
                fixed += 1
        if fixed:
            actions.append(
                RemediationAction(
                    tier=1,
                    category="Image embed syntax",
                    description=f"Fixed {fixed} image wikilink(s) in '{node.title}'",
                    status="applied" if not dry_run else "skipped",
                    details={"targets": targets},
                )
            )
    return actions


# ── Tier 2: LLM-remediable ─────────────────────────────────────────────


def _remediate_coverage_gaps(
    wiki: CompiledWiki,
    results: list[dict[str, Any]],
    *,
    llm_client: Any,
    base_dir: Path,
    sources_dir: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> list[RemediationAction]:
    """Generate concept pages for coverage gaps using cached extractions."""
    gap_concepts: list[str] = []
    for r in results:
        if r.get("rule") != "coverage-gap" or r.get("outcome") != "fail":
            continue
        concept = r.get("details", {}).get("concept") or r.get("message", "")
        if concept:
            gap_concepts.append(concept)

    if not gap_concepts:
        return []

    extraction_map = load_all_extractions(base_dir, sources_dir)
    if not extraction_map:
        logger.info("No cached extractions found — skipping coverage gap remediation")
        return []

    concept_lookup = _build_concept_lookup(extraction_map)

    # Existence check must ignore prior lint-remediation "ghost" nodes
    # (generated nodes never written to .md) so they get regenerated rather
    # than silently skipped. Real (compiled) nodes always count; .md files on
    # disk count too when output_dir is provided.
    real_nodes = [n for n in wiki.nodes if n.metadata.get("source") != "lint-remediation"]
    existing_ids = {n.id for n in real_nodes}
    existing_titles = {n.title.lower().strip() for n in real_nodes}
    if output_dir is not None and output_dir.is_dir():
        existing_ids |= {f.stem for f in output_dir.glob("*.md")}
    node_index = {n.id: i for i, n in enumerate(wiki.nodes)}
    actions: list[RemediationAction] = []

    for gap in gap_concepts:
        matched = _fuzzy_match_concept(gap.lower().strip(), concept_lookup)
        if not matched:
            actions.append(
                RemediationAction(
                    tier=3,
                    category="Coverage gap (no extraction match)",
                    description=f"'{gap}' — not found in cached extractions",
                    status="recommended",
                    details={"action": "Add a source document covering this concept.", "concept": gap},
                )
            )
            continue

        concept_name = matched.get("name", gap)
        page_id = slugify(concept_name)
        if page_id in existing_ids or concept_name.lower().strip() in existing_titles:
            continue

        if dry_run:
            actions.append(
                RemediationAction(
                    tier=2,
                    category="Coverage gap",
                    description=f"Would generate concept page for '{concept_name}'",
                    status="skipped",
                    details={"sources": matched.get("sources", []), "concept": gap, "matched_concept": concept_name},
                )
            )
            continue

        content = _generate_concept_page(llm_client, matched)
        if content is None:
            actions.append(
                RemediationAction(
                    tier=2,
                    category="Coverage gap (LLM failed)",
                    description=f"Failed to generate page for '{concept_name}'",
                    status="skipped",
                    details={"concept": concept_name},
                )
            )
            continue

        node = WikiNode(
            id=page_id,
            title=concept_name,
            content=content,
            node_type="concept",
            metadata={
                "source": "lint-remediation",
                "generated_from": matched.get("sources", []),
            },
        )
        # Replace a same-id ghost node in place; otherwise append.
        if page_id in node_index:
            wiki.nodes[node_index[page_id]] = node
        else:
            node_index[page_id] = len(wiki.nodes)
            wiki.nodes.append(node)
        existing_ids.add(page_id)
        existing_titles.add(concept_name.lower().strip())
        _write_concept_page(output_dir, page_id, concept_name, content)

        actions.append(
            RemediationAction(
                tier=2,
                category="Coverage gap",
                description=f"Generated concept page '{concept_name}' from {len(matched.get('sources', []))} source(s)",
                status="applied",
                details={"sources": matched.get("sources", []), "concept": gap, "matched_concept": concept_name},
            )
        )

    return actions


def _build_concept_lookup(
    extraction_map: dict[str, Extraction],
) -> dict[str, dict[str, Any]]:
    """Build a normalized concept lookup from cached extractions.

    Indexes both LLM-extracted concepts and entities. Standards and references
    (e.g. "ITU-T P.57") are typically extracted as entities, not concepts;
    entity ``context`` serves as the definition. Concept data takes priority.
    Each bucket also carries per-source document summaries (``Extraction.summary``)
    scoped to the sources that actually mention the concept — richer seed
    material for page generation than definitions alone.
    """
    ext_set = ExtractionSet(extractions=list(extraction_map.values()))
    concept_lookup: dict[str, dict[str, Any]] = {}
    for cp in ext_set.concept_pages():
        name = cp.get("name", "")
        if name:
            concept_lookup[name.lower().strip()] = cp

    for ext in extraction_map.values():
        for ent in ext.entities:
            name = ent.get("name", "").strip()
            if not name:
                continue
            key = name.lower().strip()
            bucket = concept_lookup.setdefault(key, {"name": name, "sources": [], "definitions": []})
            if ext.source not in bucket["sources"]:
                bucket["sources"].append(ext.source)
            context = ent.get("context", "")
            if context and context not in bucket["definitions"]:
                bucket["definitions"].append(context)

    summary_by_source: dict[str, str] = {ext.source: ext.summary for ext in extraction_map.values() if ext.summary}
    for bucket in concept_lookup.values():
        bucket_summaries: list[str] = []
        for src in bucket.get("sources", []):
            text = summary_by_source.get(src, "")
            if text and text not in bucket_summaries:
                bucket_summaries.append(text)
        bucket["summaries"] = bucket_summaries

    return concept_lookup


def _write_concept_page(output_dir: Path | None, page_id: str, title: str, content: str) -> None:
    """Write a generated concept page as a ``.md`` file next to compiled output.

    No-op when ``output_dir`` is ``None``. The page gets a ``title`` frontmatter
    block matching the compiled-wiki page format.
    """
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    body = render_frontmatter({"title": title}) + content
    (output_dir / f"{page_id}.md").write_text(body, encoding="utf-8")


def _generate_concept_page(llm_client: Any, concept_data: dict[str, Any]) -> str | None:
    """Use LLM to generate a wiki page for a concept from extraction data."""
    loader = get_template_loader()
    prompt = loader.render(
        "lint/remediate_concept.md.j2",
        concept_name=concept_data.get("name", ""),
        definitions=concept_data.get("definitions", []),
        sources=concept_data.get("sources", []),
        source_summaries=concept_data.get("summaries", []),
    )
    result = llm_client.try_complete(
        messages=[{"role": "user", "content": prompt}],
        role=LLMRole.LINT_REMEDIATE_CONCEPT,
    )
    if result is None or not result.content:
        return None
    return result.content.strip()


def _fuzzy_match_concept(
    normalized: str,
    lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the best matching concept for a normalized gap name."""
    if normalized in lookup:
        return lookup[normalized]
    for key, data in lookup.items():
        if normalized in key or key in normalized:
            return data
    return None


# ── Output formatting ──────────────────────────────────────────────────


_TIER_LABELS: dict[int, str] = {
    1: "Tier 1 - Auto-fixed",
    2: "Tier 2 - LLM-generated",
    3: "Tier 3 - Recommendations",
}

_STATUS_ICONS: dict[RemediationStatus, str] = {
    RemediationStatus.APPLIED: "+",
    RemediationStatus.SKIPPED: "~",
    RemediationStatus.RECOMMENDED: "!",
}


def remediation_to_text(actions: list[RemediationAction]) -> str:
    """Format remediation actions as a human-readable report."""
    if not actions:
        return "No remediation actions needed."

    lines: list[str] = [f"Remediation Report ({len(actions)} action{'s' if len(actions) != 1 else ''})"]

    current_tier = 0
    for action in actions:
        if action.tier != current_tier:
            current_tier = action.tier
            lines.append(f"\n[{_TIER_LABELS.get(action.tier, f'Tier {action.tier}')}]")

        icon = _STATUS_ICONS.get(action.status, "?")
        lines.append(f"  {icon} {action.category}: {action.description}")

    applied = sum(1 for a in actions if a.status == RemediationStatus.APPLIED)
    skipped = sum(1 for a in actions if a.status == RemediationStatus.SKIPPED)
    recommended = sum(1 for a in actions if a.status == RemediationStatus.RECOMMENDED)
    lines.append("")
    lines.append(f"Summary: {applied} applied, {skipped} skipped, {recommended} recommended")

    return "\n".join(lines)
