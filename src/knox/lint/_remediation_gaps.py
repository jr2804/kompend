"""Coverage-gap remediation — generates concept pages for missing topics.

Extracted from :mod:`knox.lint.remediation` to keep the remediation module
under 350 lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slugify import slugify

from knox.agents.deps import PipelineAgentDeps
from knox.agents.remediate import RemediationAgent
from knox.compilation.two_phase import Extraction, ExtractionSet
from knox.core.models import CompiledWiki, WikiNode
from knox.core.templates import get_template_loader
from knox.lint.base import LintReportResult, RemediationAction, RemediationStatus
from knox.logging import get_logger
from knox.organize.categorize import load_all_extractions
from knox.output.renderer import render_frontmatter

logger = get_logger(__name__)


def remediate_coverage_gaps(
    wiki: CompiledWiki,
    results: list[LintReportResult],
    *,
    base_dir: Path,
    agent: RemediationAgent,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> list[RemediationAction]:
    """Generate concept pages for coverage gaps using cached extractions."""
    gap_concepts: list[str] = []
    for r in results:
        if r.rule != "coverage-gap" or r.outcome != "fail":
            continue
        concept = r.details.get("concept") or r.message
        if concept:
            gap_concepts.append(concept)

    if not gap_concepts:
        return []

    extraction_map = load_all_extractions(base_dir, base_dir / "sources")
    if not extraction_map:
        logger.info("No cached extractions found -- skipping coverage gap remediation")
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

    token_index = _build_token_index(concept_lookup)
    for gap in gap_concepts:
        matched = _fuzzy_match_concept(gap.lower().strip(), concept_lookup, token_index)
        if not matched:
            actions.append(
                RemediationAction(
                    tier=3,
                    category="Coverage gap (no extraction match)",
                    description=f"'{gap}' -- not found in cached extractions",
                    status=RemediationStatus.RECOMMENDED,
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
                    status=RemediationStatus.SKIPPED,
                    details={"sources": matched.get("sources", []), "concept": gap, "matched_concept": concept_name},
                )
            )
            continue

        content = _generate_concept_page(agent, matched)
        if content is None:
            actions.append(
                RemediationAction(
                    tier=2,
                    category="Coverage gap (LLM failed)",
                    description=f"Failed to generate page for '{concept_name}'",
                    status=RemediationStatus.SKIPPED,
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
                status=RemediationStatus.APPLIED,
                details={"sources": matched.get("sources", []), "concept": gap, "matched_concept": concept_name},
            )
        )

    return actions


def _build_concept_lookup(
    extraction_map: dict[str, Extraction],
) -> dict[str, dict[str, Any]]:
    """Build a normalized concept lookup from cached extractions."""
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
    """Write a generated concept page as a ``.md`` file next to compiled output."""
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    body = render_frontmatter({"title": title}) + content
    (output_dir / f"{page_id}.md").write_text(body, encoding="utf-8")


def _generate_concept_page(agent: RemediationAgent, concept_data: dict[str, Any]) -> str | None:
    """Use the remediation agent to generate a wiki page for a concept."""
    loader = get_template_loader()
    prompt = loader.render(
        "lint/remediate_concept.md.j2",
        concept_name=concept_data.get("name", ""),
        definitions=concept_data.get("definitions", []),
        sources=concept_data.get("sources", []),
        source_summaries=concept_data.get("summaries", []),
    )
    deps = PipelineAgentDeps(text=prompt, source_id="lint-remediate", timeout=120)
    result = agent.run(deps)
    return result if result else None


def _build_token_index(lookup: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Build an inverted index from tokens to concept keys."""
    index: dict[str, list[str]] = {}
    for key in lookup:
        for token in key.lower().split():
            index.setdefault(token, []).append(key)
    return index


def _fuzzy_match_concept(
    normalized: str,
    lookup: dict[str, dict[str, Any]],
    token_index: dict[str, list[str]] | None = None,
) -> dict[str, Any] | None:
    """Find the best matching concept for a normalized gap name."""
    if normalized in lookup:
        return lookup[normalized]
    if token_index is not None:
        gap_tokens = set(normalized.split())
        for token in gap_tokens:
            for key in token_index.get(token, []):
                return lookup[key]
        return None
    # Fallback: linear scan (no index available).
    for key, data in lookup.items():
        if normalized in key or key in normalized:
            return data
    return None
