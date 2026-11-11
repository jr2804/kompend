"""Coverage-gap remediation — generates concept pages for missing topics.

Extracted from :mod:`kompend.lint.remediation` to keep the remediation module
under 350 lines.
"""

from __future__ import annotations

from pathlib import Path

from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.remediate import AgenticRemediationAgent, RemediationAgent
from kompend.agents.types import RemediationDeps
from kompend.compilation.two_phase import Extraction, ExtractionSet
from kompend.core.models import CompiledWiki, WikiNode
from kompend.core.models.enums import NodeType
from kompend.core.templates import get_template_loader
from kompend.core.text.matching import best_match
from kompend.core.text.slug import slugify
from kompend.core.utils.exceptions import LLMOutputError
from kompend.lint.base import LintResult, RemediationAction, RemediationStatus
from kompend.lint.details import LintField
from kompend.lint.gap_broker import GapBroker, GapConcept, gap_concept
from kompend.logging import get_logger
from kompend.organize.categorize import load_all_extractions
from kompend.output.markdown import MarkdownWriter

#: rapidfuzz-scale (0–100) acceptance threshold for gap-name → concept
#: matching. Mirrors the legacy 0.85 fuzzy_threshold on the new scale.
_REMEDIATION_MATCH_THRESHOLD = 85.0

logger = get_logger(__name__)


def remediate_coverage_gaps(
    wiki: CompiledWiki,
    results: list[LintResult],
    *,
    base_dir: Path,
    agent: RemediationAgent | AgenticRemediationAgent,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> list[RemediationAction]:
    """Generate concept pages for coverage gaps using cached extractions.

    When *agent* is an :class:`AgenticRemediationAgent`, the agent runs
    once over all matched gaps via a :class:`GapBroker` — collapsing N
    per-gap calls into a single multi-turn session.
    """
    gap_names: list[str] = []
    for r in results:
        if r.rule != "coverage-gap" or r.outcome != "fail":
            continue
        concept = r.details.get(LintField.CONCEPT) or r.message
        if concept:
            gap_names.append(concept)

    if not gap_names:
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
        existing_ids |= {f.stem for f in (output_dir / "concepts").glob("*.md")}
    node_index: dict[str, int] = {n.id: i for i, n in enumerate(wiki.nodes)}

    # Deterministic pre-filtering: fuzzy-match + existence check.
    matched_buckets: list[GapConcept] = []
    actions: list[RemediationAction] = []

    for gap in gap_names:
        matched = _fuzzy_match_concept(gap.lower().strip(), concept_lookup)
        if not matched:
            logger.info("'%s' not found in cached extractions; see 'kompend lint gaps' for recommendations", gap)
            continue
        concept_name = matched.get("name", gap)
        page_id = slugify(concept_name)
        if page_id in existing_ids or concept_name.lower().strip() in existing_titles:
            continue
        if dry_run:
            actions.append(
                RemediationAction(
                    category="Coverage gap",
                    description=f"Would generate concept page for '{concept_name}'",
                    status=RemediationStatus.SKIPPED,
                    details={LintField.SOURCES: matched.get("sources", []), LintField.CONCEPT: gap, LintField.MATCHED_CONCEPT: concept_name},
                )
            )
            continue
        matched_buckets.append(matched)

    if not matched_buckets:
        return actions

    # --- Agentic path: single agent session over all matched gaps ----
    if isinstance(agent, AgenticRemediationAgent):
        broker = GapBroker(matched_buckets)
        deps = RemediationDeps(gaps=broker, dry_run=dry_run)
        try:
            agent.run(deps)
        except Exception as exc:
            logger.warning("Agentic remediation failed: %s", exc)

        for page_result in deps.pages:
            concept_name = page_result.name
            page_id = slugify(concept_name)
            new_node = WikiNode(
                id=page_id,
                title=concept_name,
                content=page_result.content,
                node_type=NodeType.CONCEPT,
                metadata={"source": "lint-remediation"},
            )
            # In-place replace ghost nodes, otherwise append.
            if page_id in node_index:
                wiki.nodes[node_index[page_id]] = new_node
            else:
                node_index[page_id] = len(wiki.nodes)
                wiki.nodes.append(new_node)
            existing_ids.add(page_id)
            existing_titles.add(concept_name.lower().strip())
            actions.append(
                RemediationAction(
                    category="Coverage gap",
                    description=f"Generated concept page for '{concept_name}'",
                    status=RemediationStatus.APPLIED,
                    details={LintField.CONCEPT: concept_name, LintField.ID: page_id},
                )
            )

        for gap_name, reason in broker.unfixable.items():
            actions.append(
                RemediationAction(
                    category="Coverage gap (unfixable)",
                    description=f"Cannot generate page for '{gap_name}': {reason}",
                    status=RemediationStatus.SKIPPED,
                    details={LintField.CONCEPT: gap_name, LintField.REASON: reason},
                )
            )

        if output_dir is not None and deps.pages:
            MarkdownWriter(vault_mode=True).write(wiki, output_dir)

        return actions

    # --- Legacy single-shot path: one LLM call per gap ----------------
    for matched in matched_buckets:
        concept_name = matched.get("name", "")
        content = _generate_concept_page(agent, matched)
        if content is None:
            actions.append(
                RemediationAction(
                    category="Coverage gap (LLM failed)",
                    description=f"Failed to generate page for '{concept_name}'",
                    status=RemediationStatus.SKIPPED,
                    details={LintField.CONCEPT: concept_name},
                )
            )
            continue

        page_id = slugify(concept_name)
        new_node = WikiNode(
            id=page_id,
            title=concept_name,
            content=content,
            node_type=NodeType.CONCEPT,
            metadata={
                "source": "lint-remediation",
                "generated_from": matched.get("sources", []),
            },
        )
        if page_id in node_index:
            wiki.nodes[node_index[page_id]] = new_node
        else:
            node_index[page_id] = len(wiki.nodes)
            wiki.nodes.append(new_node)
        existing_ids.add(page_id)
        existing_titles.add(concept_name.lower().strip())
        actions.append(
            RemediationAction(
                category="Coverage gap",
                description=f"Generated concept page '{concept_name}' from {len(matched.get('sources', []))} source(s)",
                status=RemediationStatus.APPLIED,
                details={LintField.SOURCES: matched.get("sources", []), LintField.CONCEPT: gap, LintField.MATCHED_CONCEPT: concept_name},
            )
        )

    if output_dir is not None:
        MarkdownWriter(vault_mode=True).write(wiki, output_dir)

    return actions


def _build_concept_lookup(
    extraction_map: dict[str, Extraction],
) -> dict[str, GapConcept]:
    """Build a normalized concept lookup from cached extractions."""
    ext_set = ExtractionSet(extractions=list(extraction_map.values()))
    concept_lookup: dict[str, GapConcept] = {}
    for cp in ext_set.concept_pages():
        bucket = gap_concept(cp)
        name = bucket["name"]
        if name:
            concept_lookup[name.lower().strip()] = bucket

    for ext in extraction_map.values():
        for ent in ext.entities:
            name = ent.name.strip()
            if not name:
                continue
            key = name.lower().strip()
            bucket = concept_lookup.setdefault(key, GapConcept(name=name, sources=[], definitions=[]))
            sources = bucket["sources"]
            if ext.source not in sources:
                sources.append(ext.source)
            bucket["sources"] = sources
            definitions = bucket["definitions"]
            context = ent.context
            if context and context not in definitions:
                definitions.append(context)
            bucket["definitions"] = definitions

    summary_by_source: dict[str, str] = {ext.source: ext.summary for ext in extraction_map.values() if ext.summary}
    for bucket in concept_lookup.values():
        bucket_summaries: list[str] = []
        for src in bucket["sources"]:
            text = summary_by_source.get(src, "")
            if text and text not in bucket_summaries:
                bucket_summaries.append(text)
        bucket["summaries"] = bucket_summaries

    return concept_lookup


def _generate_concept_page(agent: RemediationAgent, concept_data: GapConcept) -> str | None:
    """Use the remediation agent to generate a wiki page for a concept."""
    loader = get_template_loader()
    prompt = loader.render(
        "lint/remediate_concept.md.j2",
        concept_name=concept_data.get("name", ""),
        definitions=concept_data.get("definitions", []),
        sources=concept_data.get("sources", []),
        source_summaries=concept_data.get("summaries", []),
    )
    deps = PipelineAgentDeps(text=prompt, source_id="lint-remediate")
    try:
        result = agent.run(deps)
    except LLMOutputError as exc:
        # One unresolvable concept must not abort the whole remediation pass:
        # the caller records a SKIPPED action and moves on to the next gap.
        logger.warning("Remediation failed for '%s': %s", concept_data.get("name", ""), exc)
        return None
    return result if result else None


def _fuzzy_match_concept(
    normalized: str,
    lookup: dict[str, GapConcept],
    token_index: dict[str, list[str]] | None = None,
) -> GapConcept | None:
    """Find the best matching concept for a normalized gap name.

    Exact key first, then rapidfuzz similarity over the lookup keys
    (:func:`kompend.core.text.matching.best_match` — the single source of
    truth for string matching). *token_index* is accepted for call-site
    compatibility but no longer used: token-ratio matching subsumes the
    shared-token heuristic.
    """
    if normalized in lookup:
        return lookup[normalized]
    matched = best_match(normalized, set(lookup), threshold=_REMEDIATION_MATCH_THRESHOLD)
    return lookup[matched[0]] if matched else None
