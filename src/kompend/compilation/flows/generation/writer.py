"""Page writing for the generation pass.

The writer assembles the final wiki from generated pages, resolves page
kinds against the taxonomy, learns wikilink aliases, and gates placeholder
submissions before they can reach the wiki.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Final

from kompend.agents.failure import EmptyAnswerError
from kompend.compilation.concept_keys import normalize_concept_key
from kompend.compilation.flows.assembly import (
    attach_attributes_to_nodes,
    attach_tags_to_nodes,
    build_edges,
)
from kompend.compilation.schema import GenerationOutput, GenerationPage
from kompend.compilation.two_phase import ExtractionSet, TwoPhaseRuntime, compute_quality_score, slugify
from kompend.core.keys import MANDATORY_KINDS
from kompend.core.models import CompiledWiki, ResolvedChunk, TocEntry, WikiField, WikiNode
from kompend.core.models.enums import ExtractionField, MetadataField, NodeType
from kompend.core.pipeline.entity_types import EntityTypeRegistry
from kompend.core.utils.exceptions import LLMOutputError
from kompend.lint.base import LintRuleOutcome
from kompend.lint.structural import BrokenWikilinksRule
from kompend.logging import LOG_SAMPLE_LIMIT, get_logger

# Matches wikilink-wrapped source citations like [[sources/foo.pdf]] so they
# can be normalized to the intended [sources/foo.pdf] inline citation format
# regardless of LLM compliance with the system prompt.
_SOURCE_WIKILINK_RE = re.compile(r"\[\[\s*(sources/[^\]\n]+?)\s*\]\]")


#: A page title shorter than this many characters (a bare "x") is a
#: placeholder identity, never a real page title. Two characters ("3D") pass.
_MIN_PAGE_TITLE_CHARS: Final[int] = 2

logger = get_logger(__name__)


class PlaceholderMarker(StrEnum):
    """Filler phrases that mark a page body as a placeholder.

    The two-phase writer can self-test its ``final_result`` tool and submit
    filler instead of a summary: a run recorded in ``steps.db`` accepted a
    page whose whole body was "Placeholder removed", and a reviewed wiki
    carried a page body announcing itself as prose written to test
    submission. None of these phrases occur in real source material, so a
    body containing one is a placeholder, never a summary.
    """

    #: The observed artifact body, verbatim.
    PLACEHOLDER_REMOVED = "placeholder removed"
    #: The classic typesetter filler.
    LOREM_IPSUM = "lorem ipsum"
    #: Announces the body as standing text rather than written prose.
    PLACEHOLDER_TEXT = "placeholder text"
    #: Announces the body as generic filler ("Body text here with …").
    BODY_TEXT_HERE = "body text here"
    #: Self-describing test language ("… prose to test submission").
    PROSE_TO_TEST = "prose to test"


class PlaceholderPageError(EmptyAnswerError):
    """A ``final_result`` submission carried a placeholder page.

    Subclasses :class:`~kompend.agents.failure.EmptyAnswerError` so the
    agentic pass loop applies its existing, proven remedy for "the model
    returned nothing usable": the unit is recorded, the pass is retried or
    the segment stops loudly — and the placeholder page is never silently
    dropped and never written to the wiki.
    """


def _reject_placeholder_pages(gen_output: GenerationOutput, *, unit: str) -> None:
    """Refuse the whole submission when any submitted page is a placeholder.

    The gate sits where a model submission enters the pipeline: the live
    pass result, and a transcript replayed from the step store. The
    offender is never silently dropped — the whole
    :class:`~kompend.compilation.schema.GenerationOutput` is refused with
    :class:`PlaceholderPageError`, which routes through the pass loop's
    existing failure path: the unit is recorded, and the segment either
    re-runs or fails loudly. The placeholder page never reaches the wiki.
    """
    offenders: list[str] = []
    for page in gen_output.pages:
        if reasons := _placeholder_reasons(page):
            offenders.append(f"'{page.title or '(untitled)'}' ({'; '.join(reasons)})")
    if offenders:
        raise PlaceholderPageError(f"{unit}: rejected {len(offenders)} placeholder page(s) — {'; '.join(offenders)}")


def _placeholder_reasons(page: GenerationPage) -> list[str]:
    """Reasons *page* is a placeholder, empty when the page is legitimate.

    A page with no provenance had no source to summarize (the generation
    prompt requires ``provenance`` — a list of cited source filenames — on
    every page), a filler body is the writer testing its own tool, and a
    single-character title is a test identity, not a page name.
    """
    reasons: list[str] = []
    if not [entry for entry in page.provenance if str(entry).strip()]:
        reasons.append("empty provenance (no source chunks cited)")
    body = (page.content or "").casefold()
    markers = [marker.value for marker in PlaceholderMarker if marker.value in body]
    if markers:
        reasons.append(f"filler body text ({', '.join(markers)})")
    title = (page.title or "").strip()
    if title and len(title) < _MIN_PAGE_TITLE_CHARS:
        reasons.append(f"title '{title}' is shorter than {_MIN_PAGE_TITLE_CHARS} characters")
    return reasons


def _resolve_page_kind(
    model_kind: str,
    title: str,
    preassigned: dict[str, str],
    extras: dict[str, str],
) -> NodeType:
    """Resolve a page kind: a pre-assigned kind wins, else the model's label.

    ``_with_entity_pages`` pre-assigns ``kind: entity`` to mandatory pages.
    Those are enforced deterministically so a model that drops or alters the
    kind cannot misfile the page (entity subdirectory routing depends on it).
    Pages the model invents keep their own kind, validated by ``_allowed_kind``.
    """
    expected = preassigned.get(normalize_concept_key(title))
    return _allowed_kind(expected or model_kind, extras)


def _allowed_kind(kind: str, extras: dict[str, str]) -> NodeType:
    """Validate an LLM-assigned page kind against the taxonomy.

    Mandatory kinds plus configured extras are accepted; anything else falls
    back to ``concept`` (permissive-consumption model, OKF §10).
    """
    if kind in MANDATORY_KINDS:
        return NodeType(kind)
    if kind in extras:
        return NodeType.of(kind)
    return NodeType.CONCEPT


def assemble_wiki(
    compiler: TwoPhaseRuntime,
    nodes: list[WikiNode],
    slug_to_idx: dict[str, int],
    chunks: list[ResolvedChunk],
    extractions: ExtractionSet,
    *,
    concepts: list[dict[str, Any]],
    pinned_clean: list[str],
) -> CompiledWiki:
    """Filter, dedup, and assemble the final CompiledWiki."""
    # Persist any new entity-type decisions to the on-disk registry before
    # assembly runs. Assembly reads from this registry (so previously-known
    # kinds win), but new concepts from THIS compile still need to land here
    # so the NEXT compile sees them.
    try:
        entity_types_registry = EntityTypeRegistry(compiler._config.paths.entity_types_file)
        new_entries = 0
        for c in concepts:
            if c.get("kind") != NodeType.ENTITY.value:
                continue
            key = normalize_concept_key(str(c.get(ExtractionField.NAME, "")))
            if not key:
                continue
            if entity_types_registry.get(key) is not None:
                continue
            sanitized = _sanitize_entity_type(str(c.get("type", "")))
            if sanitized and entity_types_registry.record(key, sanitized):
                new_entries += 1
        if new_entries:
            entity_types_registry.save()
            logger.info(
                "Entity-type registry: persisted %d new entry/entries (total=%d)",
                new_entries,
                len(entity_types_registry),
            )
    except OSError as exc:
        logger.warning("Could not persist entity-type registry: %s", exc, exc_info=True)
    cw = compiler._context_window

    if not nodes:
        raise LLMOutputError(
            f"Generation failed: 0 of {len(concepts)} concept(s) synthesized. "
            f"Every LLM call failed after retries (see the transient/validation warnings above - "
            f"an upstream long-request timeout is not a structured-output problem). "
            f"Check that model '{compiler._model}' can produce structured JSON output."
        )

    # Force-merge by concept key (normalize_concept_key) FIRST. This catches
    # near-duplicates that share a key but different slugs (e.g. "Absolute
    # Category Rating" and "Absolute Category Rating (ACR)") which the
    # embedding cosine pass below would also catch, but doing it
    # deterministically first avoids unnecessary embedding work and avoids
    # silent fallthrough when the embedding model is unavailable.
    nodes = compiler.merge_by_concept_key(nodes)

    compiler.merge_duplicate_nodes(
        nodes,
        compiler._config.embedding.model,
        threshold=compiler._config.budget.concept_merge_threshold,
    )
    compiler.disambiguate_duplicate_titles(nodes, chunks)
    nodes = compiler.filter_empty_nodes(nodes, min_content_length=compiler._config.budget.min_content_length)

    if not nodes:
        logger.warning(
            "TwoPhase: generation produced 0 pages after filtering (min_content_length=%d). Check LLM output quality.",
            compiler._config.budget.min_content_length,
        )
        return CompiledWiki(
            nodes=[],
            edges=[],
            toc=[],
            metadata={
                WikiField.CHUNK_COUNT: len(chunks),
                MetadataField.MODEL: compiler._model or "",
                MetadataField.TWO_PHASE: True,
                MetadataField.GENERATION_SKIPPED: True,
                MetadataField.SKIP_REASON: "all pages filtered as empty",
                MetadataField.NODE_COUNT: 0,
            },
        )

    nodes = compiler.dedup_content_duplicate_nodes(nodes)
    attach_tags_to_nodes(nodes, chunks)
    attach_attributes_to_nodes(nodes, chunks)
    edges = build_edges(nodes, extractions, min_title_len=compiler._config.budget.edge_min_title_len)
    quality = compute_quality_score(nodes, edges)
    toc: list[TocEntry] = [{"title": n.title, "id": n.id, "level": 1} for n in nodes]
    aliases = learn_wikilink_aliases(nodes)
    if aliases:
        logger.info("TwoPhase: learned %d wikilink alias(es), e.g. %s", len(aliases), list(aliases.items())[:LOG_SAMPLE_LIMIT])
    return CompiledWiki(
        nodes=nodes,
        edges=edges,
        toc=toc,
        quality_score=quality,
        metadata={
            WikiField.CHUNK_COUNT: len(chunks),
            MetadataField.NODE_COUNT: len(nodes),
            MetadataField.EDGE_COUNT: len(edges),
            MetadataField.MODEL: compiler._model or "",
            MetadataField.TWO_PHASE: True,
            MetadataField.CONCEPTS_MERGED: len(concepts),
            MetadataField.CONTEXT_WINDOW: cw.max_input,
            MetadataField.PINNED_FACTS_COUNT: len(pinned_clean),
            MetadataField.WIKILINK_ALIASES: aliases,
        },
    )


def _sanitize_entity_type(raw: str) -> str:
    """Normalize an entity type for use as a folder name ("Person" → "person")."""
    sanitized = slugify(raw or "", max_length=40)
    return sanitized or "entity"


def learn_wikilink_aliases(nodes: list[WikiNode]) -> dict[str, str]:
    """Learn an alias → canonical-title map from wikilinks in *nodes*.

    Runs :class:`BrokenWikilinksRule` over the freshly generated pages and
    collects every failure the rule tagged ``alias_of`` (a fuzzy/substring
    match against an existing page). The result persists in
    ``wiki.metadata[MetadataField.WIKILINK_ALIASES]`` and feeds
    :func:`build_known_titles` on the next compile, so the LLM is nudged
    toward canonical titles while the renderer resolves leftovers.
    """
    if not nodes:
        return {}
    wiki = CompiledWiki(nodes=nodes, edges=[], toc=[])
    id_to_title = {n.id: n.title for n in nodes}
    aliases: dict[str, str] = {}
    for result in BrokenWikilinksRule().check(wiki):
        if result.outcome != LintRuleOutcome.FAIL:
            continue
        alias_of = result.details.get("alias_of")
        canonical = id_to_title.get(str(alias_of))
        if canonical:
            aliases.setdefault(result.details.get("target", ""), canonical)
    aliases.pop("", None)
    return aliases
