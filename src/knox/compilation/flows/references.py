"""Reference-extraction flow — detect and parse bibliographic references.

Extracted from :class:`TwoPhaseCompiler` to keep :mod:`flows` as the home
for all compilation flow logic. Runs before the main extraction phase:

1. Concatenate all chunks for a source into full text.
2. Run :func:`~knox.compilation.reference_excision.excise_references`
   to split into ``(body, ref_section)``.
3. If a reference section is found, run
   :class:`~knox.agents.reference_extraction.ReferenceExtractionAgent`
   on it and store results in
   :class:`~knox.core.pipeline.references.ReferenceStore`.
4. Strip the reference section from the chunks so the extraction phase
   sees only substantive content.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from knox.agents.deps import PipelineAgentDeps
from knox.agents.reference_extraction import ReferenceExtractionAgent
from knox.agents.reference_resolver import ReferenceResolverAgent
from knox.compilation.base import BaseCompiler
from knox.compilation.reference_excision import excise_references
from knox.core.models import ResolvedChunk
from knox.core.paths import rel_to_vault
from knox.core.pipeline.references import ReferenceStore
from knox.logging import get_logger

logger = get_logger(__name__)


def run_reference_extraction(
    compiler: BaseCompiler,
    chunks: list[ResolvedChunk],
    store: ReferenceStore,
) -> None:
    """Extract bibliographic references from each source.

    For each source, concatenate its chunks into full text, excise the
    reference section, parse references with the LLM agent, and store
    results. Mutates *store* in place.

    Logs a summary at the end. Failures are non-fatal — the pipeline
    continues without reference data for that source.
    """
    model_string = compiler._model  # noqa: SLF001
    timeout = compiler._timeout  # noqa: SLF001
    base_dir = compiler._config.paths.base_dir  # noqa: SLF001

    # Group chunks by source (same pattern as run_extraction). Labels are
    # vault-relative so they persist relocatable in references.json.
    by_source: dict[str, list[ResolvedChunk]] = defaultdict(list)
    for rc in chunks:
        label = rel_to_vault(rc.chunk.source_file or "unknown", base_dir)
        by_source[label].append(rc)

    num_sources = len(by_source)
    total_refs = 0

    agent: ReferenceExtractionAgent | None = None
    if model_string:
        try:
            agent = ReferenceExtractionAgent(model_string=model_string)
        except Exception as exc:  # noqa: BLE001 — agent construction may fail without model
            logger.warning("ReferenceExtractionAgent init failed: %s — skipping reference extraction", exc)
            return

    for source_idx, (source, source_chunks) in enumerate(by_source.items(), start=1):
        full_text = "\n\n".join(rc.chunk.text for rc in source_chunks if rc.chunk.text)
        if not full_text:
            continue

        body, ref_section = excise_references(full_text)

        if not ref_section:
            logger.debug("TwoPhase: no reference section found in '%s'", source)
            continue

        logger.info(
            "TwoPhase: references [%d/%d] %s — %d chars",
            source_idx,
            num_sources,
            Path(source).name,
            len(ref_section),
        )

        if agent is None:
            logger.debug("TwoPhase: no model configured — reference section detected but not parsed")
            continue

        try:
            deps = PipelineAgentDeps(text=ref_section, source_id=source, timeout=timeout)
            result = agent.run(deps)
        except Exception as exc:  # noqa: BLE001 — PydanticAI raises many error subclasses
            logger.warning("Reference extraction failed for '%s': %s", source, exc)
            continue

        refs = result.references if result else []
        # Stamp each reference with the citing source
        for ref in refs:
            ref.citing_source = source
            if not ref.id:
                # Generate a fallback id from the short label
                ref.id = ref.short_label().lower().replace(" ", "_")

        count = store.add_source(source, refs)
        total_refs += count
        logger.info("TwoPhase: extracted %d references from '%s'", count, source)

    if total_refs > 0:
        store.save()
        logger.info(
            "TwoPhase: reference extraction complete — %d references from %d source(s)",
            total_refs,
            store.source_count(),
        )


def resolve_unresolved_references(
    store: ReferenceStore,
    agent: ReferenceResolverAgent,
    timeout: float,
    *,
    source: str | None = None,
    dry_run: bool = False,
) -> int:
    """Resolve missing DOIs/URLs for references that lack them.

    Iterates references that have neither a DOI nor a URL (optionally
    filtered to a single *source*), asks the resolver *agent* to recover a
    canonical identifier, and writes it back to the store. When *dry_run*
    is true, the store is not saved and nothing is persisted.

    Returns the number of references resolved. Failures are non-fatal — the
    pipeline continues without resolution for that entry.

    Parameters
    ----------
    store:
        The reference store to resolve against (mutated in place).
    agent:
        A constructed :class:`~knox.agents.reference_resolver.ReferenceResolverAgent`.
    timeout:
        Per-call timeout forwarded to the agent's deps.
    source:
        When given, only this source's unresolved references are scanned.
    dry_run:
        When true, do not persist resolved identifiers to disk.
    """
    pairs = store.unresolved_indices(source)
    if not pairs:
        return 0

    total = len(pairs)
    resolved = 0

    for i, (src, idx) in enumerate(pairs, 1):
        refs = store.get_source(src)
        if idx >= len(refs):
            continue
        ref = refs[idx]
        query_text = ref.raw or ref.title or ref.short_label()
        if not query_text:
            continue

        label = ref.short_label() or ref.id or f"entry {idx}"
        logger.info(
            "TwoPhase: resolve [%d/%d] %s — %s",
            i,
            total,
            Path(src).name,
            label,
        )

        deps = PipelineAgentDeps(text=query_text, source_id=src, timeout=timeout)
        try:
            result = agent.run(deps)
        except Exception as exc:  # noqa: BLE001 — PydanticAI raises many error subclasses
            logger.warning("Reference resolution failed for '%s' entry %d: %s", src, idx, exc)
            continue

        if not result.resolved:
            continue
        if store.update_reference(src, idx, doi=result.doi, url=result.url):
            resolved += 1

    if resolved and not dry_run:
        store.save()
        logger.info(
            "TwoPhase: resolved %d reference(s) (%d of %d left unresolved)",
            resolved,
            total - resolved,
            total,
        )
    elif dry_run and resolved:
        logger.info(
            "TwoPhase: %d reference(s) would be resolved (dry-run, not saved)",
            resolved,
        )

    return resolved
