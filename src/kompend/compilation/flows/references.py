"""Reference-extraction flow — detect and parse bibliographic references.

Extracted from :class:`TwoPhaseCompiler` to keep :mod:`flows` as the home
for all compilation flow logic. Runs before the main extraction phase:

1. Concatenate all chunks for a source into full text.
2. Run :func:`~kompend.compilation.reference_excision.excise_references`
   to split into ``(body, ref_section)``.
3. If a reference section is found, run
   :class:`~kompend.agents.reference_extraction.ReferenceExtractionAgent`
   on it (split into context-window-sized windows when the section
   exceeds the discovered budget) and store results in
   :class:`~kompend.core.pipeline.references.ReferenceStore`.
4. Strip the reference section from the chunks so the extraction phase
   sees only substantive content.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from pydantic_ai.messages import ModelMessage

from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.reference_extraction import ReferenceExtractionAgent
from kompend.compilation._two_phase_core import (
    TwoPhaseRuntime,
    context_safety_tokens,
    tokens_to_chars,
)
from kompend.compilation.reference_excision import excise_references
from kompend.core.models import ResolvedChunk
from kompend.core.models.references import Reference, ReferenceResolutionOutput
from kompend.core.paths import rel_to_vault
from kompend.core.pipeline.references import ReferenceStore
from kompend.core.templates import get_template_loader
from kompend.logging import get_logger

#: System-prompt template of the reference extraction agent. Its rendered
#: length is measured (the same way ``two_phase.py`` measures the
#: extraction/generation system prompts) so the window budget accounts for
#: the fixed per-request cost.
_REFERENCE_SYSTEM_TEMPLATE = "compilation/reference_extraction_system.md.j2"

logger = get_logger(__name__)


class ReferenceResolver(Protocol):
    """Minimal resolver surface used by :func:`resolve_unresolved_references`.

    :class:`~kompend.agents.reference_resolver.ReferenceResolverAgent` satisfies
    this structurally; tests use a lightweight stub.
    """

    def run(
        self,
        deps: PipelineAgentDeps,
        *,
        message_history: list[ModelMessage] | None = None,
    ) -> ReferenceResolutionOutput: ...


def run_reference_extraction(
    compiler: TwoPhaseRuntime,
    chunks: list[ResolvedChunk],
    store: ReferenceStore,
) -> None:
    """Extract bibliographic references from each source.

    For each source, concatenate its chunks into full text, excise the
    reference section, parse references with the LLM agent, and store
    results. Mutates *store* in place.

    Oversized sections are split into line-aligned windows that fit the
    discovered context-window budget (see ``_reference_char_budget``) and
    parsed window by window; results are merged per source.

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
    max_window_chars = 0
    if model_string:
        try:
            agent = ReferenceExtractionAgent(model_string=model_string)
            max_window_chars = _reference_char_budget(compiler)
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

        windows = [ref_section] if len(ref_section) <= max_window_chars else _split_windows(ref_section, max_window_chars)
        if len(windows) > 1:
            logger.info(
                "TwoPhase: references [%d/%d] %s — %d chars split into %d window(s)",
                source_idx,
                num_sources,
                Path(source).name,
                len(ref_section),
                len(windows),
            )

        refs: list[Reference] = []
        for win_idx, window in enumerate(windows, start=1):
            try:
                deps = PipelineAgentDeps(text=window, source_id=source, timeout=timeout)
                result = agent.run(deps)
            except Exception as exc:  # noqa: BLE001 — PydanticAI raises many error subclasses
                logger.warning(
                    "Reference extraction failed for '%s' (window %d/%d): %s",
                    source,
                    win_idx,
                    len(windows),
                    exc,
                )
                break  # keep references parsed from earlier windows
            if result:
                refs.extend(result.references)

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


def _reference_char_budget(compiler: TwoPhaseRuntime) -> int:
    """Character budget for one reference-extraction window.

    Mirrors ``flows/extraction.py``'s ``_extract_char_budget``: start from the
    discovered context window, subtract the measured system prompt, the
    configured output reserve (``compile.budget.reference_output_max``) and the
    shared context safety margin, then convert the remaining tokens to chars.
    A parsed bibliography is roughly as large as its raw text, so reserving a
    full output window keeps input plus expected output inside the model's
    limit — without it a large section overflowed as
    ``request (236320 tokens) exceeds the available context size (65536)``.
    """
    cw = compiler._context_window  # noqa: SLF001 — flow/runtime contract
    sys_tokens = _reference_system_tokens()
    out_tokens = compiler._budget.reference_output_max  # noqa: SLF001
    safety_tokens = context_safety_tokens(cw.max_input)
    content_tokens = max(1, cw.max_input - sys_tokens - out_tokens - safety_tokens)
    logger.info(
        "TwoPhase: reference budget (cw=%d sys=%d out=%d safety=%d) -> %d tokens for content",
        cw.max_input,
        sys_tokens,
        out_tokens,
        safety_tokens,
        content_tokens,
    )
    return tokens_to_chars(
        content_tokens,
        chars_per_token=compiler._config.llm.default_chars_per_token,  # noqa: SLF001
    )


@functools.lru_cache(maxsize=1)
def _reference_system_tokens() -> int:
    """Estimated token cost of the reference-extraction system prompt."""
    system = get_template_loader().get_body(_REFERENCE_SYSTEM_TEMPLATE)
    return max(1, len(system) // 4)


def _split_windows(text: str, max_chars: int) -> list[str]:
    """Split *text* into line-aligned windows of at most *max_chars* characters.

    Reference entries are typically one per line, so windows break between
    entries and the model never sees a torn citation. A single line longer
    than *max_chars* (rare) is hard-split as a fallback.
    """
    windows: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        line_len = len(line) + 1  # account for the join newline
        if current and size + line_len > max_chars:
            windows.append("\n".join(current))
            current, size = [], 0
        if line_len > max_chars:
            for i in range(0, len(line), max_chars):
                windows.append(line[i : i + max_chars])
            continue
        current.append(line)
        size += line_len
    if current:
        windows.append("\n".join(current))
    return windows


def resolve_unresolved_references(
    store: ReferenceStore,
    agent: ReferenceResolver,
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
        A constructed :class:`~kompend.agents.reference_resolver.ReferenceResolverAgent`.
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
