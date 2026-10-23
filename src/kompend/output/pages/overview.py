"""Generate ``overview.md`` — LLM-generated thematic summary (opt-in)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.types import OverviewDeps, OverviewPage
from kompend.agents.wiki_index import AgenticOverviewAgent, WikiIndexAgent
from kompend.core.cache.store import CacheStore
from kompend.core.models import CompiledWiki
from kompend.core.paths import OVERVIEW_MD, VaultPaths
from kompend.core.templates import get_template_loader
from kompend.logging import get_logger

logger = get_logger(__name__)

_loader = get_template_loader()


def write_overview(
    wiki: CompiledWiki,
    base_dir: Path,
    agent: WikiIndexAgent | AgenticOverviewAgent,
    *,
    overview_budget: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path | None:
    """Write ``overview.md`` (LLM-generated). Returns the path or None."""
    source_texts = [n.content for n in wiki.nodes if n.content]

    # Overview cache (inside .kompend)
    cache_store = CacheStore(VaultPaths(base_dir).cache_dir)
    text_block = "\n".join(sorted(source_texts))
    overview_deps = {
        "source_texts": hashlib.sha256(text_block.encode()).hexdigest(),
        "model": agent.model_string,  # nocheck: overview cache deps, not wiki metadata
        "overview_budget": str(overview_budget),
    }
    cached = cache_store.get("overview", overview_deps)
    if isinstance(cached, str):
        overview_file = base_dir / OVERVIEW_MD
        overview_file.write_text(cached, encoding="utf-8")
        return overview_file

    if isinstance(agent, AgenticOverviewAgent):
        pages = [OverviewPage(slug=n.id, title=n.title, content=n.content) for n in wiki.nodes if n.content]
        content = generate_overview_agentic(pages, agent, progress_callback=progress_callback)
    else:
        content = generate_overview(source_texts, agent, overview_budget=overview_budget, progress_callback=progress_callback)
    if not content:
        return None

    cache_store.set("overview", overview_deps, f"# Wiki Overview\n\n{content}\n")

    overview_file = base_dir / OVERVIEW_MD
    overview_file.write_text(f"# Wiki Overview\n\n{content}\n", encoding="utf-8")
    return overview_file


def generate_overview_agentic(
    pages: list[OverviewPage],
    agent: AgenticOverviewAgent,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> str | None:
    """Generate an overview via a single agentic multi-turn session.

    The agent discovers pages via ``list_pages()``, reads full content via
    ``read_page()``, and submits the final overview via ``save_overview()``.
    Returns the overview Markdown, or ``None`` on failure.
    """
    if not pages:
        return None
    if progress_callback:
        progress_callback(0, 1)
    deps = OverviewDeps(pages=pages)
    try:
        agent.run(deps)
    except Exception as exc:
        logger.warning("Agentic overview synthesis failed: %s", exc)
        return None
    if progress_callback:
        progress_callback(1, 1)
    return deps.overview or None


def generate_overview(
    source_texts: list[str],
    agent: WikiIndexAgent,
    *,
    overview_budget: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> str | None:
    """Generate an overview of the wiki from node content (legacy map-reduce).

    All *source_texts* are processed — no content is dropped.  If the total
    exceeds ``overview_budget`` chars, texts are batched and each batch
    produces a partial overview.  The partials are then merged in a final
    synthesis call.

    Returns Markdown text, or ``None`` when there is no content to summarise.
    LLM failures propagate to the caller — overview is optional, so the caller
    catches and reports the real error rather than silently producing nothing.
    """
    if not source_texts:
        return None

    batches: list[str] = []
    current_parts: list[str] = []
    current_len = 0
    for text in source_texts:
        if current_len + len(text) > overview_budget and current_parts:
            batches.append("\n---\n".join(current_parts))
            current_parts = []
            current_len = 0
        current_parts.append(text)
        current_len += len(text)
    if current_parts:
        batches.append("\n---\n".join(current_parts))

    total_steps = len(batches) + (1 if len(batches) > 1 else 0)
    if progress_callback:
        progress_callback(0, total_steps)

    partials: list[str] = []
    for idx, batch_text in enumerate(batches):
        prompt = _loader.render("output/synthesize.md.j2", wiki_content=batch_text)
        deps = PipelineAgentDeps(text=prompt, source_id="overview")
        try:
            result = agent.run(deps)
            if result and isinstance(result, dict):
                content = result.get("content", "") or str(result.get("text", ""))
                if content:
                    partials.append(content)
        except Exception as exc:
            logger.warning("Overview synthesis batch %d failed: %s", idx, exc)
        if progress_callback:
            progress_callback(idx + 1, total_steps)

    if not partials:
        return None

    if len(partials) == 1:
        return partials[0]

    # Merge partial overviews. A merge failure propagates to the caller,
    # which reports the real error — concatenating partials silently would
    # hide a real LLM failure behind a degraded overview.
    combined = "\n\n".join(partials)
    prompt = _loader.render("output/synthesize.md.j2", wiki_content=combined)
    deps = PipelineAgentDeps(text=prompt, source_id="overview-merge")
    try:
        result = agent.run(deps)
        if result and isinstance(result, dict):
            content = result.get("content", "") or str(result.get("text", ""))
            if content:
                if progress_callback:
                    progress_callback(total_steps, total_steps)
                return content
    except Exception as exc:
        logger.warning("Overview merge failed: %s", exc)
    return None
