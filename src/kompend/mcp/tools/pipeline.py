"""Pipeline MCP tools (ingest, index, write, organize).

Compile/parse already live in knowledge.py (wiki_compile/wiki_parse).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from kompend.agents.categorize import CategorizeAgent
from kompend.agents.deps import PipelineAgentDeps
from kompend.agents.write import WriteAgent
from kompend.cli._pipeline_helpers import resolve_compile_inputs
from kompend.cli._post_compile import write_wiki_cache
from kompend.cli._shared import build_cli_overrides
from kompend.cli._sources import resolve_files
from kompend.core import source_metadata
from kompend.core.config import resolve_config
from kompend.core.config._configs import BudgetConfig, LLMConfig
from kompend.core.models import MetadataField, load_cached_wiki, load_wiki_from_files
from kompend.core.models.enums import NodeType, OkfField
from kompend.core.models.metadata import meta_str_list
from kompend.core.paths import VaultPaths, vault_paths
from kompend.core.pipeline.ingest import ingest_documents
from kompend.core.templates import get_template_loader
from kompend.core.text.slug import slugify
from kompend.core.text.windows import split_text_windows
from kompend.core.trust.actor import resolve_actor
from kompend.core.utils.time import iso_local
from kompend.llm.format import strip_reasoning
from kompend.logging import get_logger
from kompend.organize.engine import organize_sources, resolve_organize_config
from kompend.output.markdown import MarkdownWriter
from kompend.output.models import Document
from kompend.preprocessing.parsers._registry import ParserBuildOptions, build_default_router
from kompend.rag import build_vector_index

logger = get_logger(__name__)


def wiki_ingest(
    wiki_dir: str | None = None,
    files: list[str] | None = None,
    online_urls: list[str] | None = None,
    url_file: str | None = None,
    recursive: bool | None = None,
    ignore: list[str] | None = None,
    no_cache: bool = False,
    workers: int = 1,
    in_place: bool = False,
    parser: str | None = None,
    tags: list[str] | None = None,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Ingest documents into the knowledge graph without compiling.

    Parses every source and caches the structured documents in .kompend/cache/.
    Run wiki_compile afterwards to chunk, resolve, and compile.
    """
    online_urls = online_urls or []
    vp = vault_paths(wiki_dir)
    cli_overrides = build_cli_overrides(
        no_cache=no_cache,
        workers=workers,
        in_place_sources=in_place,
    )
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=cli_overrides, config_file=config_file)
    sources_dir = vp.sources_dir
    base_dir = vp.base_dir

    all_input_files = resolve_compile_inputs(
        files or [],
        online_urls,
        Path(url_file) if url_file else None,
        sources_dir,
        cfg.kompend.paths.recursive if recursive is None else recursive,
        ignore,
        in_place=cfg.kompend.compile.in_place_sources,
        undetected=cfg.kompend.downloader.undetected,
    )
    if not all_input_files:
        return {"status": "error", "message": f"No supported files found in {sources_dir}."}

    images_dir = VaultPaths(base_dir).images_dir
    router = build_default_router(ParserBuildOptions.from_config(cfg, images_dir=images_dir))

    try:
        ingest_start = time.monotonic()
        result = ingest_documents(
            files=all_input_files,
            base_dir=base_dir,
            router=router,
            no_cache=no_cache,
            config=dict(cfg.flat()) if hasattr(cfg, "flat") else None,
            ingested_by=resolve_actor(cfg),
            tags=tags,
            rumdl_config=cfg.kompend.compile.output.rumdl_config,
            flavor=cfg.kompend.compile.output.flavor,
        )
        elapsed = time.monotonic() - ingest_start
    except Exception as exc:
        logger.exception("Ingest failed")
        return {"status": "error", "message": str(exc)}

    summary = f"Ingested {result.count} document(s) into {result.cache_dir}"
    if result.cached:
        summary += f" ({result.cached} cached)"
    if result.skipped:
        summary += f" ({result.skipped} skipped)"
    if result.failed:
        summary += f" ({result.failed} failed)"
    if result.dedup_skipped:
        summary += f" ({result.dedup_skipped} duplicates identified)"

    return {
        "status": "ok",
        "count": result.count,
        "cached": result.cached,
        "skipped": result.skipped,
        "failed": result.failed,
        "dedup_skipped": result.dedup_skipped,
        "elapsed": round(elapsed, 2),
        "message": summary,
    }


def wiki_index(
    wiki_dir: str | None = None,
    embed_model: str | None = None,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Rebuild the zvec vector index for semantic search."""
    vp = vault_paths(wiki_dir)
    cli_overrides = build_cli_overrides()
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=cli_overrides, config_file=config_file)
    output_dir = vp.output_dir

    wiki = load_wiki_from_files(vp.output_dir)
    if wiki is None:
        return {"status": "error", "message": f"No wiki pages found in {output_dir}. Run 'wiki_compile' first."}

    resolved_model = embed_model or cfg.kompend.compile.embedding.model
    if not resolved_model:
        return {"status": "error", "message": "No embedding model configured. Set compile.embedding.model or pass embed_model."}

    embed_attachments = cfg.kompend.attachments.embed

    try:
        count = build_vector_index(
            wiki=wiki,
            output_dir=output_dir,
            embed_model=resolved_model,
            base_dir=vp.base_dir,
            embed_attachments=embed_attachments,
        )
    except Exception as exc:
        logger.exception("Index build failed")
        return {"status": "error", "message": str(exc)}

    wiki.metadata[str(MetadataField.EMBEDDING_MODEL)] = resolved_model
    # Write the model through the snapshot wiki: the files-loaded wiki above has
    # no graph metrics (quality score, WiCER iteration), so caching it would
    # reset them to zero.
    cached = load_cached_wiki(vp.base_dir)
    if cached is not None:
        cached.metadata[str(MetadataField.EMBEDDING_MODEL)] = resolved_model
        write_wiki_cache(cached, vp.base_dir)

    if count > 0:
        return {"status": "ok", "count": count, "message": f"Indexed {count} nodes in zvec store for semantic search."}
    return {"status": "ok", "count": 0, "message": "Warning: no nodes indexed — vector index is empty."}


def wiki_write(
    wiki_dir: str | None = None,
    page_title: str = "",
    source: list[str] | None = None,
    frontmatter: list[str] | None = None,
) -> dict[str, Any]:
    """Create a wiki page synthesising content from one or more sources."""
    vp = vault_paths(wiki_dir)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())
    base_dir = vp.base_dir

    source_texts: list[tuple[str, str]] = []
    if source:
        source_files = resolve_files(*source, sources_dir=vp.sources_dir)
        for sf in source_files:
            stem = sf.stem
            try:
                cached = source_metadata.cached_text(vp.cache_dir, sf.name)
                text = cached if cached is not None else sf.read_text(encoding="utf-8", errors="replace")
                source_texts.append((stem, text.strip()))
            except Exception as exc:
                logger.warning("could not read %s: %s", sf, exc)

    related_titles = _load_related_titles(base_dir, source_texts)
    related_titles = related_titles[: cfg.kompend.display.related_titles]

    fm: dict[str, Any] = {}
    if frontmatter:
        for kv in frontmatter:
            if "=" in kv:
                k, v = kv.split("=", 1)
                fm[k.strip()] = v.strip()

    compiled_at = iso_local()
    fm.setdefault(OkfField.TITLE, page_title)
    fm.setdefault(OkfField.KIND, NodeType.PAGE.value)
    fm.setdefault(OkfField.COMPILED_AT, compiled_at)
    if source_texts:
        fm.setdefault(OkfField.SOURCES, [s for s, _ in source_texts])

    if not source_texts:
        body = "_(Content pending — run with source to populate)_"
    else:
        llm_model = cfg.kompend.compile.llm.default_model
        body = _synthesize_write_page(
            page_title=page_title,
            source_texts=source_texts,
            related_titles=related_titles,
            model=llm_model,
            timeout=float(cfg.kompend.compile.llm.timeout),
            source_char_max=cfg.kompend.compile.budget.capture_source_char_max,
            llm=cfg.kompend.compile.llm,
            budget=cfg.kompend.compile.budget,
        )

    slug = slugify(page_title)
    write_dir = vp.output_dir / "write"
    write_dir.mkdir(parents=True, exist_ok=True)
    output_file = write_dir / f"{slug}.md"

    writer = MarkdownWriter(description_max_chars=cfg.kompend.display.description_max_chars)
    doc = Document(title=page_title, body=body, frontmatter=fm, node_type=NodeType.PAGE)
    rendered = writer.renderer.render(doc)
    output_file.write_text(rendered, encoding="utf-8")

    return {"status": "ok", "file": str(output_file), "message": f"Written: {output_file}"}


def wiki_organize(
    wiki_dir: str | None = None,
    max_depth: int | None = None,
    min_subcategories: int | None = None,
    collapse_single_file: bool | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Categorise files in sources/ into a hierarchical directory tree."""
    vp = vault_paths(wiki_dir)
    overrides: dict[str, object] = {}
    if max_depth is not None:
        overrides["organize.max_depth"] = max_depth
    if min_subcategories is not None:
        overrides["organize.min_subcategories"] = min_subcategories
    if collapse_single_file is not None:
        overrides["organize.collapse_single_file"] = collapse_single_file

    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=overrides or None)
    sources_dir = vp.sources_dir
    output_dir = vp.output_dir
    base_dir = vp.base_dir

    config = resolve_organize_config(overrides or None)

    timeout = cfg.kompend.compile.llm.timeout
    default_model = cfg.kompend.compile.llm.default_model
    if not default_model:
        return {"status": "error", "message": "No LLM model configured. Set compile.llm.default_model in config."}

    categorize_agent = CategorizeAgent(
        default_model,
        timeout=float(timeout),
        llm=cfg.kompend.compile.llm,
        budget=cfg.kompend.compile.budget,
    )

    try:
        result = organize_sources(
            base_dir=base_dir,
            sources_dir=sources_dir,
            output_dir=output_dir,
            categorize_agent=categorize_agent,
            config=config,
            dry_run=dry_run,
        )
    except Exception as exc:
        logger.exception("Organize failed")
        return {"status": "error", "message": str(exc)}

    return {
        "status": "ok",
        "dry_run": dry_run,
        "categories": len(result.tree.categories),
        "moved_count": result.moved_count,
        "unfiled": len(result.tree.unfiled),
        "files_without_summary": len(result.files_without_summary),
        "link_updates": result.link_updates,
    }


# ── Helpers (mirrors kompend.cli.cmds.knowledge.capture) ─────────────────────


def _load_related_titles(base_dir: Path, source_texts: list[tuple[str, str]]) -> list[str]:
    related: list[str] = []
    if not source_texts:
        return related
    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        return related
    source_stems = {s for s, _ in source_texts}
    for node in wiki.nodes:
        prov = meta_str_list((node.metadata or {}).get(MetadataField.PROVENANCE))
        node_sources = {Path(p).stem for p in prov if p}
        if node_sources & source_stems:
            related.append(str(node.title))
    return related


def _synthesize_write_page(
    *,
    page_title: str,
    source_texts: list[tuple[str, str]],
    related_titles: list[str],
    model: str | None,
    timeout: float,
    source_char_max: int,
    llm: LLMConfig | None = None,
    budget: BudgetConfig | None = None,
) -> str:
    """Synthesize the sources into the body of a write page.

    A source longer than ``source_char_max`` characters is synthesized in
    several calls — the budget sizes one call, it never shortens the
    source. The per-call results are joined; a failing call keeps the
    results of the calls that succeeded before it.
    """
    references_section = _build_references_section(source_texts, related_titles)

    if not model:
        raise RuntimeError("No LLM model configured for write-page synthesis")

    agent = WriteAgent(
        model,
        timeout=timeout,
        retries=llm.output_retries if llm and llm.output_retries_enabled else 0,
        llm=llm,
        budget=budget,
    )
    loader = get_template_loader()
    pages: list[str] = []
    for block in _source_blocks(source_texts, source_char_max):
        prompt = loader.render(
            "capture/write_prompt.md.j2",
            page_title=page_title,
            source_block=block,
        )
        deps = PipelineAgentDeps(text=prompt, source_id="write", timeout=timeout)
        try:
            content = strip_reasoning(agent.run(deps))
        except Exception as exc:
            if not pages:
                raise RuntimeError(f"LLM synthesis failed for write page: {exc}") from exc
            logger.warning("Write-page synthesis stopped after %d block(s): %s", len(pages), exc)
            break
        if content.strip():
            pages.append(content.strip())
    if not pages:
        raise RuntimeError("LLM returned empty content for write page")
    return "\n\n".join(pages) + "\n\n" + references_section


def _source_blocks(source_texts: list[tuple[str, str]], source_char_max: int) -> list[str]:
    """Return one prompt block per source window.

    The sources are never shortened — a long source contributes several
    blocks, each holding the next window of its text.
    """
    blocks: list[str] = []
    for stem, text in source_texts:
        for window in split_text_windows(text, source_char_max):
            blocks.append(f"### {stem}\n\n{window}")
    return blocks


def _build_references_section(source_texts: list[tuple[str, str]], related_titles: list[str]) -> str:
    lines = ["## References\n"]
    for stem, _ in source_texts:
        lines.append(f"- {stem}")
    if related_titles:
        lines.append("\n## Related\n")
        for title in related_titles:
            lines.append(f"- [[{title}]]")
    return "\n".join(lines)
