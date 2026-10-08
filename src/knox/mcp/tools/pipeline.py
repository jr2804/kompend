"""Pipeline MCP tools (ingest, index, write, organize).

Compile/parse already live in knowledge.py (wiki_compile/wiki_parse).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from knox.agents.categorize import CategorizeAgent
from knox.agents.deps import PipelineAgentDeps
from knox.agents.write import WriteAgent
from knox.cli._pipeline_helpers import resolve_compile_inputs
from knox.cli._post_compile import write_wiki_cache
from knox.cli._shared import build_cli_overrides
from knox.cli._sources import resolve_files
from knox.core import source_metadata
from knox.core.config import resolve_config
from knox.core.models import MetadataField, load_cached_wiki, load_wiki_from_files
from knox.core.paths import VaultPaths, vault_paths
from knox.core.pipeline.ingest import ingest_documents
from knox.core.templates import get_template_loader
from knox.core.text.slug import slugify
from knox.core.trust.actor import resolve_actor
from knox.core.utils.time import iso_local
from knox.llm.format import strip_reasoning
from knox.logging import get_logger
from knox.organize.engine import organize_sources, resolve_organize_config
from knox.output.markdown import MarkdownWriter
from knox.output.models import Document
from knox.preprocessing.parsers._registry import ParserBuildOptions, build_default_router
from knox.rag import build_vector_index

logger = get_logger(__name__)


def wiki_ingest(
    wiki_dir: str | None = None,
    files: list[str] | None = None,
    online_urls: list[str] | None = None,
    url_file: str | None = None,
    recursive: bool = False,
    ignore: list[str] | None = None,
    no_cache: bool = False,
    workers: int = 1,
    in_place: bool = False,
    parser: str | None = None,
    tags: list[str] | None = None,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Ingest documents into the knowledge graph without compiling.

    Parses every source and caches the structured documents in .knox/cache/.
    Run wiki_compile afterwards to chunk, resolve, and compile.
    """
    online_urls = online_urls or []
    vp = vault_paths(wiki_dir)
    cli_overrides = build_cli_overrides(
        no_cache=no_cache,
        workers=workers,
        in_place_sources=in_place,
        parser=parser,
    )
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=cli_overrides, config_file=config_file)
    sources_dir = vp.sources_dir
    base_dir = vp.base_dir

    all_input_files = resolve_compile_inputs(
        files or [],
        online_urls,
        Path(url_file) if url_file else None,
        sources_dir,
        recursive,
        ignore,
        in_place=cfg.knox.compile.in_place_sources,
        undetected=cfg.knox.downloader.undetected,
    )
    if not all_input_files:
        return {"status": "error", "message": f"No supported files found in {sources_dir}."}

    images_dir = VaultPaths(base_dir).images_dir
    parser_name = cfg.knox.ingest.parser
    router = build_default_router(
        ParserBuildOptions.from_config(cfg, images_dir=images_dir),
        name=parser_name,
    )

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
            rumdl_config=cfg.knox.compile.output.rumdl_config,
            flavor=cfg.knox.compile.output.flavor,
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

    resolved_model = embed_model or cfg.knox.compile.embedding.model
    if not resolved_model:
        return {"status": "error", "message": "No embedding model configured. Set compile.embedding.model or pass embed_model."}

    embed_attachments = cfg.knox.attachments.embed

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
    related_titles = related_titles[: cfg.knox.display.related_titles]

    fm: dict[str, Any] = {}
    if frontmatter:
        for kv in frontmatter:
            if "=" in kv:
                k, v = kv.split("=", 1)
                fm[k.strip()] = v.strip()

    compiled_at = iso_local()
    fm.setdefault("title", page_title)
    fm.setdefault("kind", "page")
    fm.setdefault("compiled_at", compiled_at)
    if source_texts:
        fm.setdefault("sources", [s for s, _ in source_texts])

    if not source_texts:
        body = "_(Content pending — run with source to populate)_"
    else:
        llm_model = cfg.knox.compile.llm.default_model
        body = _synthesize_write_page(
            page_title=page_title,
            source_texts=source_texts,
            related_titles=related_titles,
            model=llm_model,
            timeout=float(cfg.knox.compile.llm.timeout),
            source_char_max=cfg.knox.compile.budget.capture_source_char_max,
        )

    slug = slugify(page_title)
    write_dir = vp.output_dir / "write"
    write_dir.mkdir(parents=True, exist_ok=True)
    output_file = write_dir / f"{slug}.md"

    writer = MarkdownWriter()
    doc = Document(title=page_title, body=body, frontmatter=fm, node_type="page")
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

    timeout = cfg.knox.compile.llm.timeout
    default_model = cfg.knox.compile.llm.default_model
    if not default_model:
        return {"status": "error", "message": "No LLM model configured. Set compile.llm.default_model in config."}

    categorize_agent = CategorizeAgent(
        default_model,
        timeout=float(timeout) if timeout else 120,
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


# ── Helpers (mirrors knox.cli.cmds.knowledge.capture) ─────────────────────


def _load_related_titles(base_dir: Path, source_texts: list[tuple[str, str]]) -> list[str]:
    related: list[str] = []
    if not source_texts:
        return related
    wiki = load_cached_wiki(base_dir)
    if wiki is None:
        return related
    source_stems = {s for s, _ in source_texts}
    for node in wiki.nodes:
        prov = (node.metadata or {}).get(MetadataField.PROVENANCE) or []
        node_sources = {Path(str(p)).stem for p in prov if p}
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
    source_char_max: int = 4000,
) -> str:
    references_section = _build_references_section(source_texts, related_titles)

    if not model:
        raise RuntimeError("No LLM model configured for write-page synthesis")

    agent = WriteAgent(model, timeout=timeout, retries=2)
    source_block = "\n\n".join(f"### {stem}\n\n{text[:source_char_max]}" for stem, text in source_texts)
    loader = get_template_loader()
    prompt = loader.render(
        "capture/write_prompt.md.j2",
        page_title=page_title,
        source_block=source_block,
    )
    deps = PipelineAgentDeps(text=prompt, source_id="write", timeout=timeout)
    try:
        content = agent.run(deps)
        content = strip_reasoning(content)
        if content.strip():
            return content.strip() + "\n\n" + references_section
        raise RuntimeError("LLM returned empty content for write page")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"LLM synthesis failed for write page: {exc}") from exc


def _build_references_section(source_texts: list[tuple[str, str]], related_titles: list[str]) -> str:
    lines = ["## References\n"]
    for stem, _ in source_texts:
        lines.append(f"- {stem}")
    if related_titles:
        lines.append("\n## Related\n")
        for title in related_titles:
            lines.append(f"- [[{title}]]")
    return "\n".join(lines)
