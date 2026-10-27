"""Source management MCP tools."""

from pathlib import Path
from typing import Any

from kompend.cli._shared import build_cli_overrides
from kompend.core import source_metadata
from kompend.core.config import resolve_config
from kompend.core.keys import EXPORT_ANNOTATIONS, EXPORT_SUMMARY
from kompend.core.paths import EXPORTED_DIR_NAME, INDEX_FILE, safe_stem, vault_paths
from kompend.core.pipeline.export import export_source
from kompend.core.utils.dicts import as_str
from kompend.logging import get_logger
from kompend.output.pages.index import sync_exported_sources_section

logger = get_logger(__name__)


def sources_list(
    wiki_dir: str | None = None,
    tags: list[str] | None = None,
    config_file: str | None = None,
) -> dict[str, Any]:
    """List ingested source files with their tags.

    Optionally filter by tag using --tags.
    """
    vp = vault_paths(wiki_dir)
    resolve_config(base_dir=vp.base_dir, config_file=config_file)
    cache_dir = vp.cache_dir

    cache_data = source_metadata.load_source_entries(cache_dir)
    if not cache_data:
        return {"status": "ok", "sources": [], "count": 0, "message": "No sources found. Run 'wiki_ingest' first."}

    # Parse tag filter groups (AND within group, OR between groups)
    tag_groups = []
    if tags:
        for value in tags:
            group = [t.strip() for t in value.split(",") if t.strip()]
            if group:
                tag_groups.append(group)

    results = []
    for file in sorted(cache_data):
        source_tags = source_metadata.source_tags(cache_data[file])

        if tag_groups:
            tag_set = set(source_tags)
            if not any(all(t in tag_set for t in group) for group in tag_groups):
                continue

        results.append({"file": file, "tags": source_tags})  # nocheck: MCP response payload

    if not results:
        return {"status": "ok", "sources": [], "count": 0, "message": "No sources match the given tag filter."}

    return {"status": "ok", "sources": results, "count": len(results)}


def sources_tag(
    wiki_dir: str | None = None,
    file: str = "",
    tag: str = "",
    config_file: str | None = None,
) -> dict[str, Any]:
    """Add a tag to an ingested source file."""
    vp = vault_paths(wiki_dir)
    resolve_config(base_dir=vp.base_dir, config_file=config_file)
    return _mutate_tags(vp.cache_dir, file, tag, add=True)


def sources_untag(
    wiki_dir: str | None = None,
    file: str = "",
    tag: str = "",
    config_file: str | None = None,
) -> dict[str, Any]:
    """Remove a tag from an ingested source file."""
    vp = vault_paths(wiki_dir)
    resolve_config(base_dir=vp.base_dir, config_file=config_file)
    return _mutate_tags(vp.cache_dir, file, tag, add=False)


def sources_attr(
    wiki_dir: str | None = None,
    file: str = "",
    key_value: str = "",
    config_file: str | None = None,
) -> dict[str, Any]:
    """Set a key=value attribute on an ingested source file."""
    vp = vault_paths(wiki_dir)
    resolve_config(base_dir=vp.base_dir, config_file=config_file)
    return _mutate_attributes(vp.cache_dir, file, key_value, add=True)


def sources_attr_rm(
    wiki_dir: str | None = None,
    file: str = "",
    key: str = "",
    config_file: str | None = None,
) -> dict[str, Any]:
    """Remove an attribute key from an ingested source file."""
    vp = vault_paths(wiki_dir)
    resolve_config(base_dir=vp.base_dir, config_file=config_file)
    return _mutate_attributes(vp.cache_dir, file, key, add=False)


def sources_export(
    wiki_dir: str | None = None,
    source_id: str = "",
    export_all: bool = False,
    output: str = "",
    inline: bool = False,
    annotations: bool | None = None,
    summary: bool | None = None,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Render an ingested source as clean markdown.

    OUTPUT is optional — omit it to write to the vault's ``exported/``
    directory (reference material, outside wiki/); pass a path to export
    to any arbitrary file. With EXPORT_ALL=true every ingested source is
    exported into ``exported/`` and the vault-root ``index.md`` is refreshed
    to list them (CLI ``--all`` parity).
    """
    vp = vault_paths(wiki_dir)

    cli_set: dict[str, Any] = {}
    if annotations is not None:
        cli_set[EXPORT_ANNOTATIONS] = annotations
    if summary is not None:
        cli_set[EXPORT_SUMMARY] = summary
    cfg = resolve_config(
        base_dir=vp.base_dir,
        config_file=config_file,
        cli_overrides=build_cli_overrides(cli_set=cli_set),
    )
    ann = as_str(cfg.get(EXPORT_ANNOTATIONS)) or "off"
    doc_headings = cfg.kompend.display.doc_headings
    summ = as_str(cfg.get(EXPORT_SUMMARY)) or "off"

    if export_all:
        if source_id:
            return {"status": "error", "message": "SOURCE_ID cannot be combined with export_all=True."}
        if output:
            return {"status": "error", "message": "export_all=True exports into exported/; OUTPUT must be empty."}
        cache_data = source_metadata.load_source_entries(vp.cache_dir)
        if not cache_data:
            return {"status": "ok", "exported": [], "count": 0, "message": "No sources found. Run 'wiki_ingest' first."}
        exported_dir = vp.base_dir / EXPORTED_DIR_NAME
        exported_dir.mkdir(parents=True, exist_ok=True)
        exported: list[str] = []
        skipped: list[str] = []
        for file in sorted(cache_data):
            out_file = exported_dir / f"{safe_stem(file) or 'source'}.md"
            try:
                export_source(
                    base_dir=vp.base_dir,
                    cache_dir=vp.cache_dir,
                    source_id=file,
                    out_path=out_file,
                    inline=inline,
                    annotations=ann,
                    summary=summ,
                    doc_headings=doc_headings,
                )
                exported.append(str(out_file))
            except FileNotFoundError as exc:
                skipped.append(str(exc))
        sync_exported_sources_section(vp.base_dir / INDEX_FILE, vp.base_dir)
        return {"status": "ok", "exported": exported, "count": len(exported), "skipped": skipped}

    if not source_id:
        return {"status": "error", "message": "Pass a source_id or use export_all=True."}
    if output:
        output_file = Path(output)
    else:
        exported_dir = vp.base_dir / EXPORTED_DIR_NAME
        output_file = exported_dir / f"{safe_stem(source_id) or 'source'}.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.suffix.lower() != ".md":
        return {"status": "error", "message": f"Output file must end in .md, got '{output_file.suffix or '<none>'}'"}

    try:
        export_source(
            base_dir=vp.base_dir,
            cache_dir=vp.cache_dir,
            source_id=source_id,
            out_path=output_file,
            inline=inline,
            annotations=ann,
            summary=summ,
            doc_headings=doc_headings,
        )
    except FileNotFoundError as exc:
        return {"status": "error", "message": str(exc)}
    if not output:
        sync_exported_sources_section(vp.base_dir / INDEX_FILE, vp.base_dir)
    return {"status": "ok", "message": f"Exported {source_id} -> {output_file}"}


# ── Helpers ─────────────────────────────────────────────────────────────


def _mutate_tags(cache_dir: Path, file: str, tag: str, add: bool) -> dict[str, Any]:
    """Add or remove *tag* on the source's parse-cache record."""
    try:
        result = source_metadata.apply_tag(cache_dir, file, tag, add=add)
    except LookupError:
        return {"status": "error", "message": f"No cache entry found for '{file}'. Run 'wiki_ingest' first."}
    return {"status": "ok", "message": result.message, "changed": result.changed}


def _mutate_attributes(cache_dir: Path, file: str, key_value: str, add: bool) -> dict[str, Any]:
    """Add or remove a key=value attribute on the source's parse-cache record."""
    try:
        result = source_metadata.apply_attribute(cache_dir, file, key_value, add=add)
    except LookupError:
        return {"status": "error", "message": f"No cache entry found for '{file}'. Run 'wiki_ingest' first."}
    except ValueError:
        return {"status": "error", "message": f"Use KEY=VALUE format for attributes, got: '{key_value}'"}
    return {"status": "ok", "message": result.message, "changed": result.changed}
