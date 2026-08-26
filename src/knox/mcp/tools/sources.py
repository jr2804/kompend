"""Source management MCP tools."""

import json
from pathlib import Path
from typing import Any

from knox.cli._shared import build_cli_overrides
from knox.core.config import resolve_config
from knox.core.config.keys import EXPORT_ANNOTATIONS, EXPORT_SUMMARY
from knox.core.models import MetadataField
from knox.core.paths import EXPORTED_DIR_NAME, vault_paths
from knox.core.pipeline.export import export_source
from knox.core.pipeline.ingest import parse_cache_glob
from knox.core.utils.dicts import as_str
from knox.logging import get_logger

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

    cache_data = _load_all_cache_entries(cache_dir)
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
        source_tags = _get_source_tags(file, cache_data)

        if tag_groups:
            tag_set = set(source_tags)
            if not any(all(t in tag_set for t in group) for group in tag_groups):
                continue

        results.append({"file": file, "tags": source_tags})

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
    _mutate_attributes(vp.cache_dir, file, key_value, add=True)
    return {"status": "ok"}


def sources_attr_rm(
    wiki_dir: str | None = None,
    file: str = "",
    key: str = "",
    config_file: str | None = None,
) -> dict[str, Any]:
    """Remove an attribute key from an ingested source file."""
    vp = vault_paths(wiki_dir)
    resolve_config(base_dir=vp.base_dir, config_file=config_file)
    _mutate_attributes(vp.cache_dir, file, key, add=False)
    return {"status": "ok"}


def sources_export(
    wiki_dir: str | None = None,
    source_id: str = "",
    output: str = "",
    inline: bool = False,
    annotations: bool | None = None,
    summary: bool | None = None,
    config_file: str | None = None,
) -> dict[str, Any]:
    """Render an ingested source as clean markdown.

    OUTPUT is optional — omit it to write to the vault's ``exported/``
    directory (reference material, outside wiki/); pass a path to export
    to any arbitrary file.
    """
    vp = vault_paths(wiki_dir)
    if output:
        output_file = Path(output)
    else:
        exported_dir = vp.base_dir / EXPORTED_DIR_NAME
        stem = Path(source_id).stem or "source"
        output_file = exported_dir / f"{stem}.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.suffix.lower() != ".md":
        return {"status": "error", "message": f"Output file must end in .md, got '{output_file.suffix or '<none>'}'"}

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

    try:
        export_source(
            base_dir=vp.base_dir,
            cache_dir=vp.cache_dir,
            source_id=source_id,
            out_path=output_file,
            inline=inline,
            annotations=as_str(cfg.get(EXPORT_ANNOTATIONS)) or "off",
            summary=as_str(cfg.get(EXPORT_SUMMARY)) or "off",
        )
    except FileNotFoundError as exc:
        return {"status": "error", "message": str(exc)}
    return {"status": "ok", "message": f"Exported {source_id} -> {output_file}"}


# ── Helpers ─────────────────────────────────────────────────────────────


def _load_all_cache_entries(cache_dir: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for cache_file in sorted(cache_dir.glob("*.json")):
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            source_file = data.get(MetadataField.SOURCE_FILE, "")
            if source_file:
                entries[source_file] = data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load cache entry %s: %s", cache_file.name, exc)
    return entries


def _get_source_tags(file: str, cache_data: dict[str, Any]) -> list[str]:
    entry = cache_data.get(file, {})
    meta = entry.get("metadata", {})
    return meta.get("tags", [])


def _mutate_tags(cache_dir: Path, file: str, tag: str, add: bool) -> dict[str, Any]:
    cache_file = _find_cache_file(cache_dir, file)
    if cache_file is None:
        return {"status": "error", "message": f"No cache entry found for '{file}'. Run 'wiki_ingest' first."}

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"status": "error", "message": f"Could not read cache entry: {exc}"}

    meta = data.setdefault("metadata", {})
    existing_tags: list[str] = meta.get("tags", [])

    if add:
        if tag in existing_tags:
            return {"status": "ok", "message": f"Tag '{tag}' already present on {file}", "changed": False}
        existing_tags.append(tag)
        action = "Added"
    else:
        if tag not in existing_tags:
            return {"status": "ok", "message": f"Tag '{tag}' not found on {file}", "changed": False}
        existing_tags.remove(tag)
        action = "Removed"

    meta["tags"] = existing_tags

    try:
        cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        return {"status": "error", "message": f"Could not write cache entry: {exc}"}

    return {"status": "ok", "message": f"{action} tag '{tag}' on {file}", "changed": True}


def _mutate_attributes(cache_dir: Path, file: str, key_value: str, add: bool) -> dict[str, Any]:
    """Add or remove a key=value attribute on a source file's cache entry."""
    cache_file = _find_cache_file(cache_dir, file)
    if cache_file is None:
        return {"status": "error", "message": f"No cache entry found for '{file}'. Run 'wiki_ingest' first."}

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"status": "error", "message": f"Could not read cache entry: {exc}"}

    meta = data.setdefault("metadata", {})
    existing: dict[str, str] = meta.get("attributes", {})

    if add:
        if "=" not in key_value:
            return {"status": "error", "message": f"Use KEY=VALUE format for attributes, got: '{key_value}'"}
        key, value = key_value.split("=", 1)
        if existing.get(key) == value:
            return {"status": "ok", "message": f"Attribute '{key}={value}' already present on {file}", "changed": False}
        existing[key] = value
        action = f"Set {key}={value}"
    else:
        if key_value not in existing:
            return {"status": "ok", "message": f"Attribute key '{key_value}' not found on {file}", "changed": False}
        del existing[key_value]
        action = f"Removed {key_value}"

    meta["attributes"] = existing

    try:
        cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        return {"status": "error", "message": f"Could not write cache entry: {exc}"}

    return {"status": "ok", "message": action, "changed": True}


def _find_cache_file(cache_dir: Path, file: str) -> Path | None:
    stem = Path(file).stem
    matches = sorted(cache_dir.glob(parse_cache_glob(stem)))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    for m in matches:
        try:
            data = json.loads(m.read_text(encoding="utf-8"))
            if data.get(MetadataField.SOURCE_FILE) == file:
                return m
        except (json.JSONDecodeError, OSError):
            continue
    return matches[0]
