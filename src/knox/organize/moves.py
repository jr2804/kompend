"""File moves and wiki-link updates for ``knox organize``."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from knox.core.models import MetadataField, WikiField
from knox.core.paths import VaultPaths
from knox.logging import get_logger
from knox.organize.categorize import CategoryNode, CategoryTree

logger = get_logger(__name__)

_CITATION_RE = re.compile(r"\^\[([^\]\n:]+?):(\d+)-(\d+)\]")


# ── Data ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FileMove:
    """One planned or executed file move (paths relative to sources_dir)."""

    old: str
    new: str


@dataclass(frozen=True)
class LinkRemap:
    """old_filename -> new_filename mapping for wiki link updates."""

    mapping: dict[str, str]

    def get(self, old: str) -> str | None:
        return self.mapping.get(old)


# ── File operations ─────────────────────────────────────────────────


def flatten_sources(sources_dir: Path) -> list[FileMove]:
    """Move every nested file to the top level of sources_dir."""
    sources_dir = sources_dir.resolve()
    if not sources_dir.is_dir():
        return []
    moves: list[FileMove] = []
    for path in sorted(sources_dir.rglob("*")):
        if not path.is_file() or path.parent == sources_dir:
            continue
        rel = path.relative_to(sources_dir)
        target = _unique_target(sources_dir, path.name)
        target_rel = target.relative_to(sources_dir)
        shutil.move(str(path), str(target))
        moves.append(FileMove(old=str(rel).replace("\\", "/"), new=str(target_rel).replace("\\", "/")))
    return moves


def _unique_target(sources_dir: Path, name: str) -> Path:
    candidate = sources_dir / name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for idx in range(2, 10_000):
        candidate = sources_dir / f"{stem}-{idx}{suffix}"
        if not candidate.exists():
            return candidate
    msg = f"Could not find a unique name for {name} under {sources_dir}"
    raise FileExistsError(msg)


def materialize_tree(sources_dir: Path, tree: CategoryTree) -> None:
    """Create the file-tree directories implied by the category tree."""
    sources_dir = sources_dir.resolve()
    for cat in tree.categories:
        _materialize_node(sources_dir / cat.name, cat)


def _materialize_node(current: Path, node: CategoryNode) -> None:
    has_content = bool(node.files) or any(c.files or c.subcategories for c in node.subcategories)
    if not has_content:
        return
    current.mkdir(parents=False, exist_ok=True)
    for child in node.subcategories:
        _materialize_node(current / child.name, child)


def move_files(sources_dir: Path, tree: CategoryTree) -> list[FileMove]:
    """Move files to their category directories. Returns old -> new mapping."""
    sources_dir = sources_dir.resolve()
    moves: list[FileMove] = []
    for cat in tree.categories:
        moves.extend(_move_node(sources_dir / cat.name, cat, prefix=""))
    return moves


def _move_node(target_dir: Path, node: CategoryNode, *, prefix: str) -> list[FileMove]:
    moves: list[FileMove] = []
    for filename in node.files:
        src = target_dir.parent / filename
        if not src.is_file():
            logger.warning("File in tree but missing on disk: %s", filename)
            continue
        new_rel = f"{prefix}{node.name}/{filename}" if prefix else f"{node.name}/{filename}"
        new_abs = target_dir.parent / new_rel
        if new_abs.exists() and new_abs.resolve() != src.resolve():
            p = Path(filename)
            for idx in range(2, 10_000):
                candidate = f"{p.stem}-{idx}{p.suffix}"
                candidate_rel = f"{prefix}{node.name}/{candidate}" if prefix else f"{node.name}/{candidate}"
                if not (target_dir.parent / candidate_rel).exists():
                    new_rel, new_abs = candidate_rel, target_dir.parent / candidate_rel
                    break
        new_abs.parent.mkdir(parents=True, exist_ok=True)
        if new_abs.resolve() != src.resolve():
            shutil.move(str(src), str(new_abs))
        moves.append(FileMove(old=filename, new=new_rel))
    for child in node.subcategories:
        moves.extend(_move_node(target_dir / child.name, child, prefix=f"{prefix}{node.name}/" if prefix else f"{node.name}/"))
    return moves


# ── Link updates ────────────────────────────────────────────────────


def build_link_remap(moves: list[tuple[str, str]]) -> LinkRemap:
    """Reduce (old_rel, new_rel) moves to a filename-level remap."""
    mapping: dict[str, str] = {}
    for old, new in moves:
        old_name = Path(old).name
        new_name = Path(new).name
        if old_name == new_name or old_name in mapping:
            continue
        mapping[old_name] = new_name
    return LinkRemap(mapping=mapping)


def update_wiki_links(base_dir: Path, output_dir: Path, remap: LinkRemap) -> dict[str, int]:
    """Update wiki content, provenance and tier state to match new paths."""
    counts = {"wiki_files": 0, "provenance_files": 0, "wiki_node_metadata": 0, "tier_state": 0}
    if not remap.mapping:
        return counts
    paths = VaultPaths(base_dir)
    if output_dir.is_dir():
        for fpath in output_dir.rglob("*.md"):
            counts["wiki_files"] += _rewrite_citations(fpath, remap)
    if (prov_dir := paths.provenance_dir).is_dir():
        for fpath in prov_dir.glob("*.json"):
            counts["provenance_files"] += _rewrite_provenance(fpath, remap)
    counts["wiki_node_metadata"] = _rewrite_wiki_node_paths(output_dir, remap)
    counts["tier_state"] = _rewrite_tier_state(paths.compile_state_path, remap, output_dir)
    return counts


def _rewrite_citations(filepath: Path, remap: LinkRemap) -> int:
    try:
        text = filepath.read_text(encoding="utf-8")
    except OSError:
        return 0
    edits = 0

    def _replacer(m: re.Match[str]) -> str:
        nonlocal edits
        s = m.group(1)
        if (n := remap.get(s)) and n != s:
            edits += 1
            return f"^[{n}:{m.group(2)}-{m.group(3)}]"
        return m.group(0)

    updated = _CITATION_RE.sub(_replacer, text)
    if edits:
        filepath.write_text(updated, encoding="utf-8")
    return edits


def _rewrite_provenance(filepath: Path, remap: LinkRemap) -> int:
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(data, dict):
        return 0
    citations = data.get("citations", [])
    if not isinstance(citations, list):
        return 0
    edits = 0
    for i, marker in enumerate(citations):
        if not isinstance(marker, str):
            continue
        m = _CITATION_RE.match(marker)
        if not m or not (new := remap.get(m.group(1))):
            continue
        citations[i] = f"^[{new}:{m.group(2)}-{m.group(3)}]"
        edits += 1
    if edits:
        filepath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return edits


def _rewrite_tier_state(filepath: Path, remap: LinkRemap, wiki_root: Path) -> int:
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(data, dict):
        return 0
    entries = data.get("entries")
    if not isinstance(entries, dict):
        return 0
    edits = 0
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        source = entry.get(MetadataField.SOURCE_FILE)
        if not isinstance(source, str):
            continue
        new_source = _remap_path(source, remap)
        if new_source != source:
            entry[MetadataField.SOURCE_FILE] = new_source
            edits += 1
    if edits:
        filepath.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return edits


def _rewrite_wiki_node_paths(wiki_root: Path, remap: LinkRemap) -> int:
    """Rewrite source_file metadata in wiki_cache.json."""
    cache = VaultPaths(wiki_root.parent).wiki_cache_path
    if not cache.is_file():
        return 0
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(data, dict):
        return 0
    nodes = data.get(WikiField.NODES, [])
    if not isinstance(nodes, list):
        return 0
    edits = 0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        meta = node.get(WikiField.METADATA)
        if not isinstance(meta, dict):
            continue
        source = meta.get(MetadataField.SOURCE_FILE)
        if not isinstance(source, str):
            continue
        new = _remap_path(source, remap)
        if new != source:
            meta[MetadataField.SOURCE_FILE] = new
            edits += 1
    if edits:
        cache.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return edits


def _remap_path(value: str, remap: LinkRemap) -> str:
    """Replace filename portion of path using remap."""
    parsed = Path(value.replace("\\", "/"))
    new_name = remap.get(parsed.name)
    if new_name is None or new_name == parsed.name:
        return value
    return str(parsed.with_name(new_name)).replace("\\", "/")
