"""Wiki link updates after a ``knox organize`` run.

The compiled wiki (and the cached provenance records) reference source
files in three different shapes:

1. Inline citations of the form ``^[filename:start-end]`` inside wiki
   page content.
2. Full source paths stored in
   :attr:`knox.core.models.WikiNode.metadata` (key ``source_file``).
3. Source identifiers inside per-node provenance files under
   ``.knox/provenance/<node_id>.json``.

After a move we need a stable ``old → new`` mapping for each of these.
The mapping keys are always the *filename* portion (the path that survived
in the citations and provenance records) rather than absolute paths so the
function is robust against machines with different root locations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from knox.core.models import MetadataField, WikiField
from knox.core.paths import VaultPaths
from knox.logging import get_logger

logger = get_logger(__name__)

_CITATION_RE = re.compile(r"\^\[([^\]\n:]+?):(\d+)-(\d+)\]")


@dataclass(frozen=True)
class LinkRemap:
    """``old_filename -> new_filename`` mapping for wiki link updates."""

    mapping: dict[str, str]

    def get(self, old: str) -> str | None:
        return self.mapping.get(old)


def build_link_remap(moves: list[tuple[str, str]]) -> LinkRemap:
    """Reduce a list of ``(old_rel, new_rel)`` moves to filename-level map.

    Only the filename portion is compared and remapped — wiki links do not
    encode directory depth. If multiple files with the same name were
    moved to different directories we keep the first occurrence and log a
    warning; this is an extreme edge case and the compile cache will catch
    any real collision.
    """
    mapping: dict[str, str] = {}
    for old, new in moves:
        old_name = Path(old).name
        new_name = Path(new).name
        if old_name == new_name:
            continue
        if old_name in mapping and mapping[old_name] != new_name:
            logger.warning("Filename %s mapped to multiple targets; using %s", old_name, mapping[old_name])
            continue
        mapping[old_name] = new_name
    return LinkRemap(mapping=mapping)


def update_wiki_links(
    base_dir: Path,
    output_dir: Path,
    remap: LinkRemap,
) -> dict[str, int]:
    """Update wiki content, provenance and tier state to match new paths.

    Returns a counter dict with the number of edits made in each location
    so the caller can print a meaningful summary.
    """
    counts = {
        "wiki_files": 0,
        "wiki_node_metadata": 0,
        "provenance_files": 0,
        "tier_state": 0,
    }
    if not remap.mapping:
        return counts

    wiki_root = output_dir
    paths = VaultPaths(base_dir)
    if wiki_root.is_dir():
        for path in wiki_root.rglob("*.md"):
            counts["wiki_files"] += _rewrite_inline_citations(path, remap)

    prov_dir = paths.provenance_dir
    if prov_dir.is_dir():
        for path in prov_dir.glob("*.json"):
            counts["provenance_files"] += _rewrite_provenance(path, remap)

    counts["wiki_node_metadata"] = _rewrite_wiki_node_paths(wiki_root, remap)

    state_file = paths.compile_state_path
    if state_file.is_file():
        counts["tier_state"] = _rewrite_tier_state(state_file, remap)

    return counts


def _rewrite_inline_citations(md_file: Path, remap: LinkRemap) -> int:
    try:
        original = md_file.read_text(encoding="utf-8")
    except OSError:
        return 0

    edits = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal edits
        source = match.group(1)
        new_source = remap.get(source)
        if new_source is None or new_source == source:
            return match.group(0)
        edits += 1
        return f"^[{new_source}:{match.group(2)}-{match.group(3)}]"

    updated = _CITATION_RE.sub(_replace, original)
    if edits:
        md_file.write_text(updated, encoding="utf-8")
    return edits


def _rewrite_provenance(prov_file: Path, remap: LinkRemap) -> int:
    try:
        data = json.loads(prov_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(data, dict):
        return 0
    citations = data.get("citations", [])
    if not isinstance(citations, list):
        return 0
    edits = 0
    for idx, marker in enumerate(citations):
        if not isinstance(marker, str):
            continue
        m = _CITATION_RE.match(marker)
        if not m:
            continue
        old_name = m.group(1)
        new_name = remap.get(old_name)
        if new_name is None or new_name == old_name:
            continue
        citations[idx] = f"^[{new_name}:{m.group(2)}-{m.group(3)}]"
        edits += 1
    if edits:
        prov_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return edits


def _rewrite_wiki_node_paths(wiki_root: Path, remap: LinkRemap) -> int:
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
        new_source = _remap_path(source, remap)
        if new_source != source:
            meta[MetadataField.SOURCE_FILE] = new_source
            edits += 1
    if edits:
        cache.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return edits


def _rewrite_tier_state(state_file: Path, remap: LinkRemap) -> int:
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
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
        path_value = entry.get(MetadataField.SOURCE_FILE)
        if not isinstance(path_value, str):
            continue
        new_value = _remap_path(path_value, remap)
        if new_value != path_value:
            entry[MetadataField.SOURCE_FILE] = new_value
            edits += 1
    if edits:
        state_file.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return edits


def _remap_path(value: str, remap: LinkRemap) -> str:
    """Replace the filename portion of *value* using *remap*.

    The wiki tier state stores full paths (``sources/foo/bar.md``) while
    citations use bare filenames. We rewrite the basename and leave the
    directory portion intact, so a move of ``sources/foo/x.md`` to
    ``sources/bar/x.md`` becomes ``sources/bar/x.md``.
    """
    parsed_file = Path(value.replace("\\", "/"))
    new_name = remap.get(parsed_file.name)
    if new_name is None or new_name == parsed_file.name:
        return value
    return str(parsed_file.with_name(new_name)).replace("\\", "/")
