"""Wiki cache serialization.

Single source of truth for writing ``.knox/last_wiki.json`` — kept here so
that pure-core modules (e.g. ``knox.lint.flow``) can persist wiki state
without depending on ``knox.cli._post_compile``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from knox.core.models import CompiledWiki, MetadataField, WikiField
from knox.core.paths import VaultPaths


def write_wiki_cache(wiki: CompiledWiki, base_dir: Path) -> None:
    """Serialize the compiled wiki to ``.knox/last_wiki.json``."""

    def _relative(p: str) -> str:
        try:
            return str(Path(p).relative_to(base_dir))
        except ValueError:
            return p

    paths = VaultPaths(base_dir)
    cache_file = paths.wiki_cache_file
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    nodes_dicts: list[dict[str, Any]] = []
    for n in wiki.nodes:
        nd = dataclasses.asdict(n)
        meta = nd.get("metadata", {})
        if MetadataField.SOURCE_FILE in meta:
            meta = dict(meta)
            meta[MetadataField.SOURCE_FILE] = _relative(meta[MetadataField.SOURCE_FILE])
            nd["metadata"] = meta
        nodes_dicts.append(nd)
    data = {
        WikiField.NODES: nodes_dicts,
        WikiField.EDGES: list(wiki.edges),
        WikiField.TOC: list(wiki.toc),
        WikiField.QUALITY_SCORE: wiki.quality_score,
        WikiField.ITERATION: wiki.iteration,
        "metadata": dict(wiki.metadata),
    }
    cache_file.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
