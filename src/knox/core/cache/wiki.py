"""Wiki cache serialization.

Single source of truth for writing the ``last`` wiki snapshot — kept here so
that pure-core modules (e.g. ``knox.lint.flow``) can persist wiki state
without depending on ``knox.cli._post_compile``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from knox.core.models import CompiledWiki, MetadataField, WikiField
from knox.core.models.wiki import WIKI_SNAPSHOT_LAST, write_wiki_snapshot_payload


def write_wiki_cache(wiki: CompiledWiki, base_dir: Path) -> None:
    """Serialize the compiled wiki to the ``last`` snapshot in ``knox.db``."""

    def _relative(raw: str) -> str:
        try:
            return str(Path(raw).relative_to(base_dir))
        except ValueError:
            return raw

    nodes_dicts: list[dict[str, Any]] = []
    for node in wiki.nodes:
        node_dict = dataclasses.asdict(node)
        meta = node_dict.get(WikiField.METADATA, {})
        if MetadataField.SOURCE_FILE in meta:
            meta = dict(meta)
            meta[MetadataField.SOURCE_FILE] = _relative(str(meta[MetadataField.SOURCE_FILE]))
            node_dict[WikiField.METADATA] = meta
        nodes_dicts.append(node_dict)
    write_wiki_snapshot_payload(
        base_dir,
        WIKI_SNAPSHOT_LAST,
        nodes=nodes_dicts,
        edges=list(wiki.edges),
        toc=list(wiki.toc),
        quality_score=wiki.quality_score,
        iteration=wiki.iteration,
        metadata=dict(wiki.metadata),
    )
