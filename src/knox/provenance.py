"""N4 claim-level provenance: ``[source.pdf:42-58]`` citations.

Provenance lives in two places:

- :class:`WikiNode.metadata["provenance"]` — a list of source labels per node.
- Inline markers inside ``WikiNode.content`` — parsed by :func:`extract_citations`
  and exposed via :func:`knox provenance <slug>`.

The format is intentionally simple and human-readable so it can survive
a round-trip through the markdown output writer. The per-node record is
persisted in the ``provenance`` table of ``.knox/knox.db``; the legacy
``.knox/provenance/`` directory is no longer written or read.

Note: ``Citation``, ``extract_citations``, ``merge_duplicate_citations``,
and lifecycle constants moved to :mod:`knox.core.models`. Import from there.
"""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants;
# no runtime value ever enters an identifier or statement.

from __future__ import annotations

import json
from pathlib import Path

from knox.core.models import (
    CompiledWiki,
    WikiNode,
    extract_citations,
)
from knox.core.store import open_store, vault_db_path
from knox.core.store.schema import PROVENANCE_TABLE
from knox.logging import get_logger

logger = get_logger(__name__)


def annotate_wiki_with_provenance(wiki: CompiledWiki, base_dir: Path) -> CompiledWiki:
    """Persist provenance citations for each node and return a new wiki.

    Upserts one ``provenance`` row per node (id, title, citations) in the vault
    store. The wiki itself is returned with updated metadata so callers can
    chain output steps.
    """
    rows: list[tuple[str, str, str]] = []
    new_nodes: list[WikiNode] = []
    for node in wiki.nodes:
        citations = extract_citations(node.content)
        rows.append((node.id, node.title, json.dumps([c.to_marker() for c in citations], ensure_ascii=False)))
        new_nodes.append(
            WikiNode(
                id=node.id,
                title=node.title,
                content=node.content,
                node_type=node.node_type,
                source_chunks=node.source_chunks,
                metadata={**(node.metadata or {}), "citation_count": len(citations)},
            )
        )

    with open_store(vault_db_path(base_dir)) as conn:
        conn.executemany(
            f"INSERT INTO {PROVENANCE_TABLE}(node_id, title, citations) VALUES (?, ?, ?) "
            f"ON CONFLICT(node_id) DO UPDATE SET title = excluded.title, citations = excluded.citations",
            rows,
        )

    return CompiledWiki(
        nodes=new_nodes,
        edges=list(wiki.edges),
        toc=list(wiki.toc),
        quality_score=wiki.quality_score,
        iteration=wiki.iteration,
        metadata={**wiki.metadata, "provenance_annotated": True},
    )
