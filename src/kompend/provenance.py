"""N4 claim-level provenance: ``[source.pdf:42-58]`` citations.

Provenance lives in two places:

- :class:`WikiNode.metadata["provenance"]` — a list of source labels per node.
- Inline markers inside ``WikiNode.content`` — parsed by :func:`extract_citations`
  and exposed via :func:`kompend provenance <slug>`.

The format is intentionally simple and human-readable so it can survive
a round-trip through the markdown output writer. The per-node record is
persisted in the ``provenance`` table of ``.kompend/kompend.db``.

.. note::
   ``Citation``, ``extract_citations``, ``merge_duplicate_citations`` and the
   lifecycle constants live in :mod:`kompend.core.models`; import them there.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from kompend.core.models import (
    CompiledWiki,
    WikiNode,
    extract_citations,
)
from kompend.core.store import core_connection, open_store, vault_db_path
from kompend.core.store.tables import provenance
from kompend.logging import get_logger

logger = get_logger(__name__)


def annotate_wiki_with_provenance(wiki: CompiledWiki, base_dir: Path) -> CompiledWiki:
    """Persist provenance citations for each node and return a new wiki.

    Upserts one ``provenance`` row per node (id, title, citations) in the vault
    store. The wiki itself is returned with updated metadata so callers can
    chain output steps.
    """
    rows: list[dict[str, str]] = []
    new_nodes: list[WikiNode] = []
    for node in wiki.nodes:
        citations = extract_citations(node.content)
        rows.append(
            {
                "node_id": node.id,
                "title": node.title,
                "citations": json.dumps([c.to_marker() for c in citations], ensure_ascii=False),
            }
        )
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

    statement = sqlite_insert(provenance)
    statement = statement.on_conflict_do_update(
        index_elements=[provenance.c.node_id],
        set_={"title": statement.excluded.title, "citations": statement.excluded.citations},
    )
    with open_store(vault_db_path(base_dir)) as conn, core_connection(conn) as core:
        if rows:
            core.execute(statement, rows)

    return CompiledWiki(
        nodes=new_nodes,
        edges=list(wiki.edges),
        toc=list(wiki.toc),
        quality_score=wiki.quality_score,
        iteration=wiki.iteration,
        metadata={**wiki.metadata, "provenance_annotated": True},
    )
