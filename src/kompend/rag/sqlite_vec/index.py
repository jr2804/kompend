"""The write path — embed the indexable nodes and insert them into the store.

One node becomes one ``vec_items`` row, one ``vec_fts`` row, and one row per
tag / attribute token / attribute key in the side tables. The rows are
buffered and flushed every *batch_size* nodes, because the embedding call is
the expensive part and the inserts are not.

Coverage is decided by the caller: :func:`kompend.rag.vector_store.indexable_nodes`
is the single rule, applied in ``build_index`` so the manifest records exactly
what was embedded.
"""

# ruff: noqa: S608 — SQL is built from Final[str] table-name constants only.

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import sqlite_vec

from kompend.core.models import MetadataField, WikiNode
from kompend.core.models.metadata import meta_str_list_dict
from kompend.core.paths import rel_to_vault
from kompend.logging import LOG_SAMPLE_LIMIT, get_logger
from kompend.rag.embedding_base import EmbeddingProvider
from kompend.rag.sqlite_vec.schema import (
    VEC_ATTR_KEYS,
    VEC_ATTRIBUTES,
    VEC_FTS,
    VEC_ITEMS,
    VEC_TAGS,
)
from kompend.rag.vector_store import attributes_to_list, embed_node, embed_text, figure_paths

logger = get_logger(__name__)


def index_nodes(
    conn: sqlite3.Connection,
    provider: EmbeddingProvider,
    nodes: list[WikiNode],
    batch_size: int,
    *,
    preview_len: int,
    db_file: Path,
    base_dir: Path | None = None,
) -> int:
    """Embed and write *nodes* into the store."""
    indexed_count = 0
    figure_nodes = 0
    figure_embedded = 0
    degraded: list[str] = []

    items: list[tuple[Any, ...]] = []
    fts_rows: list[tuple[str, str, str, str, str]] = []
    tag_rows: list[tuple[str, str]] = []
    attribute_rows: list[tuple[str, str]] = []
    attr_key_rows: list[tuple[str, str]] = []

    def flush() -> int:
        """Write the pending rows and clear the buffers."""
        if not items:
            return 0
        count = len(items)
        conn.executemany(
            f"INSERT INTO {VEC_ITEMS} (id, embedding, title, node_type, source_file, content_preview) VALUES (?, ?, ?, ?, ?, ?)",
            items,
        )
        conn.executemany(
            f"INSERT INTO {VEC_FTS} (id, title, content_preview, node_type, source_file) VALUES (?, ?, ?, ?, ?)",
            fts_rows,
        )
        conn.executemany(f"INSERT INTO {VEC_TAGS} (item_id, tag) VALUES (?, ?)", tag_rows)
        conn.executemany(f"INSERT INTO {VEC_ATTRIBUTES} (item_id, token) VALUES (?, ?)", attribute_rows)
        conn.executemany(f"INSERT INTO {VEC_ATTR_KEYS} (item_id, key) VALUES (?, ?)", attr_key_rows)
        items.clear()
        fts_rows.clear()
        tag_rows.clear()
        attribute_rows.clear()
        attr_key_rows.clear()
        return count

    for node in nodes:
        raw_source = str(node.metadata.get(MetadataField.SOURCE, "")) if node.metadata else ""
        source_file = rel_to_vault(raw_source, base_dir) if base_dir else raw_source
        text = embed_text(node)
        vector, note = embed_node(provider, node, text, base_dir=base_dir)
        if figure_paths(node):
            figure_nodes += 1
            if note is None:
                figure_embedded += 1
            else:
                degraded.append(f"{node.id}: {note}")
        attributes = meta_str_list_dict(node.metadata.get(MetadataField.ATTRIBUTES))
        preview = node.content[:preview_len]
        items.append(
            (
                node.id,
                sqlite_vec.serialize_float32(vector),
                node.title,
                node.node_type.value,
                source_file,
                preview,
            )
        )
        fts_rows.append((node.id, node.title, preview, node.node_type.value, source_file))
        for tag in node.metadata.get(MetadataField.TAGS, []):
            tag_rows.append((node.id, str(tag)))
        for token in attributes_to_list(attributes):
            attribute_rows.append((node.id, token))
        for key in attributes:
            attr_key_rows.append((node.id, key))

        if len(items) >= batch_size:
            indexed_count += flush()
    indexed_count += flush()
    conn.commit()

    if figure_nodes:
        logger.info(
            "Embedded %d of %d node(s) with figures; %d node(s) were indexed from text alone",
            figure_embedded,
            figure_nodes,
            figure_nodes - figure_embedded,
        )
    if degraded:
        logger.warning(
            "%d node(s) with figures were indexed without their image: %s",
            len(degraded),
            "; ".join(degraded[:LOG_SAMPLE_LIMIT]),
        )
    logger.info("Finished indexing %d nodes in the vector store at %s", indexed_count, db_file)
    return indexed_count
