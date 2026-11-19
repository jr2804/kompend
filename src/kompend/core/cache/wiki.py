"""Wiki cache serialization.

Single source of truth for writing the ``last`` wiki snapshot — kept here so
that pure-core modules (e.g. ``kompend.lint.flow``) can persist wiki state
without depending on ``kompend.cli.workflow.post_compile``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from kompend.core.models import CompiledWiki, MetadataField, WikiField
from kompend.core.models.wiki import WIKI_SNAPSHOT_LAST, load_wiki_snapshot, write_wiki_snapshot_payload
from kompend.core.types import ReasonCode
from kompend.logging import get_logger

logger = get_logger(__name__)


def write_wiki_cache(wiki: CompiledWiki, base_dir: Path) -> None:
    """Serialize the compiled wiki to the ``last`` snapshot in ``kompend.db``.

    An empty wiki never overwrites a populated snapshot: a degraded compile
    (extraction failed on every batch) produces a graph with no pages, and
    writing it would erase the last good graph that lint and gap analysis
    read — while the pages themselves still sit in the output directory. The
    run gets a warning naming what was kept. A vault legitimately emptied of
    sources refreshes its cache on its first producing compile; until then
    the stale cached graph matches the stale pages on disk.
    """

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
    if not nodes_dicts:
        previous = load_wiki_snapshot(base_dir, WIKI_SNAPSHOT_LAST)
        if previous is not None and previous.nodes:
            logger.warning(
                "Compile produced no wiki pages — keeping the previous wiki cache (%d page(s)) instead of overwriting it with an empty graph",
                len(previous.nodes),
                extra={"reason_code": ReasonCode.EMPTY_WIKI},
            )
            return
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
