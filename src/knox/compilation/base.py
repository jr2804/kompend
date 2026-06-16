"""Abstract base for KNOX compilers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

from knox.core.models import CompiledWiki, ResolvedChunk, WikiNode

logger = logging.getLogger(__name__)


class BaseCompiler(ABC):
    """Compiler interface for KNOX Tier 4."""

    @abstractmethod
    def compile(self, chunks: list[ResolvedChunk]) -> CompiledWiki:
        """Compile resolved chunks into a knowledge graph wiki."""
        ...

    @staticmethod
    def disambiguate_duplicate_titles(
        nodes: list[WikiNode],
        chunks: list[ResolvedChunk],
    ) -> None:
        """Rename nodes that share a title by appending source context.

        Modifies ``nodes`` in-place. Nodes with unique titles are left
        unchanged. Duplicate titles get a parenthesised suffix derived
        from the node's source file (e.g. ``"References (paper-a)"``).
        When no source file is available, a numeric counter is used.
        """
        chunk_sources: dict[str, str] = {}
        for rc in chunks:
            if rc.chunk.source_file and rc.chunk.id:
                chunk_sources[rc.chunk.id] = Path(rc.chunk.source_file).stem

        title_groups: dict[str, list[int]] = defaultdict(list)
        for idx, node in enumerate(nodes):
            title_groups[node.title.lower().strip()].append(idx)

        for indices in title_groups.values():
            if len(indices) <= 1:
                continue
            used_suffixes: set[str] = set()
            for node_idx in indices:
                node = nodes[node_idx]
                source_stems = sorted({
                    chunk_sources[cid]
                    for cid in (node.source_chunks or [])
                    if cid in chunk_sources
                })
                suffix = source_stems[0] if source_stems else str(node_idx + 1)
                base = suffix
                counter = 2
                while suffix.lower() in used_suffixes:
                    suffix = f"{base} {counter}"
                    counter += 1
                used_suffixes.add(suffix.lower())
                node.title = f"{node.title} ({suffix})"
                logger.debug("Disambiguated node %s title -> %s", node.id, node.title)
