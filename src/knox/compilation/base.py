"""Abstract base for KNOX compilers."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

from knox.core.models import CompiledWiki, ResolvedChunk, WikiNode

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")


class BaseCompiler(ABC):
    """Compiler interface for KNOX Tier 4."""

    @abstractmethod
    def compile(self, chunks: list[ResolvedChunk]) -> CompiledWiki:
        """Compile resolved chunks into a knowledge graph wiki."""
        ...

    @staticmethod
    def effective_content_length(text: str) -> int:
        """Return character count of body text, excluding headings and whitespace."""
        body_lines = [line for line in (text or "").splitlines() if line.strip() and not _HEADING_RE.match(line)]
        return len("\n".join(body_lines).strip())

    @staticmethod
    def filter_empty_groups(
        groups: dict[str, list[ResolvedChunk]],
        *,
        min_content_length: int,
    ) -> dict[str, list[ResolvedChunk]]:
        """Remove groups whose merged content falls below *min_content_length*.

        Content is measured after stripping markdown headings and
        whitespace. Groups with title-only or heading-only chunks are
        removed. The ``_root`` group is always retained.
        """
        if min_content_length <= 0:
            return groups
        kept: dict[str, list[ResolvedChunk]] = {}
        for heading, group_chunks in groups.items():
            if heading == "_root":
                kept[heading] = group_chunks
                continue
            merged = "\n\n".join(rc.chunk.text for rc in group_chunks if rc.chunk.text)
            body_len = BaseCompiler.effective_content_length(merged)
            if body_len >= min_content_length:
                kept[heading] = group_chunks
            else:
                logger.debug("Filtered empty group '%s' (%d chars body)", heading, body_len)
        return kept

    @staticmethod
    def filter_empty_nodes(
        nodes: list[WikiNode],
        *,
        min_content_length: int,
    ) -> list[WikiNode]:
        """Remove nodes whose content falls below *min_content_length*.

        Content is measured after stripping markdown headings and
        whitespace. Overview nodes (``node_type == 'overview'``) are
        always retained.
        """
        if min_content_length <= 0:
            return nodes
        kept: list[WikiNode] = []
        for node in nodes:
            if node.node_type == "overview":
                kept.append(node)
                continue
            body_len = BaseCompiler.effective_content_length(node.content)
            if body_len >= min_content_length:
                kept.append(node)
            else:
                logger.debug(
                    "Filtered empty node %s '%s' (%d chars body)",
                    node.id,
                    node.title,
                    body_len,
                )
        return kept

    @staticmethod
    def deduplicate_group_text(group_chunks: list[ResolvedChunk]) -> str:
        """Merge chunks for one heading group, removing repeated paragraphs.

        The chunker prefixes each text chunk with ``[heading]`` and headings
        may be repeated across overlapping chunks. This method strips the
        prefix, tracks already-seen paragraphs (case/whitespace-normalised),
        and returns the deduplicated body text.
        """
        seen: set[str] = set()
        out_paragraphs: list[str] = []
        for rc in group_chunks:
            text = (rc.chunk.text or "").strip()
            if not text:
                continue
            # Strip leading ``[heading_context] `` prefix added by the chunker.
            if text.startswith("["):
                close = text.find("] ")
                if close != -1:
                    text = text[close + 2 :]
            for raw_para in text.split("\n\n"):
                para = raw_para.strip()
                if not para:
                    continue
                # Strip another bracket prefix if it appears inside a paragraph.
                if para.startswith("["):
                    close = para.find("] ")
                    if close != -1:
                        stripped = para[close + 2 :].strip()
                        if not stripped:
                            continue
                        para = stripped
                normalized = " ".join(para.split()).lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                out_paragraphs.append(para)
        return "\n\n".join(out_paragraphs)

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
                source_stems = sorted({chunk_sources[cid] for cid in (node.source_chunks or []) if cid in chunk_sources})
                suffix = source_stems[0] if source_stems else str(node_idx + 1)
                base = suffix
                counter = 2
                while suffix.lower() in used_suffixes:
                    suffix = f"{base} {counter}"
                    counter += 1
                used_suffixes.add(suffix.lower())
                node.title = f"{node.title} ({suffix})"
                logger.debug("Disambiguated node %s title -> %s", node.id, node.title)
