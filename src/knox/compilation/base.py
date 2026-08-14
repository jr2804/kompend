"""Abstract base for KNOX compilers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from knox.compilation._two_phase_core import _merge_node
from knox.compilation.flows.assembly import cosine_similarity
from knox.core.models import ChunkTree, CompiledWiki, ResolvedChunk, WikiNode
from knox.logging import get_logger
from knox.rag.zvec_store import make_provider

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")

logger = get_logger(__name__)


class BaseCompiler(ABC):
    """Compiler interface for KNOX Tier 4."""

    # Optional progress hook: ``(current, total, description)``. The pipeline
    # sets this before ``compile`` so the compiler can report real, countable
    # progress (e.g. per-source extraction counts) to a determinate CLI bar.
    # ``None`` = no display attached (tests, library use).
    on_progress: Callable[[int, int, str], None] | None = None

    @abstractmethod
    def compile(
        self,
        chunks: list[ResolvedChunk],
        *,
        trees: list[ChunkTree] | None = None,
        pinned_facts: list[str] | None = None,
    ) -> CompiledWiki:
        """Compile resolved chunks into a knowledge graph wiki.

        Parameters
        ----------
        chunks:
            Resolved leaf chunks to compile.
        trees:
            Optional ChunkTree list for multi-resolution compilation.
            When provided, internal node summaries and structured data
            are used to build richer wiki pages.
        pinned_facts:
            Optional list of facts the compiler MUST preserve in the
            generated wiki. Used by WiCER to forward CEGAR-style
            preservation constraints discovered during the
            compile-evaluate-refine loop. Synthesis-based compilers
            (e.g. :class:`TwoPhaseCompiler`) inject these as explicit
            constraints in the generation prompt. Compilers that only
            rearrange raw chunk text (e.g. :class:`GraphCompiler`)
            ignore this parameter.
        """
        ...

    def merge_duplicate_nodes(
        self,
        nodes: list[WikiNode],
        embedding_model: str,
        *,
        threshold: float = 0.82,  # set from BudgetConfig.concept_merge_threshold at call site
    ) -> list[WikiNode]:
        """Merge nodes whose title embeddings exceed *threshold*.

        Post-generation fuzzy merge: clusters concept nodes by title
        embedding cosine similarity and merges each cluster via
        :func:`_merge_node` (section-aware content merge, provenance/
        related/source_chunks unioned). This catches near-duplicate
        concepts that survived the concept-stage merge under slightly
        different surface forms.

        When embeddings are unavailable (``embedding_model == 'hash'``
        or provider failure), nodes are returned unchanged —
        :meth:`disambiguate_duplicate_titles` handles exact-title
        collisions as a fallback.
        """
        if len(nodes) < 2 or not embedding_model or embedding_model == "hash":
            return nodes

        concept_nodes = [n for n in nodes if n.node_type == "concept"]
        if len(concept_nodes) < 2:
            return nodes

        try:
            provider = make_provider(embedding_model)
            titles = [n.title for n in concept_nodes]
            vecs = [provider.embed(t) for t in titles]
        except Exception as exc:
            logger.warning(
                "Node merge disabled (%s): %s — falling through to disambiguation",
                embedding_model,
                exc,
            )
            return nodes

        n = len(concept_nodes)
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x: int, y: int) -> None:
            rx, ry = _find(x), _find(y)
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry

        merged_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                if cosine_similarity(vecs[i], vecs[j]) >= threshold:
                    _union(i, j)
                    merged_pairs += 1

        if merged_pairs == 0:
            return nodes

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(_find(i), []).append(i)

        merged_concepts: list[WikiNode] = []
        for members in groups.values():
            if len(members) == 1:
                merged_concepts.append(concept_nodes[members[0]])
            else:
                survivor = concept_nodes[members[0]]
                for m in members[1:]:
                    survivor = _merge_node(survivor, concept_nodes[m])
                merged_concepts.append(survivor)

        non_concept = [n for n in nodes if n.node_type != "concept"]
        result = non_concept + merged_concepts

        logger.info(
            "Node merge: %d concept pairs merged (%d → %d concept nodes, threshold=%.2f)",
            merged_pairs,
            n,
            len(merged_concepts),
            threshold,
        )
        return result

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
    def dedup_content_duplicate_nodes(nodes: list[WikiNode]) -> list[WikiNode]:
        """Drop nodes whose normalized body duplicates an earlier node's.

        Catches the case where extraction surfaces one concept under
        singular/plural or reordered titles and the generator emits
        identical content for both (e.g. ``phoneme-posteriorgram`` /
        ``phoneme-posteriorgrams``). Exact-body match only — no fuzzy
        heuristic, so no false merges of genuinely distinct concepts.

        Before dropping a duplicate, its ``source_chunks``, provenance,
        and related lists are unioned into the survivor so no provenance
        is lost.
        """
        seen: dict[str, int] = {}  # body-key -> index in kept
        kept: list[WikiNode] = []
        for node in nodes:
            key = " ".join((node.content or "").split()).lower()
            if not key or key not in seen:
                if key:
                    seen[key] = len(kept)
                kept.append(node)
            else:
                survivor = kept[seen[key]]
                survivor.source_chunks = list(
                    dict.fromkeys(
                        (survivor.source_chunks or []) + (node.source_chunks or []),
                    ),
                )
                ex_prov = {str(x) for x in (survivor.metadata.get("provenance") or [])}
                inc_prov = {str(x) for x in (node.metadata.get("provenance") or [])}
                if ex_prov or inc_prov:
                    survivor.metadata["provenance"] = sorted(ex_prov | inc_prov)
                ex_rel = {str(x) for x in (survivor.metadata.get("related") or [])}
                inc_rel = {str(x) for x in (node.metadata.get("related") or [])}
                if ex_rel or inc_rel:
                    survivor.metadata["related"] = sorted(ex_rel | inc_rel)
                logger.debug(
                    "Deduped content-duplicate node '%s' (merged into '%s')",
                    node.title,
                    survivor.title,
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

    def _report(self, current: int, total: int, description: str) -> None:
        """Report compile progress to the display (no-op if no hook).

        *current*/*total* let the bar fill determinately; *description*
        updates the status line. Compilers MUST call this so the bar
        counts real work units rather than faking activity.
        """
        if self.on_progress is not None:
            self.on_progress(current, total, description)
