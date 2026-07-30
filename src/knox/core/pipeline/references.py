"""Persistent storage for extracted bibliographic references.

Follows the :class:`TierStateStore` pattern: a simple JSON file
(``references.json``) keyed by source file path, where each entry is a
list of CSL JSON-aligned :class:`~knox.core.models.references.Reference`
objects.

The store is loaded once at pipeline start and saved after all sources
are processed. A re-run merges new references with existing ones so the
store is cumulative across compilations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knox.core.models.references import Reference
from knox.logging import get_logger
from knox.rag.zvec_store import make_provider

logger = get_logger(__name__)

__all__ = ["ReferenceStore", "consolidate_references", "merge_similar_references"]

#: Cosine similarity threshold for fuzzy reference dedup.
_REF_MERGE_THRESHOLD = 0.85


class ReferenceStore:
    """JSON-backed store of bibliographic references per source.

    File shape::

        {
          "sources/example_paper.pdf": [
            {"id": "smith2020", "type": "article-journal", ...},
            ...
          ],
          "sources/report.md": [ ... ]
        }

    Parameters
    ----------
    references_file:
        File to read/write. Created lazily on first save.
    """

    def __init__(self, references_file: Path) -> None:
        self._file = references_file
        self._data: dict[str, list[dict[str, Any]]] = {}
        self.load()

    def load(self) -> None:
        """Read the references file into memory.

        A missing file means a fresh start (empty store). A corrupt file
        is logged as a warning and treated as empty rather than crashing
        the pipeline.
        """
        if not self._file.exists():
            self._data = {}
            return
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("ReferenceStore: corrupt file %s — starting fresh: %s", self._file, exc)
            self._data = {}
            return
        if isinstance(raw, dict):
            self._data = {k: v for k, v in raw.items() if isinstance(v, list)}
        else:
            self._data = {}

    def save(self) -> None:
        """Persist the in-memory store to disk."""
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(
            json.dumps(self._data, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def add_source(self, source: str, refs: list[Reference]) -> int:
        """Replace the reference list for *source* with *refs*.

        Returns the number of references stored.
        """
        self._data[source] = [r.model_dump() for r in refs]
        return len(self._data[source])

    def get_source(self, source: str) -> list[Reference]:
        """Return all references for a given source."""
        raw_list = self._data.get(source, [])
        out: list[Reference] = []
        for raw in raw_list:
            try:
                out.append(Reference.model_validate(raw))
            except Exception as exc:  # noqa: BLE001 — skip malformed entries
                logger.debug("ReferenceStore: skipping malformed entry in %s: %s", source, exc)
        return out

    def unresolved_indices(self, source: str | None = None) -> list[tuple[str, int]]:
        """Return ``(source, index)`` pairs for stored refs lacking DOI and URL.

        Parameters
        ----------
        source:
            When given, only that source is scanned. Otherwise every source
            in the store is scanned.
        """
        sources = [source] if source else list(self._data)
        pairs: list[tuple[str, int]] = []
        for src in sources:
            for i, raw in enumerate(self._data.get(src, [])):
                if not raw.get("doi") and not raw.get("url"):
                    pairs.append((src, i))
        return pairs

    def update_reference(
        self,
        source: str,
        index: int,
        *,
        doi: str = "",
        url: str = "",
    ) -> bool:
        """Set the DOI/URL of a stored reference.

        Returns ``True`` if a non-empty value was written to a previously
        empty field. Never overwrites an existing identifier.
        """
        ref_list = self._data.get(source)
        if not ref_list or index < 0 or index >= len(ref_list):
            return False
        raw = ref_list[index]
        changed = False
        doi_norm = doi.strip()
        if doi_norm and not raw.get("doi"):
            raw["doi"] = doi_norm
            changed = True
        url_norm = url.strip()
        if url_norm and not raw.get("url"):
            raw["url"] = url_norm
            changed = True
        return changed

    def all_references(self) -> list[Reference]:
        """Return all references across all sources (deduped by label).

        When the same reference appears in multiple sources, the first
        occurrence is kept and its ``citing_source`` is updated to include
        all sources that cite it.
        """
        seen: dict[str, Reference] = {}
        for source in sorted(self._data):
            for ref in self.get_source(source):
                label = ref.short_label()
                if label:
                    if label in seen:
                        # Merge citing sources from duplicate
                        existing = seen[label]
                        existing_sources = [s.strip() for s in existing.citing_source.split(";") if s.strip()]
                        new_sources = [s.strip() for s in ref.citing_source.split(";") if s.strip()]
                        merged = list(dict.fromkeys(existing_sources + new_sources))
                        existing.citing_source = "; ".join(merged)
                    else:
                        seen[label] = ref
        return list(seen.values())

    def total_count(self) -> int:
        """Total number of references across all sources."""
        return sum(len(v) for v in self._data.values())

    def source_count(self) -> int:
        """Number of sources with at least one reference."""
        return len([k for k, v in self._data.items() if v])

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Return a plain-dict copy of the store (for serialization/tests)."""
        return self._data.copy()

    def deduplicate(self, embedding_model: str = "", *, threshold: float = _REF_MERGE_THRESHOLD) -> int:
        """In-place fuzzy dedup of references across all sources.

        Uses embedding cosine similarity (same pattern as
        :func:`knox.compilation.flows.assembly.merge_similar_concepts`)
        to find near-duplicate references whose titles are slightly
        different (e.g. capitalisation, abbreviation, minor typos).

        When *embedding_model* is empty or ``"hash"``, no fuzzy merge
        occurs — only exact-key dedup (DOI, label) runs.

        Returns the number of references removed.
        """
        all_refs = self.all_references()
        before = len(all_refs)
        if before <= 1:
            return 0

        merged = merge_similar_references(
            all_refs,
            embedding_model,
            threshold=threshold,
        )
        removed = before - len(merged)
        if removed == 0:
            return 0

        # Rewrite store: group merged references by their citing_source.
        # A reference cited by multiple sources keeps all source keys.
        self._data = {}
        for ref in merged:
            source = ref.citing_source or "unknown"
            self._data.setdefault(source, []).append(ref.model_dump())

        logger.info(
            "ReferenceStore: fuzzy dedup removed %d near-duplicate(s) (%d → %d, threshold=%.2f)",
            removed,
            before,
            len(merged),
            threshold,
        )
        return removed


def consolidate_references(
    references: list[Reference],
    embedding_model: str = "",
    *,
    threshold: float = _REF_MERGE_THRESHOLD,
) -> list[Reference]:
    """Deduplicate and consolidate references into a flat list.

    This is the public entry point for producing a global OKF
    ``references.json`` from per-source reference stores. It runs
    exact-key dedup first, then optional fuzzy merge.
    """
    return merge_similar_references(references, embedding_model, threshold=threshold)


def merge_similar_references(
    references: list[Reference],
    embedding_model: str,
    *,
    threshold: float = _REF_MERGE_THRESHOLD,
) -> list[Reference]:
    """Merge references whose embeddings are above *threshold*.

    Uses the configured embedding provider (same pattern as
    :func:`knox.compilation.flows.assembly.merge_similar_concepts`).
    Groups references greedily via Union-Find: for every pair with
    cosine similarity >= threshold, metadata is merged (sources unioned,
    in-text keys unioned, longest title kept as canonical).

    Exact-key dedup (same DOI or same short_label) always runs first,
    even when embeddings are unavailable.

    When the embedding provider fails (e.g. model not found), references
    are returned with only exact-key dedup applied.
    """
    if not references:
        return []

    deduped = _exact_dedup_references(references)
    return _fuzzy_dedup_references(deduped, embedding_model, threshold)


def _exact_dedup_references(references: list[Reference]) -> list[Reference]:
    """Deduplicate references by DOI or short-label match.

    Returns merged references where duplicates are combined via
    :func:`_merge_ref_group`.
    """
    exact_groups: list[list[Reference]] = []
    seen_doi: dict[str, int] = {}
    seen_label: dict[str, int] = {}

    for ref in references:
        matched_idx: int | None = None

        doi = ref.doi.strip().lower()
        if doi and doi in seen_doi:
            matched_idx = seen_doi[doi]

        if matched_idx is None:
            label = ref.short_label()
            if label and label in seen_label:
                matched_idx = seen_label[label]

        if matched_idx is not None:
            exact_groups[matched_idx].append(ref)
        else:
            exact_groups.append([ref])
            idx = len(exact_groups) - 1
            if doi:
                seen_doi[doi] = idx
            label = ref.short_label()
            if label:
                seen_label[label] = idx

    return [_merge_ref_group(g) for g in exact_groups]


def _fuzzy_dedup_references(
    references: list[Reference],
    embedding_model: str,
    threshold: float,
) -> list[Reference]:
    """Embedding-based fuzzy dedup using cosine similarity.

    Uses Union-Find to group references whose embeddings are above
    *threshold*, then merges each group via :func:`_merge_ref_group`.

    When the embedding provider or embedding call fails, references are
    returned unchanged (exact dedup only).
    """
    if len(references) <= 1:
        return references
    if not embedding_model or embedding_model == "hash":
        return references

    try:
        provider = make_provider(embedding_model, dim=None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Reference fuzzy merge disabled: provider creation failed: %s", exc)
        return references

    texts = [_ref_embedding_text(r) for r in references]
    try:
        vecs = [provider.embed(t) for t in texts]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Reference fuzzy merge disabled: embedding call failed: %s", exc)
        return references

    n = len(references)
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

    for i in range(n):
        for j in range(i + 1, n):
            if _cosine_similarity(vecs[i], vecs[j]) >= threshold:
                _union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(_find(i), []).append(i)

    result = [_merge_ref_group([references[m] for m in members]) for members in groups.values()]

    merged_count = n - len(result)
    if merged_count:
        logger.info(
            "Reference fuzzy merge: %d near-duplicate group(s) combined (%d → %d, threshold=%.2f)",
            merged_count,
            n,
            len(result),
            threshold,
        )
    return result


# ── Standalone functions ────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (no numpy needed)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _ref_embedding_text(ref: Reference) -> str:
    """Build the text used to embed a reference for fuzzy matching."""
    parts: list[str] = []
    if ref.title:
        parts.append(ref.title)
    first_author = ""
    if ref.author:
        a0 = ref.author[0]
        first_author = a0.literal or a0.family or ""
    if first_author:
        parts.append(first_author)
    if ref.container_title:
        parts.append(ref.container_title)
    if ref.issued:
        parts.append(ref.issued)
    return " ".join(parts).strip().lower()


def _merge_ref_group(group: list[Reference]) -> Reference:
    """Merge a list of references into a single canonical reference.

    Keeps the reference with the most populated fields as the base,
    then unions ``citing_source`` and ``in_text_keys`` from all members.
    """
    if len(group) == 1:
        return group[0]

    # Pick the reference with the most non-empty CSL fields as base.
    def _richness(r: Reference) -> int:
        return sum(1 for v in (r.title, r.doi, r.author, r.container_title, r.volume, r.pages, r.publisher, r.url) if v)

    base = max(group, key=_richness)

    all_sources: list[str] = []
    all_keys: list[str] = []
    seen_sources: set[str] = set()
    seen_keys: set[str] = set()

    for ref in group:
        src = ref.citing_source
        if src and src not in seen_sources:
            all_sources.append(src)
            seen_sources.add(src)
        for k in ref.in_text_keys:
            if k not in seen_keys:
                all_keys.append(k)
                seen_keys.add(k)

    return base.model_copy(
        update={
            "citing_source": "; ".join(all_sources) if len(all_sources) > 1 else (all_sources[0] if all_sources else base.citing_source),
            "in_text_keys": all_keys,
        }
    )
