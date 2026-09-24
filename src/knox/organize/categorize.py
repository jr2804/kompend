"""LLM-driven hierarchical categorisation of source files.

The module is responsible for the *brain* of the organize workflow:

- :func:`load_all_extractions` reads every cached ``Extraction`` object
  from the ``extract`` cache layer in the single store and pairs it with the
  on-disk source file.
- :func:`propose_categories` asks the LLM for a JSON category tree.
- :func:`validate_tree` enforces the user-visible constraints (max depth,
  minimum siblings, single-file collapse).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from knox.agents.categorize import CategorizeAgent
from knox.agents.deps import CategorizeDeps
from knox.compilation._two_phase_core import Extraction
from knox.core.cache import CacheStore
from knox.core.config import OrganizeConfig
from knox.core.paths import VaultPaths, is_protected_source
from knox.core.text.slug import slugify
from knox.logging import get_logger
from knox.types import JSONDict

logger = get_logger(__name__)


@dataclass
class CategoryNode:
    """A single node in the proposed category tree.

    Files appear in :attr:`files`. Children are stored in :attr:`subcategories`
    in declaration order. The :meth:`all_files` helper recursively collects
    every filename under the node, which is the input for collapsing and
    for the move plan.
    """

    name: str
    files: list[str] = field(default_factory=list)
    subcategories: list[CategoryNode] = field(default_factory=list)

    def all_files(self) -> list[str]:
        out: list[str] = list(self.files)
        for child in self.subcategories:
            out.extend(child.all_files())
        return out


@dataclass
class CategoryTree:
    """Root container returned by the LLM and consumed by the engine."""

    categories: list[CategoryNode] = field(default_factory=list)
    unfiled: list[str] = field(default_factory=list)

    def all_files(self) -> list[str]:
        out: list[str] = list(self.unfiled)
        for cat in self.categories:
            out.extend(cat.all_files())
        return out

    def remap_files(self, mapping: dict[str, str]) -> None:
        """Rewrite every file path in-place using ``mapping`` (old -> new).

        Called after ``flatten_sources`` so the tree's file paths reflect
        the post-flatten top-level locations.
        """
        self.unfiled = [mapping.get(f, f) for f in self.unfiled]
        for cat in self.categories:
            _remap_node(cat, mapping)


def _remap_node(node: CategoryNode, mapping: dict[str, str]) -> None:
    """Recursively rewrite file paths in ``node`` using ``mapping``."""
    node.files = [mapping.get(f, f) for f in node.files]
    for child in node.subcategories:
        _remap_node(child, mapping)


def load_all_extractions(
    base_dir: Path,
    sources_dir: Path,
) -> dict[str, Extraction]:
    """Read every cached extraction and match it to a file in ``sources_dir``.

    Extractions are read from the ``extract`` cache layer of the single
    ``.knox/knox.db`` store. Only entries whose ``source`` matches a file
    currently inside ``sources_dir`` are returned — stale extractions for
    moved or removed files are silently dropped.
    """
    paths = VaultPaths(base_dir)
    values = CacheStore(paths.cache_dir, base_dir).load_layer("extract")

    sources_root = sources_dir.resolve()
    # Index files currently on disk by basename so we survive moves (e.g.
    # after `knox organize` relocates a file into a subdirectory): the cache
    # stores the path the parser saw, which may be stale.
    files_by_name: dict[str, Path] = {}
    for path in sources_root.rglob("*"):
        if path.is_file() and _is_within(path, sources_root):
            if is_protected_source(path, sources_root):
                continue
            files_by_name.setdefault(path.name, path)

    known: dict[str, Extraction] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        cached = cast("JSONDict", value)
        source = cached.get("source", "")
        if not source:
            continue
        ext = Extraction.from_dict(cached)
        if not ext.summary:
            continue
        candidate = _resolve_cached_source(str(source), sources_root, files_by_name)
        if candidate is None:
            continue
        if is_protected_source(candidate, sources_root):
            continue
        rel = candidate.relative_to(sources_root).as_posix()
        known[rel] = ext
    if known:
        logger.info("Loaded %d extractions with summaries for sources in %s", len(known), sources_dir)
    elif files_by_name:
        logger.warning(
            "No cached extractions for %d source(s) in %s — knox organize would categorise from nothing; run knox ingest/compile first",
            len(files_by_name),
            sources_dir,
        )
    return known


def _resolve_cached_source(source: str, sources_root: Path, files_by_name: dict[str, Path]) -> Path | None:
    """Resolve a cached ``source`` path to a file currently under ``sources_root``.

    Tries the stored path verbatim first (handles relative and absolute).
    Falls back to a basename lookup so extractions survive file moves (e.g.
    after `knox organize` puts a file in a subdirectory). Returns ``None``
    when no matching file exists on disk.
    """
    candidate = Path(source)
    if not candidate.is_absolute():
        candidate = sources_root / candidate
    try:
        candidate = candidate.resolve()
    except OSError:
        candidate = Path(source)
    if candidate.is_file() and _is_within(candidate, sources_root):
        return candidate
    # Stale path (file moved) — try by basename.
    return files_by_name.get(Path(source).name)


def _is_within(subject: Path, root_dir: Path) -> bool:
    """Return True if *subject* is the same as or nested inside *root_dir*."""
    try:
        subject.relative_to(root_dir)
    except ValueError:
        return False
    return True


def propose_categories(
    extractions: dict[str, Extraction],
    config: OrganizeConfig,
    categorize_agent: CategorizeAgent,
) -> CategoryTree:
    """Ask the LLM to categorise every file into a hierarchical tree.

    The prompt comes from ``organize/categorize_prompt.md.j2`` and the
    system prompt from ``organize/categorize_system.md.j2``.
    """
    if not extractions:
        return CategoryTree()

    files_json = _build_files_payload(extractions)

    deps = CategorizeDeps(
        files_json=files_json,
        max_depth=config.max_depth,
        min_subcategories=config.min_subcategories,
        min_files_per_category=config.min_files_per_category,
        timeout=categorize_agent.timeout,
    )
    try:
        result = categorize_agent.run(deps)
    except Exception:
        logger.warning("LLM categorization failed; falling back to flat layout")
        return CategoryTree(categories=[], unfiled=sorted(extractions))

    return _tree_from_json(result.model_dump(), sorted(extractions))


def _build_files_payload(extractions: dict[str, Extraction]) -> str:
    """Serialise extractions to a compact JSON array for the prompt.

    The output is a single-line JSON list so the LLM can echo it back
    unchanged when it wants to reference specific files.
    """
    payload: list[dict[str, object]] = []
    for source in sorted(extractions):
        ext = extractions[source]
        payload.append(
            {
                "filename": source,
                "summary": ext.summary,
                "tags": list(ext.tags),  # nocheck: categorize output, not wiki metadata
            }
        )
    return json.dumps(payload, ensure_ascii=False)


def _tree_from_json(payload: dict[str, Any], known_files: list[str]) -> CategoryTree:
    """Parse and normalise the JSON returned by the LLM.

    Unknown files (the LLM invented a filename) are dropped; any file the
    LLM omitted ends up in :attr:`CategoryTree.unfiled` so nothing is lost.
    """
    if not isinstance(payload, dict):
        return CategoryTree(unfiled=known_files)
    raw_cats = payload.get("categories", [])
    raw_unfiled = payload.get("unfiled", [])
    if not isinstance(raw_cats, list):
        raw_cats = []
    if not isinstance(raw_unfiled, list):
        raw_unfiled = []

    assigned: set[str] = set()
    categories: list[CategoryNode] = []
    for entry in raw_cats:
        if not isinstance(entry, dict):
            continue
        node = _node_from_json(entry, assigned, known_files)
        if node is not None:
            categories.append(node)

    unfiled: list[str] = []
    for f in raw_unfiled:
        if isinstance(f, str) and f in known_files and f not in assigned:
            unfiled.append(f)
            assigned.add(f)
    for f in known_files:
        if f not in assigned:
            unfiled.append(f)
            assigned.add(f)

    return CategoryTree(categories=categories, unfiled=sorted(unfiled))


def _node_from_json(entry: dict[str, Any], assigned: set[str], known: list[str]) -> CategoryNode | None:
    name_raw = entry.get("name", "")
    if not isinstance(name_raw, str) or not name_raw.strip():
        return None
    name = slugify(name_raw, default="misc")
    files_raw = entry.get("files", [])
    if not isinstance(files_raw, list):
        files_raw = []
    files: list[str] = []
    for f in files_raw:
        if isinstance(f, str) and f in known and f not in assigned:
            files.append(f)
            assigned.add(f)
    subs_raw = entry.get("subcategories", [])
    if not isinstance(subs_raw, list):
        subs_raw = []
    subs: list[CategoryNode] = []
    for s in subs_raw:
        if isinstance(s, dict):
            sub = _node_from_json(s, assigned, known)
            if sub is not None:
                subs.append(sub)
    if not files and not subs:
        return None
    return CategoryNode(name=name, files=files, subcategories=subs)


def validate_tree(
    tree: CategoryTree,
    config: OrganizeConfig,
) -> CategoryTree:
    """Apply depth, sibling-count and single-file constraints.

    The transform is **non-destructive** for files: collapsing a single-file
    subcategory moves the file up to the parent rather than deleting it.
    """
    total = len(tree.all_files())
    # Clamp the per-category minimum to the corpus size so a value larger than
    # the document count degrades gracefully (near-flat layout) instead of
    # dissolving every category into unfiled.
    effective_min = min(config.min_files_per_category, total) if total else 1
    validated = CategoryTree(unfiled=list(tree.unfiled))
    for cat in tree.categories:
        pruned = _prune(cat, config, depth=1, effective_min=effective_min)
        # Top-level leaf with too few files → unfiled (no parent to promote to).
        if not pruned.subcategories and len(pruned.files) < effective_min:
            validated.unfiled.extend(pruned.all_files())
        else:
            validated.categories.append(pruned)
    return validated


def _prune(
    node: CategoryNode,
    config: OrganizeConfig,
    *,
    depth: int,
    effective_min: int,
) -> CategoryNode:
    name = slugify(node.name, default="misc")
    kept_subs: list[CategoryNode] = []
    collapsed_files: list[str] = []
    for child in node.subcategories:
        pruned = _prune(child, config, depth=depth + 1, effective_min=effective_min)
        # Under-populated leaf subcategory → dissolve, promote files to parent.
        if not pruned.subcategories and len(pruned.files) < effective_min:
            collapsed_files.extend(pruned.all_files())
        else:
            kept_subs.append(pruned)

    if depth >= config.max_depth:
        for child in kept_subs:
            collapsed_files.extend(child.all_files())
        kept_subs = []
    elif config.collapse_single_file and len(kept_subs) == 1 and not node.files:
        only = kept_subs[0]
        if not only.subcategories and len(only.files) <= 1:
            collapsed_files.extend(only.files)
            kept_subs = []

    # A non-leaf category must have at least min_subcategories children.
    # Categories with fewer are dissolved and their files promoted to the parent —
    # unless the remaining subcategories are already valid (have enough files),
    # in which case the parent is kept.
    if 0 < len(kept_subs) < config.min_subcategories and all(len(child.files) < effective_min for child in kept_subs):
        promote_files: list[str] = []
        for child in kept_subs:
            promote_files.extend(child.all_files())
        collapsed_files.extend(promote_files)
        kept_subs = []

    return CategoryNode(
        name=name,
        files=list(node.files) + collapsed_files,
        subcategories=kept_subs,
    )
