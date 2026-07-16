"""Top-level orchestration for ``knox organize``.

The single entry point :func:`organize_sources` ties together extraction
loading, LLM categorisation, tree validation, file moves and wiki link
updates.  Callers (CLI, tests) get back an :class:`OrganizeResult` that
contains the proposed tree, the list of moves, and the link-updater
counters so they can render a meaningful summary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knox.agents.categorize import CategorizeAgent
from knox.core.config import OrganizeConfig, resolve_config
from knox.core.config.keys import (
    ORGANIZE_COLLAPSE_SINGLE_FILE,
    ORGANIZE_MAX_DEPTH,
    ORGANIZE_MIN_FILES_PER_CATEGORY,
    ORGANIZE_MIN_SUBCATEGORIES,
)
from knox.core.paths import VaultPaths, is_protected_source
from knox.logging import get_logger
from knox.organize.categorize import (
    CategoryNode,
    CategoryTree,
    load_all_extractions,
    propose_categories,
    validate_tree,
)
from knox.organize.moves import FileMove, LinkRemap, build_link_remap, flatten_sources, materialize_tree, move_files, update_wiki_links

logger = get_logger(__name__)


@dataclass
class OrganizeResult:
    """Outcome of one ``organize`` run."""

    tree: CategoryTree
    moves: list[FileMove] = field(default_factory=list)
    link_updates: dict[str, int] = field(default_factory=dict)
    remap: LinkRemap | None = None
    dry_run: bool = False
    files_without_summary: list[str] = field(default_factory=list)

    @property
    def moved_count(self) -> int:
        return len(self.moves)


def resolve_organize_config(overrides: dict[str, object] | None = None) -> OrganizeConfig:
    """Build an :class:`OrganizeConfig` from the resolved knox config."""
    cfg = resolve_config(cli_overrides=overrides)
    min_files = int(cfg.get(ORGANIZE_MIN_FILES_PER_CATEGORY, 2))
    if min_files < 1:
        raise ValueError(f"organize.min_files_per_category must be >= 1, got {min_files}")
    return OrganizeConfig(
        max_depth=int(cfg.get(ORGANIZE_MAX_DEPTH, 3)),  # type: ignore[arg-type]
        min_subcategories=int(cfg.get(ORGANIZE_MIN_SUBCATEGORIES, 2)),  # type: ignore[arg-type]
        min_files_per_category=min_files,
        collapse_single_file=bool(cfg.get(ORGANIZE_COLLAPSE_SINGLE_FILE, True)),  # type: ignore[arg-type]
    )


def organize_sources(
    base_dir: Path,
    sources_dir: Path,
    output_dir: Path,
    categorize_agent: CategorizeAgent,
    config: OrganizeConfig | None = None,
    *,
    dry_run: bool = False,
) -> OrganizeResult:
    """Run the full organise workflow and return a structured report.

    Parameters
    ----------
    base_dir:
        Project root (where ``.knox/`` lives).
    sources_dir:
        The directory whose files will be re-categorised. Must already
        contain the files — we never auto-download.
    output_dir:
        Where the wiki output lives. Wiki links in this tree will be
        rewritten after the move.
    categorize_agent:
        The PydanticAI agent used for categorisation.
    config:
        Optional :class:`OrganizeConfig`. When ``None`` we resolve it
        from the project config.
    dry_run:
        When ``True`` the proposed tree and planned moves are returned
        without touching the filesystem or wiki links.
    """
    base_dir = base_dir.resolve()
    sources_dir = sources_dir.resolve()
    output_dir = output_dir.resolve()

    if config is None:
        config = resolve_organize_config()

    extractions = load_all_extractions(base_dir, sources_dir)
    if not extractions:
        logger.warning("No cached extractions found — run `knox compile` first")
        return OrganizeResult(
            tree=CategoryTree(),
            dry_run=dry_run,
            files_without_summary=[],
        )

    files_without_summary = _files_without_summary(sources_dir, extractions)

    proposed = propose_categories(extractions, config, categorize_agent)
    validated = validate_tree(proposed, config)

    if dry_run:
        logger.info("Dry run — skipping flatten / move / link update")
        return OrganizeResult(
            tree=validated,
            dry_run=True,
            files_without_summary=files_without_summary,
        )

    flat_moves = flatten_sources(sources_dir)
    flatten_remap = {m.old: m.new for m in flat_moves}
    if flatten_remap:
        validated.remap_files(flatten_remap)
    materialize_tree(sources_dir, validated)
    moves = move_files(sources_dir, validated)

    remap = build_link_remap([(m.old, m.new) for m in moves])
    counts = update_wiki_links(base_dir, output_dir, remap)

    _write_report(base_dir, validated, moves, counts, dry_run=False)

    return OrganizeResult(
        tree=validated,
        moves=moves,
        link_updates=counts,
        remap=remap,
        dry_run=False,
        files_without_summary=files_without_summary,
    )


def _files_without_summary(sources_dir: Path, extractions: dict[str, Any]) -> list[str]:
    """Files in ``sources_dir`` that lack an extraction summary."""
    known = set(extractions)
    missing: list[str] = []
    for path in sorted(sources_dir.rglob("*")):
        if not path.is_file():
            continue
        if is_protected_source(path, sources_dir):
            continue
        # Path must live under sources_dir; skip symlinked escapes otherwise.
        if not path.is_relative_to(sources_dir):
            continue
        rel = path.relative_to(sources_dir).as_posix()
        if rel not in known:
            missing.append(rel)
    return missing


def _write_report(
    base_dir: Path,
    tree: CategoryTree,
    moves: list[FileMove],
    link_updates: dict[str, int],
    *,
    dry_run: bool,
) -> None:
    """Persist a JSON report next to the wiki metadata."""
    report_file = VaultPaths(base_dir).organize_report_path
    payload = {
        "dry_run": dry_run,
        "categories": _tree_to_json(tree),
        "moves": [{"old": m.old, "new": m.new} for m in moves],
        "link_updates": link_updates,
    }
    report_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _tree_to_json(tree: CategoryTree) -> list[dict[str, object]]:
    return [_node_to_json(n) for n in tree.categories]


def _node_to_json(node: CategoryNode) -> dict[str, object]:
    return {
        "name": node.name,
        "files": list(node.files),
        "subcategories": [_node_to_json(c) for c in node.subcategories],
    }
