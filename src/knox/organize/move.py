"""File operations for ``knox organize``.

Three responsibilities:

- :func:`flatten_sources` moves every regular file under ``sources_dir`` to
  the top level so the new tree can be built from a clean slate.
- :func:`materialize_tree` creates the directories implied by a
  :class:`CategoryTree`.
- :func:`move_files` performs the actual file moves and returns the
  ``old → new`` mapping used by the link-updater.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from knox.logging import get_logger
from knox.organize.categorize import CategoryNode, CategoryTree

logger = get_logger(__name__)


@dataclass(frozen=True)
class FileMove:
    """One planned or executed file move.

    ``old`` is a path *relative* to ``sources_dir`` and ``new`` is a path
    also relative to ``sources_dir``. Storing relatives keeps the mapping
    portable across machines and makes the JSON-safe report deterministic.
    """

    old: str
    new: str


def flatten_sources(sources_dir: Path) -> list[FileMove]:
    """Move every nested file to the top level of *sources_dir*.

    Name collisions are resolved by suffixing ``-2``, ``-3`` ... with the
    original extension preserved. Returns the list of moves performed
    (with relative paths) so the caller can keep an audit trail.
    """
    sources_dir = sources_dir.resolve()
    if not sources_dir.is_dir():
        return []

    moves: list[FileMove] = []
    for path in sorted(sources_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.parent == sources_dir:
            continue
        rel = path.relative_to(sources_dir)
        target = _unique_target(sources_dir, path.name)
        target_rel = target.relative_to(sources_dir)
        shutil.move(str(path), str(target))
        moves.append(FileMove(old=str(rel).replace("\\", "/"), new=str(target_rel).replace("\\", "/")))
        logger.debug("Flattened %s -> %s", rel, target_rel)
    return moves


def _unique_target(sources_dir: Path, name: str) -> Path:
    """Return a non-colliding target path under *sources_dir*."""
    candidate = sources_dir / name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for idx in range(2, 10_000):
        candidate = sources_dir / f"{stem}-{idx}{suffix}"
        if not candidate.exists():
            return candidate
    msg = f"Could not find a unique name for {name} under {sources_dir}"
    raise FileExistsError(msg)


def materialize_tree(
    sources_dir: Path,
    tree: CategoryTree,
) -> None:
    """Create directories for the proposed tree.

    Empty subcategories are skipped (no point creating ``category/x/`` with
    no files inside). Unfiled files stay at the top level and need no
    directory.
    """
    sources_dir = sources_dir.resolve()
    for cat in tree.categories:
        _materialize_node(sources_dir, sources_dir / cat.name, cat)


def _materialize_node(root: Path, current: Path, node: CategoryNode) -> None:
    has_content = bool(node.files) or any(c.files or c.subcategories for c in node.subcategories)
    if not has_content:
        return
    current.mkdir(parents=False, exist_ok=True)
    for child in node.subcategories:
        _materialize_node(root, current / child.name, child)


def move_files(
    sources_dir: Path,
    tree: CategoryTree,
) -> list[FileMove]:
    """Move each file from the top level of *sources_dir* to its new path.

    Files in :attr:`CategoryTree.unfiled` stay at the top level. The list
    of :class:`FileMove` records every successful move with relative paths.
    """
    sources_dir = sources_dir.resolve()
    moves: list[FileMove] = []
    for cat in tree.categories:
        moves.extend(_move_node(sources_dir, sources_dir / cat.name, cat, prefix=""))
    return moves


def _move_node(
    sources_dir: Path,
    target_dir: Path,
    node: CategoryNode,
    *,
    prefix: str,
) -> list[FileMove]:
    moves: list[FileMove] = []
    for filename in node.files:
        src = sources_dir / filename
        if not src.is_file():
            logger.warning("File listed in tree but missing on disk: %s", filename)
            continue
        new_rel = f"{prefix}{node.name}/{filename}" if prefix else f"{node.name}/{filename}"
        new_abs = sources_dir / new_rel
        if new_abs.exists() and new_abs.resolve() != src.resolve():
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            for idx in range(2, 10_000):
                candidate_name = f"{stem}-{idx}{suffix}"
                candidate_rel = f"{prefix}{node.name}/{candidate_name}" if prefix else f"{node.name}/{candidate_name}"
                candidate_abs = sources_dir / candidate_rel
                if not candidate_abs.exists():
                    new_rel = candidate_rel
                    new_abs = candidate_abs
                    break
        new_abs.parent.mkdir(parents=True, exist_ok=True)
        if new_abs.resolve() != src.resolve():
            shutil.move(str(src), str(new_abs))
        moves.append(FileMove(old=filename, new=new_rel))
    for child in node.subcategories:
        child_prefix = f"{prefix}{node.name}/" if prefix else f"{node.name}/"
        moves.extend(_move_node(sources_dir, target_dir / child.name, child, prefix=child_prefix))
    return moves
