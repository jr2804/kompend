"""Generation checkpoint persistence for two-phase compilation.

Extracted from :mod:`knox.compilation._two_phase_core` — per-batch
generation checkpoints allow resuming a partially completed generation
phase after a crash or interrupt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from knox.core.cache.store import decode_json, encode_json, make_cache_key
from knox.core.models import MetadataField, WikiNode
from knox.core.paths import rel_to_vault
from knox.core.pipeline.config import PipelineConfig
from knox.logging import get_logger

logger = get_logger(__name__)


def load_gen_checkpoint(
    config: PipelineConfig,
    gen_deps: dict[str, str],
) -> tuple[list[WikiNode], dict[str, int], set[int]] | None:
    """Load a generation checkpoint if it exists and deps match.

    Returns ``(nodes, slug_to_idx, completed_batches)`` or ``None``
    when the checkpoint is absent, stale, or malformed.
    """
    if getattr(config, "no_cache", False):
        return None
    path = _gen_checkpoint_file(config, gen_deps)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Generation checkpoint read error -- ignoring", exc_info=True)
        return None
    stored_deps = raw.get("deps", {})
    if stored_deps != gen_deps:
        logger.debug("Generation checkpoint deps mismatch -- ignoring")
        return None
    try:
        nodes: list[WikiNode] = [
            n for n in (decode_json(n) for n in raw.get("nodes", [])) if isinstance(n, WikiNode)
        ]  # guard: checkpoint format is untrusted JSON
    except Exception:
        logger.debug("Generation checkpoint node deserialization error -- ignoring", exc_info=True)
        return None
    slug_to_idx: dict[str, int] = dict(raw.get("slug_to_idx", {}))
    completed: set[int] = set(raw.get("completed_batches", []))
    return nodes, slug_to_idx, completed


def save_gen_checkpoint(
    config: PipelineConfig,
    gen_deps: dict[str, str],
    nodes: list[WikiNode],
    slug_to_idx: dict[str, int],
    completed_batches: set[int],
) -> None:
    """Write generation checkpoint to disk (atomic write-to-temp + rename)."""
    if getattr(config, "no_cache", False):
        return
    path = _gen_checkpoint_file(config, gen_deps)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = config.paths.base_dir
    data = {
        "deps": gen_deps,
        "nodes": [encode_json(_relativize_node(n, base_dir)) for n in nodes],  # nocheck: checkpoint format, not wiki metadata
        "slug_to_idx": slug_to_idx,
        "completed_batches": list(completed_batches),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    try:
        os.replace(tmp, path)
        logger.debug("Generation checkpoint saved: %s", path)
    except OSError:
        logger.debug("Generation checkpoint write failed -- ignoring", exc_info=True)


def _relativize_node(node: WikiNode, base_dir: Path) -> WikiNode:
    """Return a copy of *node* with ``SOURCE_FILE`` metadata relativized to *base_dir*.

    Paths are absolute in memory but must be vault-relative on disk so the
    checkpoint (under ``.knox/``) stays relocatable — see Antipattern 6 in
    ``src/knox/AGENTS.md``.
    """
    meta = node.metadata or {}
    if MetadataField.SOURCE_FILE not in meta:
        return node
    rel = rel_to_vault(meta[MetadataField.SOURCE_FILE], base_dir)
    if rel == meta[MetadataField.SOURCE_FILE]:
        return node
    new_meta = {**meta, MetadataField.SOURCE_FILE: rel}
    return WikiNode(
        id=node.id,
        title=node.title,
        content=node.content,
        node_type=node.node_type,
        source_chunks=node.source_chunks,
        metadata=new_meta,
    )


def delete_gen_checkpoint(config: PipelineConfig, gen_deps: dict[str, str]) -> None:
    """Remove a generation checkpoint from disk."""
    if getattr(config, "no_cache", False):
        return
    path = _gen_checkpoint_file(config, gen_deps)
    try:
        path.unlink()
        logger.debug("Generation checkpoint deleted: %s", path)
    except OSError:
        logger.debug("Generation checkpoint delete failed -- ignoring", exc_info=True)


def _gen_checkpoint_file(config: PipelineConfig, gen_deps: dict[str, str]) -> Path:
    """On-disk path for the generation checkpoint."""
    return _gen_checkpoint_dir(config) / f"{make_cache_key(gen_deps)}.json"


def _gen_checkpoint_dir(config: PipelineConfig) -> Path:
    """Directory for per-batch generation checkpoint files."""
    return config.paths.cache_dir / "gen_checkpoint"
