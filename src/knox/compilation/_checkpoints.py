"""Generation checkpoint persistence for two-phase compilation.

Extracted from :mod:`knox.compilation._two_phase_core` — per-batch
generation checkpoints allow resuming a partially completed generation
phase after a crash or interrupt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from knox.core.cache.store import _decode_json, _encode_json, make_cache_key
from knox.core.models import WikiNode
from knox.core.pipeline.config import PipelineConfig
from knox.logging import get_logger

logger = get_logger(__name__)


def _gen_checkpoint_dir(config: PipelineConfig) -> Path:
    """Directory for per-batch generation checkpoint files."""
    return config.paths.cache_dir / "gen_checkpoint"


def _gen_checkpoint_path(config: PipelineConfig, gen_deps: dict[str, str]) -> Path:
    """On-disk path for the generation checkpoint."""
    return _gen_checkpoint_dir(config) / f"{make_cache_key(gen_deps)}.json"


def _load_gen_checkpoint(
    config: PipelineConfig,
    gen_deps: dict[str, str],
) -> tuple[list[WikiNode], dict[str, int], set[int]] | None:
    """Load a generation checkpoint if it exists and deps match.

    Returns ``(nodes, slug_to_idx, completed_batches)`` or ``None``
    when the checkpoint is absent, stale, or malformed.
    """
    if getattr(config, "no_cache", False):
        return None
    path = _gen_checkpoint_path(config, gen_deps)
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
        nodes: list[WikiNode] = [_decode_json(n) for n in raw.get("nodes", [])]  # nocheck: checkpoint format, not wiki metadata
    except Exception:
        logger.debug("Generation checkpoint node deserialization error -- ignoring", exc_info=True)
        return None
    slug_to_idx: dict[str, int] = dict(raw.get("slug_to_idx", {}))
    completed: set[int] = set(raw.get("completed_batches", []))
    return nodes, slug_to_idx, completed


def _save_gen_checkpoint(
    config: PipelineConfig,
    gen_deps: dict[str, str],
    nodes: list[WikiNode],
    slug_to_idx: dict[str, int],
    completed_batches: set[int],
) -> None:
    """Write generation checkpoint to disk (atomic write-to-temp + rename)."""
    if getattr(config, "no_cache", False):
        return
    path = _gen_checkpoint_path(config, gen_deps)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "deps": gen_deps,
        "nodes": [_encode_json(n) for n in nodes],  # nocheck: checkpoint format, not wiki metadata
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


def _delete_gen_checkpoint(config: PipelineConfig, gen_deps: dict[str, str]) -> None:
    """Remove a generation checkpoint from disk."""
    if getattr(config, "no_cache", False):
        return
    path = _gen_checkpoint_path(config, gen_deps)
    try:
        path.unlink()
        logger.debug("Generation checkpoint deleted: %s", path)
    except OSError:
        logger.debug("Generation checkpoint delete failed -- ignoring", exc_info=True)
