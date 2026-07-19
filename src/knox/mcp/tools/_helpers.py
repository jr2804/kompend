"""Shared helpers for MCP tools."""

from __future__ import annotations

from pathlib import Path

from knox.core.config import resolve_config
from knox.core.config.keys import COMPILE_EMBEDDING_MODEL, PATHS_WIKI_DIR
from knox.core.trust.actor import resolve_actor
from knox.logging import get_logger
from knox.memory import MemoryStore

logger = get_logger(__name__)


def _make_memory_store(wiki_dir: str) -> tuple[MemoryStore, str]:
    """Create a MemoryStore and resolve the acting user from config."""
    cfg = resolve_config(cli_overrides={PATHS_WIKI_DIR: wiki_dir})
    embed_model = cfg.get(COMPILE_EMBEDDING_MODEL)
    return MemoryStore(base_dir=Path(wiki_dir), embedding_model=embed_model), resolve_actor(cfg)
