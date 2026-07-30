"""Shared helpers for MCP tools."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from knox.core.paths import vault_paths
from knox.core.config import resolve_config
from knox.core.config.keys import COMPILE_EMBEDDING_MODEL
from knox.core.trust.actor import resolve_actor
from knox.logging import get_logger
from knox.memory import MemoryStore

logger = get_logger(__name__)


def resolve_wiki_dir(wiki_dir: str | None = None) -> str:
    """Resolve the wiki directory from most-specific source.

    Priority: explicit argument > server-wide default (set via
    ``knox mcp --wiki-dir``) > ``KNOX_WIKI_DIR`` env var > current directory.

    Delegates to :func:`knox.core.paths.vault_paths`.
    """
    return str(vault_paths(wiki_dir).base_dir)


def set_default_wiki_dir(path: str) -> None:
    """Set the server-wide default wiki directory."""
    from knox.core.paths import set_default_wiki_dir as _set_default

    _set_default(path)


def make_memory_store(wiki_dir: str) -> tuple[MemoryStore, str]:
    """Create a MemoryStore and resolve the acting user from config."""
    cfg = resolve_config(cli_overrides={PATHS_WIKI_DIR: wiki_dir})
    embed_model = cfg.get(COMPILE_EMBEDDING_MODEL)
    return MemoryStore(base_dir=Path(wiki_dir), embedding_model=embed_model), resolve_actor(cfg)
