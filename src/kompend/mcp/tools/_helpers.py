"""Shared helpers for MCP tools."""

from __future__ import annotations

from pathlib import Path

from kompend.core.config import resolve_config
from kompend.core.keys import COMPILE_EMBEDDING_MODEL
from kompend.core.paths import set_default_wiki_dir as _set_default_wiki_dir
from kompend.core.paths import vault_paths
from kompend.core.trust.actor import resolve_actor
from kompend.core.utils.dicts import as_str
from kompend.logging import get_logger
from kompend.memory import MemoryStore

logger = get_logger(__name__)


def resolve_wiki_dir(wiki_dir: str | None = None) -> str:
    """Resolve the wiki directory from most-specific source.

    Priority: explicit argument > server-wide default (set via
    ``kompend mcp --wiki-dir``) > ``KOMP_WIKI_DIR`` env var > current directory.

    Delegates to :func:`kompend.core.paths.vault_paths`.
    """
    return str(vault_paths(wiki_dir).base_dir)


def set_default_wiki_dir(path: str) -> None:
    """Set the server-wide default wiki directory."""
    _set_default_wiki_dir(path)


def make_memory_store(wiki_dir: str) -> tuple[MemoryStore, str]:
    """Create a MemoryStore and resolve the acting user from config."""
    cfg = resolve_config(base_dir=wiki_dir)
    embed_model = as_str(cfg.get(COMPILE_EMBEDDING_MODEL))
    return (
        MemoryStore(base_dir=Path(wiki_dir), embedding_model=embed_model, display=cfg.kompend.display),
        resolve_actor(cfg),
    )
