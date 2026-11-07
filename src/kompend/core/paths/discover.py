"""Find the vault root and fail fast when the directory is not a vault.

Resolution order is explicit argument → server-wide default (set by the MCP
server) → ``KOMP_WIKI_DIR`` → current working directory. This is the single
discovery path for the CLI, MCP and programmatic callers; everything else
receives a :class:`~kompend.core.paths.names.VaultPaths`.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from kompend.core.env import EnvVar
from kompend.core.paths.names import VAULT_MARKERS, VaultPaths


class _DefaultWikiDir:
    """Holds the server-wide default wiki directory (set by MCP server)."""

    def __init__(self) -> None:
        self._path: str | None = None

    def set(self, path: str) -> None:
        self._path = path

    def get(self) -> str | None:
        return self._path


def set_default_wiki_dir(path: str) -> None:
    """Set the server-wide default wiki directory (used by MCP server)."""
    _default_wiki_dir().set(path)


def vault_paths(wiki_dir: str | Path | None = None, *, require_vault: bool = True) -> VaultPaths:
    """Resolve VaultPaths from the most specific source.

    Priority: explicit arg > server default (set by MCP) >
    ``KOMP_WIKI_DIR`` env var > current working directory.
    This is the single source of truth for resolving vault paths across
    all code paths (CLI, MCP, API).

    Args:
        wiki_dir: explicit vault root, or ``None`` for env/CWD resolution.
        require_vault: fail fast when the resolved directory is not a kompend
            vault root. Only config-introspection paths (``kompend config get``)
            may opt out — they read defaults, not vault data.

    Raises:
        LookupError: when ``require_vault`` is set and the resolved directory
            has no ``kompend.toml`` / ``.kompend/config.toml``. Silent operation on
            arbitrary directories masks misconfiguration — config, vector
            index, and LLM would all silently degrade.
    """
    if wiki_dir is None:
        default = _default_wiki_dir().get()
        # KOMP_WIKI_DIR is defined in the config-map (keys.PATHS_WIKI_DIR); read
        # directly here because vault resolution predates config discovery.
        wiki_dir = default if default is not None else os.environ.get(EnvVar.WIKI_DIR, ".")
    base_dir = Path(wiki_dir).expanduser().resolve()
    if require_vault and not any((base_dir / marker).exists() for marker in VAULT_MARKERS):
        raise LookupError(
            f"{base_dir} is not a kompend vault root: no {VAULT_MARKERS[0]} or "
            f"{VAULT_MARKERS[1]} found. Run 'kompend init' to create a vault, or "
            "point --wiki-dir at the directory containing .kompend/."
        )
    return VaultPaths(base_dir)


@cache
def _default_wiki_dir() -> _DefaultWikiDir:
    """Return the singleton default-wiki-dir holder."""
    return _DefaultWikiDir()
