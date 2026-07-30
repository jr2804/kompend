"""Command modules for KNOX CLI.

This package organizes CLI commands into three domain subpackages plus
standalone modules:

Subpackages:

- ``pipeline/``   — ``compile``, ``ingest``, ``parse``, ``config``, ``sources``
- ``quality/``    — ``doctor``, ``lint``, ``review``, ``log``
- ``knowledge/``  — ``query``, ``capture``, ``learn``, ``distill``, ``write``,
                     ``organize``, ``memory``, ``pack``

Standalone modules:

- ``agent.py``    — agent REPL commands
- ``agents.py``   — agent lifecycle management
- ``init.py``     — project scaffolding
- ``serve.py``    — local HTTP server and MCP server

All commands register themselves with the app.Typer instance on import.

BACKWARD COMPATIBILITY:
- Old paths are preserved as re-export wrappers (``from knox.cli.cmds.lint import ...``
  still works by delegating to ``knox.cli.cmds.quality.lint``).
"""

from __future__ import annotations

import importlib
from types import ModuleType

_MODULE_NAMES: frozenset[str] = frozenset(
    {
        "capture",
        "config",
        "doctor",
        "init",
        "lint",
        "log",
        "memory",
        "organize",
        "pack",
        "pipeline",
        "quality",
        "query",
        "review",
        "serve",
        "status",
    }
)


__all__ = [
    "capture",
    "config",
    "doctor",
    "init",
    "lint",
    "log",
    "memory",
    "organize",
    "pack",
    "pipeline",
    "quality",
    "query",
    "review",
    "serve",
    "status",
]


def __getattr__(name: str) -> ModuleType:
    if name in _MODULE_NAMES:
        return importlib.import_module(f"knox.cli.cmds.{name}")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Include lazy module names in dir()."""
    return sorted(_MODULE_NAMES)
