"""Command modules for KNOX CLI.

This package organizes all CLI command definitions into logical modules:

- capture.py   → /knox capture (document capture)
- config.py    → /knox config (configuration management)
- doctor.py    → /knox doctor (health checks)
- init.py      → /knox init (project initialization)
- lint.py      → /knox lint (linting and validation)
- log.py       → /knox log (regenerate pipeline log report)
- memory.py    → /knox memory (memory/RAG operations)
- organize.py  → /knox organize (wiki organization)
- pack.py      → /knox pack (archive packing)
- pipeline.py  → /knox pipeline (tiered compilation pipeline)
- quality.py   → /knox quality (quality diagnostics)
- query.py     → /knox query (vector search / RAG queries)
- review.py    → /knox review (review queue management)
- serve.py     → /knox serve (local HTTP server)
- status.py    → /knox status (wiki statistics)

All commands register themselves with the app.Typer instance on import.
This __init__.py re-exports all modules for backward compatibility.

BACKWARD COMPATIBILITY:
- Old: from knox.cli.lint_cmds import get_lint_report
- New: from knox.cli.cmds.lint import get_lint_report
- Both work via re-exports below.
"""

from __future__ import annotations

import importlib
from types import ModuleType

_MODULE_NAMES: frozenset[str] = frozenset({
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
})


def __getattr__(name: str) -> ModuleType:
    if name in _MODULE_NAMES:
        return importlib.import_module(f"knox.cli.cmds.{name}")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


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


def __dir__() -> list[str]:
    """Include lazy module names in dir()."""
    return sorted(_MODULE_NAMES)
