"""Command modules for KNOX CLI.

This package organizes all CLI command definitions into logical modules:

- capture.py   → /knox capture (document capture)
- config.py    → /knox config (configuration management)
- doctor.py    → /knox doctor (health checks)
- init.py      → /knox init (project initialization)
- lint.py      → /knox lint (linting and validation)
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

from knox.cli.cmds import capture as capture
from knox.cli.cmds import config as config
from knox.cli.cmds import doctor as doctor
from knox.cli.cmds import init as init
from knox.cli.cmds import lint as lint
from knox.cli.cmds import memory as memory
from knox.cli.cmds import organize as organize
from knox.cli.cmds import pack as pack
from knox.cli.cmds import pipeline as pipeline
from knox.cli.cmds import quality as quality
from knox.cli.cmds import query as query
from knox.cli.cmds import review as review
from knox.cli.cmds import serve as serve
from knox.cli.cmds import status as status

__all__ = [
    "capture",
    "config",
    "doctor",
    "init",
    "lint",
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
