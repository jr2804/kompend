"""KNOX document parsers.

Backends are registered lazily: importing this package only exposes the
lightweight base types and the router factory. A backend's ``register_backend``
calls run on demand (see ``_registry._ensure_backend``) when a router is built,
the parser list is queried, or its extensions are requested — never at package
import time. This keeps CLI ``--help`` and unrelated command startup free of
heavy optional dependencies (docling, marker, mineru, torch, …).

``build_default_router`` warns and skips uninstalled backends at runtime.
"""

from __future__ import annotations

from knox.preprocessing.parsers._registry import (
    build_default_router,
    get_registered_parsers,
)
from knox.preprocessing.parsers.base import BaseParser, ParserRouter

__all__ = [
    "BaseParser",
    "ParserRouter",
    "build_default_router",
    "get_registered_parsers",
]
