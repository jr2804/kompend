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

import lazy_loader as lazy

from knox.preprocessing.parsers._registry import (
    ParserBuildOptions as ParserBuildOptions,
)
from knox.preprocessing.parsers._registry import (
    build_default_router as build_default_router,
)
from knox.preprocessing.parsers._registry import (
    get_backend as get_backend,
)
from knox.preprocessing.parsers._registry import (
    get_registered_parsers as get_registered_parsers,
)
from knox.preprocessing.parsers.base import BaseParser as BaseParser
from knox.preprocessing.parsers.base import ParserRouter as ParserRouter

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "docling_parser": ["DoclingParser"],
    },
)
