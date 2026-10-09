"""KNOX document parsers.

Exactly one backend is registered per router — the one named by
``[ingest] parser``, which is ``DEFAULT_PARSER`` (parsecraft) unless the
operator names another. A backend's ``register_backend`` call runs on demand
(see ``_registry._ensure_backend``) when a router is built, the parser list is
queried, or its extensions are requested — never at package import time. That
keeps CLI ``--help`` and unrelated command startup free of heavy optional
dependencies.

``build_default_router`` raises when the named backend is unknown, or when a
known backend's optional dependency is missing (naming the install command).
"""

from __future__ import annotations

from knox.preprocessing.parsers._registry import (
    DEFAULT_PARSER as DEFAULT_PARSER,
)
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

__all__ = [
    "DEFAULT_PARSER",
    "BaseParser",
    "ParserBuildOptions",
    "ParserRouter",
    "build_default_router",
    "get_backend",
    "get_registered_parsers",
]
