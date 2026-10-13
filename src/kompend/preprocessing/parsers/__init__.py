"""kompend document parsers.

Exactly one backend is registered: parsecraft, the ingestion pipeline
(``DEFAULT_PARSER``). There is no config key selecting it — parsecraft routes
each page across its own sub-backends internally, so there is nothing left for a
kompend-side selector to choose between. A backend's ``register_backend`` call runs
on demand (see ``_registry._ensure_backend``) when a router is built, the parser
list is queried, or its extensions are requested — never at package import time.
That keeps CLI ``--help`` and unrelated command startup free of heavy optional
dependencies.

``build_default_router`` raises when the named backend is unknown, or when a
known backend's optional dependency is missing (naming the install command).
"""

from __future__ import annotations

from kompend.preprocessing.parsers._registry import (
    DEFAULT_PARSER as DEFAULT_PARSER,
)
from kompend.preprocessing.parsers._registry import (
    ParserBuildOptions as ParserBuildOptions,
)
from kompend.preprocessing.parsers._registry import (
    build_default_router as build_default_router,
)
from kompend.preprocessing.parsers._registry import (
    get_backend as get_backend,
)
from kompend.preprocessing.parsers._registry import (
    get_registered_parsers as get_registered_parsers,
)
from kompend.preprocessing.parsers.base import BaseParser as BaseParser
from kompend.preprocessing.parsers.base import ParserRouter as ParserRouter

__all__ = [
    "DEFAULT_PARSER",
    "BaseParser",
    "ParserBuildOptions",
    "ParserRouter",
    "build_default_router",
    "get_backend",
    "get_registered_parsers",
]
