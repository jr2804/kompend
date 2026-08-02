"""KNOX document parsers.

Each optional parser frontend is imported defensively: if its heavy
dependency is not installed, the import is skipped and the backend is
simply unavailable in ``PARSER_BACKENDS``. ``build_default_router``
warns and skips uninstalled backends at runtime.
"""

from __future__ import annotations

from knox.preprocessing.parsers._registry import build_default_router, get_registered_parsers
from knox.preprocessing.parsers.base import BaseParser, ParserRouter
from knox.preprocessing.parsers.liteparse_parser import LiteParseParser
from knox.preprocessing.parsers.pandoc_parser import PandocParser

__all__ = [
    "BaseParser",
    "LiteParseParser",
    "PandocParser",
    "ParserRouter",
    "build_default_router",
    "get_registered_parsers",
]

# Optional frontends — guarded so a missing dependency never breaks the package.
try:
    from knox.preprocessing.parsers.docling_parser import DoclingParser  # noqa: F401 — triggers register_backend

    __all__.append("DoclingParser")
except ImportError:  # pragma: no cover - optional dep
    pass

try:
    from knox.preprocessing.parsers.web_parser import WebPageParser  # noqa: F401 — triggers register_backend

    __all__.append("WebPageParser")
except ImportError:  # pragma: no cover - optional dep
    pass

try:
    from knox.preprocessing.parsers.marker_parser import MarkerParser  # noqa: F401 — triggers register_backend

    __all__.append("MarkerParser")
except ImportError:  # pragma: no cover - optional dep
    pass
