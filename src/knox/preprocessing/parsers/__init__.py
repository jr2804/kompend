"""KNOX document parsers.

Each optional parser frontend is imported defensively: if its heavy
dependency is not installed, the import is skipped and the backend is
simply unavailable in ``PARSER_BACKENDS``. ``build_default_router``
warns and skips uninstalled backends at runtime.
"""

from __future__ import annotations

from knox.preprocessing.parsers._registry import build_default_router, get_registered_parsers
from knox.preprocessing.parsers.base import BaseParser, ParserRouter

__all__ = [
    "BaseParser",
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
    from knox.preprocessing.parsers.glm_ocr_parser import GlmOcrParser  # noqa: F401 — triggers register_backend

    __all__.append("GlmOcrParser")
except ImportError:  # pragma: no cover - optional dep
    pass

try:
    from knox.preprocessing.parsers.liteparse_parser import LiteParseParser  # noqa: F401 — triggers register_backend

    __all__.append("LiteParseParser")
except ImportError:  # pragma: no cover - optional dep
    pass

try:
    from knox.preprocessing.parsers.marker_parser import MarkerParser  # noqa: F401 — triggers register_backend

    __all__.append("MarkerParser")
except ImportError:  # pragma: no cover - optional dep
    pass

try:
    from knox.preprocessing.parsers.pandoc_parser import PandocParser  # noqa: F401 — triggers register_backend

    __all__.append("PandocParser")
except ImportError:  # pragma: no cover - optional dep
    pass

try:
    from knox.preprocessing.parsers.web_parser import WebPageParser  # noqa: F401 — triggers register_backend

    __all__.append("WebPageParser")
except ImportError:  # pragma: no cover - optional dep
    pass
