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

# ── Optional parser backends ────────────────────────────────────────────
# Each block is guarded so a missing third-party package does not break the
# package import. The parser module's ``register_backend`` call runs only when
# its dependency is available.
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
    from knox.preprocessing.parsers.mineru_parser import MinerUParser  # noqa: F401 — triggers register_backend

    __all__.append("MinerUParser")
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
