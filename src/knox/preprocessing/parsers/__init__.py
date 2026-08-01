"""KNOX document parsers.

Each parser backend is an *optional* dependency. Only the backends whose
third-party packages are installed are available at runtime — importing
``knox.preprocessing.parsers`` never fails just because a parser package is
missing. Use ``knox list-parsers`` to see which backends are active, or
``build_default_router`` / ``get_backend`` to access them.
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
    from knox.preprocessing.parsers.docling_parser import DoclingParser

    __all__ += ["DoclingParser"]
except ImportError:
    pass

try:
    from knox.preprocessing.parsers.liteparse_parser import LiteParseParser

    __all__ += ["LiteParseParser"]
except ImportError:
    pass

try:
    from knox.preprocessing.parsers.mineru_parser import MinerUParser

    __all__ += ["MinerUParser"]
except ImportError:
    pass

try:
    from knox.preprocessing.parsers.pandoc_parser import PandocParser

    __all__ += ["PandocParser"]
except ImportError:
    pass

try:
    from knox.preprocessing.parsers.web_parser import WebPageParser

    __all__ += ["WebPageParser"]
except ImportError:
    pass
