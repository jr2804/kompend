"""KNOX document parsers."""

from __future__ import annotations

from knox.preprocessing.parsers._registry import build_default_router, get_registered_parsers
from knox.preprocessing.parsers.base import BaseParser, ParserRouter
from knox.preprocessing.parsers.docling_parser import DoclingParser
from knox.preprocessing.parsers.liteparse_parser import LiteParseParser
from knox.preprocessing.parsers.mineru_parser import MinerUParser
from knox.preprocessing.parsers.pandoc_parser import PandocParser
from knox.preprocessing.parsers.web_parser import WebPageParser

__all__ = [
    "BaseParser",
    "DoclingParser",
    "LiteParseParser",
    "MinerUParser",
    "PandocParser",
    "ParserRouter",
    "WebPageParser",
    "build_default_router",
    "get_registered_parsers",
]
