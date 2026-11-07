"""The one place a parser instance is built.

:func:`build_parser` replaced a registry that had exactly one backend in it. Every
site that used to call ``build_default_router(ParserOptions.from_config(...))``
now calls this with :class:`ParserOptions`, and gets the same guarded object back.

The adapter is fetched through :func:`importlib.import_module` rather than a
top-level import — the same on-demand mechanism :mod:`kompend.telemetry` and
``kompend.core.pipeline.bounded_parse`` use. It pulls parsecraft's native stack
(~1.4 s), which ``kompend --help`` must not pay for.
"""

from __future__ import annotations

import importlib

from kompend.logging import get_logger
from kompend.preprocessing.parsers.base import DocumentParser
from kompend.preprocessing.parsers.options import ParserOptions

#: The runtime-only import at its call site (``.agents/CODE.md``: no
#: ``TYPE_CHECKING`` blocks — they diverge runtime from the type checker).
_ADAPTER_MODULE = "kompend.preprocessing.parsers.parsecraft"

logger = get_logger(__name__)


def build_parser(options: ParserOptions | None = None) -> DocumentParser:
    """Build the configured parser, guarded by :class:`DocumentParser`.

    Args:
        options: Construction inputs. Defaults to :class:`ParserOptions` when
            ``None``, which is what a caller that only needs the suffix table
            (``kompend list-parsers``, extension discovery) wants.
    """
    parser_cls = importlib.import_module(_ADAPTER_MODULE).ParsecraftParser
    return DocumentParser(parser_cls.from_options(options or ParserOptions()))
