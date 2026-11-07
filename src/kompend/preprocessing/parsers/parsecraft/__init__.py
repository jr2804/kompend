"""The parsecraft adapter package — one concern per module.

The package name matches the third-party distribution on purpose (it lives
*inside* ``kompend.preprocessing.parsers``, so absolute imports stay
unambiguous). The public surface is re-exported through :mod:`lazy_loader`,
never eagerly: ``adapter`` pulls the parsecraft pipeline (~1.4 s of native
stack), and the parsers package must stay importable without it —
``--help``, suffix discovery and ``list-parsers`` never instantiate a
backend (see ``build_parser`` in ``document_parser.py``, the only factory).

Import discipline for anything placed here: modules inside this package may
import each other one-way (adapter → identity/conversion/assets/pages,
pages → assets), and nothing here may import the adapter back.
"""

from __future__ import annotations

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "adapter": ["ParsecraftParser"],
        "assets": ["ParsecraftParseError"],
        "pages": ["DEGRADED_BLANK_PAGE"],
    },
)
