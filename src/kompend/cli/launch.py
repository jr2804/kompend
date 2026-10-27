"""Console entry point: answer a bare ``--version`` before Typer is imported.

``kompend.cli.app`` builds the whole command tree at import time - Typer,
Click, Rich, the flag types - and none of it is needed to print one string.
The launcher handles the version request first and delegates every other
invocation unchanged. Both paths read ``kompend.__version__``, so the fast
path cannot drift from the flag the callback declares.
"""

from __future__ import annotations

import importlib
import sys

import kompend
from kompend.cli._registry import OnDemandModule

#: The flags the callback's eager version option answers on their own.
_VERSION_FLAGS = frozenset({"--version", "-v"})


def main() -> None:
    """Run the CLI; a lone version flag never imports the command tree."""
    if len(sys.argv) == 2 and sys.argv[1] in _VERSION_FLAGS:
        print(kompend.__version__)
        return
    importlib.import_module(OnDemandModule.APP.value).main()
