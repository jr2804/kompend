"""Console entry point: answer a bare ``--version`` before the command tree loads.

``kompend.cli.app`` builds the whole command tree at import time — Cyclopts,
Rich, the flag types — and none of it is needed to print one string. The
launcher bounds the numeric thread pools, makes the console decode UTF-8,
answers the version request, and only then installs the things a real run needs:
the console proxy, the failure hooks and the fault handler. Both version paths
read ``kompend.__version__``, so the fast path cannot drift from the flag the
meta app declares.
"""

from __future__ import annotations

import importlib
import sys
from typing import Final

import kompend
from kompend.cli.runtime.console import ensure_utf8_code_page, harden_console_streams
from kompend.cli.runtime.faults import enable_fault_handler, install_failure_hooks
from kompend.cli.runtime.thread_pools import cap_thread_pools
from kompend.cli.tree.registry import OnDemandModule

#: The flags the meta app's eager version option answers on their own.
VERSION_FLAGS: Final[frozenset[str]] = frozenset({"--version", "-v"})


def main() -> None:
    """Run the CLI; a lone version flag never imports the command tree."""
    cap_thread_pools()
    ensure_utf8_code_page()
    if len(sys.argv) == 2 and sys.argv[1] in VERSION_FLAGS:
        print(kompend.__version__)
        return
    harden_console_streams()
    install_failure_hooks()
    enable_fault_handler()
    importlib.import_module(OnDemandModule.APP.value).main()
