"""Console entry point: answer a bare ``--version`` before the command tree loads.

``kompend.cli.app`` builds the whole command tree at import time — Cyclopts,
Rich, the flag types — and none of it is needed to print one string. The
launcher handles the version request first, bounds the numeric thread pools, and
delegates every other invocation unchanged. Both paths read
``kompend.__version__``, so the fast path cannot drift from the flag the meta app
declares.
"""

from __future__ import annotations

import importlib
import os
import sys

import kompend
from kompend.cli._registry import OnDemandModule
from kompend.core.env import EnvVar

#: The flags the meta app's eager version option answers on their own.
_VERSION_FLAGS = frozenset({"--version", "-v"})

#: Thread-pool variables that numeric libraries read at import time. OpenBLAS
#: multiplies every large allocation by its thread count, so an unbounded pool is
#: a memory bomb: on a 16-core host a single docling parse of an ITU-T PDF reached
#: the commit limit and OpenBLAS killed the process outright ("Memory allocation
#: still failed after 10 retries, giving up") instead of raising `MemoryError` —
#: which a caller could have handled. Capping the pool converts that abort into
#: slower arithmetic.
_THREAD_POOL_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: Default cap. Eight threads keep a 16-core box responsive while a parse runs and
#: halve the per-allocation thread buffers; raise it with ``KOMP_BLAS_THREADS``.
_DEFAULT_THREAD_CAP = 8


def main() -> None:
    """Run the CLI; a lone version flag never imports the command tree."""
    _cap_thread_pools()
    if len(sys.argv) == 2 and sys.argv[1] in _VERSION_FLAGS:
        print(kompend.__version__)
        return
    importlib.import_module(OnDemandModule.APP.value).main()


def _cap_thread_pools() -> None:
    """Bound the numeric thread pools before numpy, torch or OpenBLAS load.

    Values already present in the environment win — a user who pinned a pool size
    made a decision this default must not overwrite.
    """
    requested = os.environ.get(EnvVar.BLAS_THREADS, "")
    cap = requested.strip() or str(min(_DEFAULT_THREAD_CAP, os.cpu_count() or 1))
    for name in _THREAD_POOL_ENV_VARS:
        os.environ.setdefault(name, cap)
