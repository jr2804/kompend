"""Bound the numeric thread pools before a native library reads them.

The environment is the only lever: numpy, torch and OpenBLAS read their pool size
when they are imported, so a cap applied after the first import does nothing.
"""

from __future__ import annotations

import os
from typing import Final

from kompend.core.env import EnvVar

#: Thread-pool variables that numeric libraries read at import time. OpenBLAS
#: multiplies every large allocation by its thread count, so an unbounded pool is
#: a memory bomb: on a 16-core host a single docling parse of an ITU-T PDF reached
#: the commit limit and OpenBLAS killed the process outright ("Memory allocation
#: still failed after 10 retries, giving up") instead of raising `MemoryError` —
#: which a caller could have handled. Capping the pool converts that abort into
#: slower arithmetic.
THREAD_POOL_ENV_VARS: Final[tuple[str, ...]] = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: Default cap. Eight threads keep a 16-core box responsive while a parse runs and
#: halve the per-allocation thread buffers; raise it with ``KOMP_BLAS_THREADS``.
DEFAULT_THREAD_CAP: Final[int] = 8


def cap_thread_pools() -> None:
    """Bound the numeric thread pools before numpy, torch or OpenBLAS load.

    Values already present in the environment win — a user who pinned a pool size
    made a decision this default must not overwrite.
    """
    requested = os.environ.get(EnvVar.BLAS_THREADS, "")
    cap = requested.strip() or str(min(DEFAULT_THREAD_CAP, os.cpu_count() or 1))
    for name in THREAD_POOL_ENV_VARS:
        os.environ.setdefault(name, cap)
