"""Host memory facts, the plan-aware budget check, and the heavy-parse throttle.

A heavy conversion is a memory cost, not a CPU cost: the docling engine loads
its layout, table-structure and OCR models plus an ONNX arena, which peaks at
about 1830 MB of commit charge for one document at the configured four threads
(2790 MB at the engine's own thread count, 1533 MB at a single thread) —
measured on Windows with docling-rs 2.65.1 (2026-11-04). The peak is a fixed
per-conversion cost, not a per-page one: an 8-page and a 113-page PDF both cost
about the same.

Which engine a source reaches is decided per page by parsecraft's router, and
that decision is what the budget depends on: the same PDF needs a few hundred
megabytes when liteparse reads it and gigabytes when docling does. The budget
check therefore takes the backends the routing plan names —
:func:`ensure_engine_headroom` — and runs in the parser, the one layer that
holds a plan, immediately before the conversion starts. The file suffix only
decides serialization (:class:`HeavyParseGuard`), which is cheap to
over-approximate.

The measurements behind the numbers are recorded in the beads; the thresholds
are config fields, because every behaviour-affecting value in this project is
one.
"""

from __future__ import annotations

import ctypes
import functools
import os
import threading
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Final

from kompend.logging import get_logger
from kompend.preprocessing.parsers.base import FileExtension, known_extension

#: Suffixes whose conversion *may* load a heavy engine, used for
#: serialization. Derived from the canonical extension table, never
#: re-declared: a binary container (:attr:`FileExtension.is_text` is false)
#: may load the engine, a text format never does. The set deliberately
#: over-approximates what the router can route to a heavy backend, because a
#: serialized parse costs wall-clock time while an unguarded one costs memory.
HEAVY_SUFFIXES: Final[frozenset[str]] = frozenset(extension.value for extension in FileExtension if not extension.is_text)

#: Bytes in one megabyte — the unit the config fields speak.
MEGABYTE: Final[int] = 1024 * 1024

#: Backends that extract text with native code and load no model stack, so one
#: parse stays in the hundreds of megabytes. Listed positively because "light"
#: is the claim that needs evidence: every other backend — the ML engines and
#: the OCR models, including one added to parsecraft after this line — is
#: treated as heavy, so the budget check fails safe.
LIGHT_BACKENDS: Final[frozenset[str]] = frozenset(
    {
        "liteparse",
        "native-csv",
        "native-html",
        "native-markdown",
        "native-text",
        "openpyxl",
        "pandoc",
        "pdf-inspector",
        "pymupdf",
        "trafilatura",
    },
)

#: Measured peak commit charge of one conversion, per engine, at this project's
#: defaults. Only measured engines are listed: an unmeasured heavy backend is
#: still guarded, its message just omits a figure instead of inventing one.
MEASURED_ENGINE_COMMIT_MB: Final[Mapping[str, int]] = MappingProxyType({"docling": 1830})

logger = get_logger(__name__)


class InsufficientHostMemoryError(RuntimeError):
    """The host cannot back the memory the requested parse needs.

    Raised before the conversion starts, so the run stops with an actionable
    message instead of dying inside the native extension.
    """


class HeavyParseGuard:
    """Serializes heavy parses — the concurrency half of the memory budget.

    One instance belongs to one run: the semaphore is the run's
    ``max_concurrent`` cap, because N simultaneous conversions cost N times the
    peak. The headroom half lives in :func:`ensure_engine_headroom`, which
    needs the routing plan and therefore belongs to the parser.
    """

    def __init__(self, *, max_concurrent: int = 1) -> None:
        self._semaphore = threading.BoundedSemaphore(max(1, max_concurrent))

    @contextmanager
    def guard(self, source: Path) -> Generator[None]:
        """Hold a heavy-parse slot for *source*; a text source passes through."""
        if not is_heavy_source(source):
            yield
            return
        with self._semaphore:
            yield


def ensure_engine_headroom(source: Path, backends: Iterable[str], *, min_free_commit_bytes: int) -> None:
    """Refuse *source* when the engines its plan names cannot fit in the host.

    Args:
        source: The file the plan is for; named in the refusal message.
        backends: Every backend the routing plan named for this source. Only
            the ones that load a model stack count, so a PDF routed to
            liteparse is never refused for docling's budget.
        min_free_commit_bytes: Headroom the config demands
            (``ingest.parse_min_free_commit_mb``); ``0`` disables the check.

    Raises:
        InsufficientHostMemoryError: A planned engine is heavy and free commit
            charge is below ``min_free_commit_bytes``.
    """
    if min_free_commit_bytes <= 0:
        return
    engines = sorted(heavy_backends(backends))
    if not engines:
        return
    available = available_commit_bytes()
    if available is None:
        _warn_unknown_headroom_once()
        return
    if available >= min_free_commit_bytes:
        return
    raise InsufficientHostMemoryError(_refusal_message(source, engines, available, min_free_commit_bytes))


def heavy_backends(backends: Iterable[str]) -> frozenset[str]:
    """The names in *backends* that may load a multi-gigabyte model stack."""
    return frozenset(name for name in backends if name not in LIGHT_BACKENDS)


def _refusal_message(source: Path, engines: list[str], available: int, required: int) -> str:
    """The refusal text: what was planned, what is short, and the levers.

    The remedies spell the keys the way the config does — ``[ingest]
    parse_min_free_commit_mb`` — so they can be found in `kompend.toml`. The
    terminal handler runs with ``markup=False`` for exactly this reason
    (``logging.configure_logging``): with markup on, Rich read the table name
    as a style tag and the user saw "lower  parse_min_free_commit_mb".
    """
    measured = ""
    for engine in engines:
        peak = MEASURED_ENGINE_COMMIT_MB.get(engine)
        if peak is not None:
            measured += f" ({engine} peaks at about {peak} MB per conversion)"
    steps = ["free memory and run again", "lower [ingest] parse_min_free_commit_mb"]
    if "docling" in engines:
        steps.append(
            "bound docling with [ingest] parsecraft_docling_threads = 1, "
            "[routing.backend_options.docling] do_ocr = false, [routing.backend_options.docling] do_table_structure = false",
        )
    return (
        f"Not enough commit charge to parse {source.name}: its plan routes it to "
        f"{', '.join(engines)}{measured}, and only {available / MEGABYTE:.0f} MB of commit charge is free "
        f"against the {required / MEGABYTE:.0f} MB minimum. {', '.join(steps)}."
    )


@functools.cache
def _warn_unknown_headroom_once() -> None:
    logger.warning(
        "Host commit headroom is unknown on this platform — the parse budget cannot be checked. A conversion may still exhaust the host.",
    )


def available_commit_bytes() -> int | None:
    """Return the commit charge the OS can still back, or ``None`` if unknown.

    On Windows this is the ``ullAvailPageFile`` figure of
    ``GlobalMemoryStatusEx`` — the commit headroom the kernel will honour
    before it starts refusing allocations. On Linux there is no commit
    charge, so the closest equivalent (free physical memory plus free swap)
    is returned and the caller treats it as an estimate.
    """
    if os.name == "nt":
        return _windows_commit_bytes()
    return _posix_commit_bytes()


def _windows_commit_bytes() -> int | None:
    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    kernel32 = ctypes.windll.kernel32
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPageFile)


def _posix_commit_bytes() -> int | None:
    try:
        fields: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            value = rest.strip().split(" ", 1)[0]
            if value.isdigit():
                fields[key] = int(value) * 1024
    except OSError:
        return None
    available = fields.get("MemAvailable")
    if available is None:
        return None
    # No commit charge concept on Linux; free swap is the closest stand-in
    # for the part of the reservation that must be backed by disk.
    return available + fields.get("SwapFree", 0)


def is_heavy_source(source: Path) -> bool:
    """True when the source may load a heavy conversion engine."""
    extension = known_extension(source.suffix)
    return extension is not None and not extension.is_text
