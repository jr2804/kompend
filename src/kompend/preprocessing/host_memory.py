"""Host memory facts and the guard that keeps a parse inside the host's budget.

A heavy PDF conversion is not a CPU cost, it is a memory cost: the docling
engine loads its layout, table-structure and OCR models plus an ONNX arena,
which peaks at roughly 5.4 GB of working set and 8.0 GB of commit charge for
one document — measured on Windows with docling-rs 2.55.4 (2026-11-03). The
peak is a fixed per-conversion cost, not a per-page one: an 8-page and a
113-page PDF both cost about the same.

Two consequences shape this module:

* the OS may refuse to back the commit charge, and the failure arrives as an
  abort inside the native extension (or as an unrelated allocation failure in
  numpy) long after the request that caused it. A parse that cannot fit is
  therefore refused *before* it starts, with a message that names the remedy.
* several heavy parses running at once cost several times the headroom, so
  they are serialized.

The measurements behind the numbers here are recorded in the beads; the
thresholds themselves are config fields, because every behaviour-affecting
value in this project is one.
"""

from __future__ import annotations

import ctypes
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from kompend.logging import get_logger
from kompend.preprocessing.parsers.base import FileExtension, known_extension

#: Suffixes whose conversion may load the heavy PDF/office engine. Derived
#: from the canonical extension table, never re-declared: a binary container
#: (:attr:`FileExtension.is_text` is false) may load the engine, a text
#: format never does (native and liteparse need a few hundred megabytes at
#: most). The set deliberately over-approximates the formats a backend can
#: actually parse today (``.gif`` parses nothing at the moment) — the guard
#: only runs for sources that parse, so an unparseable suffix costs nothing
#: and a newly parseable binary format is covered from day one.
HEAVY_SUFFIXES: Final[frozenset[str]] = frozenset(extension.value for extension in FileExtension if not extension.is_text)

#: Bytes in one megabyte — the unit the config fields speak.
MEGABYTE: Final[int] = 1024 * 1024

#: The measured peak of one docling conversion (working set / commit charge),
#: quoted in the refusal message so the number the user sees is the number
#: that was measured, not a guess.
MEASURED_DOCLING_COMMIT_BYTES: Final[int] = 8 * 1000 * MEGABYTE

logger = get_logger(__name__)


class InsufficientHostMemoryError(RuntimeError):
    """The host cannot back the memory the requested parse needs.

    Raised before the conversion starts, so the run stops with an actionable
    message instead of dying inside the native extension.
    """


class HeavyParseGuard:
    """Serializes heavy parses and refuses them when the host has no headroom.

    One instance belongs to one run: the semaphore is the run's concurrency
    cap and the headroom check is the run's memory budget.
    """

    def __init__(self, *, min_free_commit_bytes: int = 0, max_concurrent: int = 1) -> None:
        self._min_free_commit_bytes = max(0, min_free_commit_bytes)
        self._semaphore = threading.BoundedSemaphore(max(1, max_concurrent))
        self._warned_unknown = False

    @contextmanager
    def guard(self, source: Path) -> Generator[None]:
        """Hold the heavy-parse budget for *source*.

        A light source passes through untouched — it never loads the engine
        that needs the budget. A heavy source takes the semaphore (so at most
        ``max_concurrent`` of them run at once) and is refused when the host
        cannot back the conversion.
        """
        if not is_heavy_source(source):
            yield
            return
        with self._semaphore:
            self._check_headroom(source)
            yield

    def _check_headroom(self, source: Path) -> None:
        if self._min_free_commit_bytes == 0:
            return
        available = available_commit_bytes()
        if available is None:
            if not self._warned_unknown:
                self._warned_unknown = True
                logger.warning(
                    "Host commit headroom is unknown on this platform — the heavy-parse guard cannot check it. A conversion may still exhaust the host.",
                )
            return
        if available >= self._min_free_commit_bytes:
            return
        raise InsufficientHostMemoryError(
            f"Not enough commit charge for a heavy parse of {source.name}: "
            f"{available / MEGABYTE:.0f} MB free, {self._min_free_commit_bytes / MEGABYTE:.0f} MB required "
            f"(one docling conversion peaks at about {MEASURED_DOCLING_COMMIT_BYTES / MEGABYTE:.0f} MB). "
            "Free memory and run again, lower [ingest] parse_min_free_commit_mb, or bound the engine "
            "itself with [ingest] parsecraft_docling_threads = 1 and "
            "[ingest] parsecraft_backend_options.docling = { do_ocr = false, do_table_structure = false }.",
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
    """True when the source may load the heavy conversion engine."""
    extension = known_extension(source.suffix)
    return extension is not None and not extension.is_text
