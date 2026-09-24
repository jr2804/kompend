"""Resolve docling's LibreOffice backend without per-clone environment setup.

Docling shells out to LibreOffice for the OLE formats (``.doc``, ``.rtf``). Its
own probe is ``DOCLING_LIBREOFFICE_CMD`` → ``which("libreoffice")`` →
``which("soffice")`` → a macOS app path, with **no Windows default-path probe**.
So on a machine where LibreOffice is installed in ``Program Files`` but not on
``PATH``, docling fails unless the env var is set — a per-clone setup step.

This module adds the missing discovery: :func:`ensure_libreoffice_env` fills in
``DOCLING_LIBREOFFICE_CMD`` (via ``setdefault``, never overwriting an explicit
user value) when it can find a binary. It is called lazily, only when an OLE
format is actually parsed.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Final

from knox.logging import get_logger

#: Docling's env var for a LibreOffice executable (its public override knob).
ENV_VAR: Final[str] = "DOCLING_LIBREOFFICE_CMD"

#: Extensions docling routes through LibreOffice (OLE containers, not OOXML).
LIBREOFFICE_EXTENSIONS: Final[frozenset[str]] = frozenset({".doc", ".rtf"})

_WINDOWS_EXE: Final[Path] = Path("LibreOffice") / "program" / "soffice.exe"
_WINDOWS_ROOT_ENV: Final[tuple[str, ...]] = ("PROGRAMFILES", "PROGRAMFILES(X86)")
_MAC_SOFFICE: Final[Path] = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")

logger = get_logger(__name__)


def needs_libreoffice(source_file: Path) -> bool:
    """True when *source_file*'s extension requires the LibreOffice backend."""
    return source_file.suffix.lower() in LIBREOFFICE_EXTENSIONS


def ensure_libreoffice_env() -> Path | None:
    r"""Point docling at a LibreOffice binary, resolving one when unconfigured.

    Resolution order (first hit wins):

    1. ``DOCLING_LIBREOFFICE_CMD`` already set — respected verbatim (never
       overwritten, even when we would resolve differently). Returns its path
       when it exists, otherwise ``None``.
    2. ``libreoffice`` / ``soffice`` on ``PATH``.
    3. Platform default install location: Windows ``%PROGRAMFILES%`` /
       ``%PROGRAMFILES(X86)%`` → ``LibreOffice\\program\\soffice.exe``, and the
       macOS app bundle.
    4. Nothing found — returns ``None`` and logs nothing: docling emits its own
       actionable error naming the env var when it needs a binary.

    A value resolved by (2)/(3) is installed via ``setdefault`` and logged once
    at INFO. Idempotent and cheap; callers invoke it only for OLE formats.
    """
    configured = os.environ.get(ENV_VAR)
    if configured:
        path = Path(configured)
        if path.exists():
            return path
        logger.debug("%s is set to %s but that path does not exist; leaving it untouched", ENV_VAR, configured)
        return None

    found = _from_path() or _from_platform_default()
    if found is None:
        return None

    os.environ.setdefault(ENV_VAR, str(found))
    logger.info("LibreOffice found at %s; using it for .doc/.rtf parsing (set %s to override)", found, ENV_VAR)
    return found


def _from_path() -> Path | None:
    """A LibreOffice binary on ``PATH`` (docling's own candidate names)."""
    for name in ("libreoffice", "soffice"):
        located = shutil.which(name)
        if located:
            return Path(located)
    return None


def _from_platform_default() -> Path | None:
    """A LibreOffice binary at its conventional install location, if any."""
    for root_var in _WINDOWS_ROOT_ENV:
        root = os.environ.get(root_var)
        if root:
            candidate = Path(root) / _WINDOWS_EXE
            if candidate.is_file():
                return candidate
    if _MAC_SOFFICE.is_file():
        return _MAC_SOFFICE
    return None
