"""rumdl invocation for markdown *text* and per-element normalisation.

``output/renderer.format_output`` owns the file-sweep variant (used by
``knox lint format`` and the post-compile pass). This module owns the two
text-level variants:

- :func:`format_markdown_text` — one ``rumdl fmt`` call over an in-memory
  string (stdin), used to normalise a rendered document.
- :func:`normalize_element_markdown` — the parse-time pass. Every markdown
  element of a parsed document is wrapped as a standalone block, formatted in
  a **single** subprocess call, then unwrapped. That means the cached parsed
  content is already canonical and backend differences (docling, liteparse,
  marker, mineru, web, ...) are levelled at the part level, not only at the
  final wiki output.

Both are best-effort: a missing ``rumdl`` binary or a formatting failure
leaves the input untouched rather than breaking a compile.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Final

from knox.core.models import DocumentElement
from knox.logging import get_logger
from knox.preprocessing.parsers.markdown import element_to_markdown, markdown_to_element

__all__ = [
    "format_markdown_text",
    "normalize_element_markdown",
    "rumdl_available",
]

#: Delimiter between wrapped elements in a batched call. HTML comments are
#: opaque to rumdl (it never rewrites or reflows them), so the batch can be
#: split back losslessly.
_PART_MARKER: Final = "knox-part"

_TIMEOUT_SECONDS: Final = 300

logger = get_logger(__name__)


def normalize_element_markdown(
    elements: Iterable[DocumentElement],
    *,
    rumdl_config: Path | None = None,
    flavor: str | None = None,
) -> int:
    """Rewrite markdown element contents through one batched rumdl call.

    Returns the number of elements whose content changed. Non-markdown
    elements (tables, images, metadata) are skipped — their representation is
    already canonical and a markdown formatter must not touch them.
    """
    targets: list[tuple[DocumentElement, str]] = []
    for element in elements:
        fragment = element_to_markdown(element)
        if fragment is not None:
            targets.append((element, fragment))
    if not targets:
        return 0

    batch = "\n\n".join(f"<!--{_PART_MARKER}-{index}-->\n\n{fragment}" for index, (_element, fragment) in enumerate(targets))
    formatted = format_markdown_text(batch, flavor=flavor, rumdl_config=rumdl_config)
    if formatted is None:
        return 0

    parts = _split_batch(formatted, expected=len(targets))
    if parts is None:
        logger.debug("rumdl batch split mismatch — leaving element content unchanged")
        return 0

    changed = 0
    for (element, _fragment), part in zip(targets, parts, strict=True):
        normalized = markdown_to_element(element.type, part)
        if normalized is None or normalized == element.content:
            continue
        element.content = normalized
        changed += 1
    return changed


def format_markdown_text(
    text: str,
    *,
    flavor: str | None = None,
    rumdl_config: Path | None = None,
) -> str | None:
    """Run ``rumdl fmt`` over *text* via stdin.

    Returns the formatted markdown, or ``None`` when rumdl is unavailable or
    the call failed (callers keep the original text).
    """
    if not text.strip() or not rumdl_available():
        return None
    cmd = ["rumdl", "fmt", "--silent"]
    if flavor:
        cmd.extend(["--flavor", flavor])
    if rumdl_config:
        cmd.extend(["--config", str(rumdl_config)])
    cmd.append("-")
    try:
        result = subprocess.run(  # noqa: S603 — internally constructed cmd list, no user input
            cmd,
            input=text,
            capture_output=True,
            encoding="utf-8",
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.debug("rumdl fmt on text failed", exc_info=True)
        return None
    return result.stdout or None


def rumdl_available() -> bool:
    """True when the ``rumdl`` binary is on PATH."""
    return shutil.which("rumdl") is not None


def _split_batch(formatted: str, *, expected: int) -> list[str] | None:
    """Split a batched rumdl result back into element fragments.

    Returns ``None`` when the marker count does not match *expected* — the
    caller then keeps every original content rather than mis-assigning.
    """
    chunks: list[str] = []
    current: list[str] | None = None
    for line in formatted.splitlines():
        if line.strip().startswith(f"<!--{_PART_MARKER}-"):
            if current is not None:
                chunks.append("\n".join(current))
            current = []
            continue
        if current is not None:
            current.append(line)
    if current is not None:
        chunks.append("\n".join(current))
    if len(chunks) != expected:
        return None
    return chunks
