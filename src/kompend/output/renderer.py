"""Render :class:`Document` to one or more Markdown flavors.

``DocumentRenderer`` is the single path for all Markdown output in kompend.
It produces canonical Markdown (frontmatter + heading + body). The heading
is assembled via :class:`~kompend.core.page_builder.PageBuilder` for
consistent AST-driven output.

Flavor-specific formatting via ``rumdl fmt`` has been replaced by wenmode's
AST renderer, which produces consistent formatting without a subprocess.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from kompend.core.config import Flavor, resolve_flavor
from kompend.core.page_builder import PageBuilder
from kompend.core.text.rumdl import rumdl_available
from kompend.core.verbosity import is_verbose
from kompend.logging import get_logger
from kompend.output.models import Document

#: Tail of rumdl's stderr included in the failure warning — enough to show the
#: cause without dumping the whole diagnostic.
_STDERR_PREVIEW_CHARS = 300

#: Character budget for the file arguments of one ``rumdl fmt`` invocation.
#: Windows caps a spawned process' command line at ~32,767 characters
#: (CreateProcess); half that leaves headroom for ``list2cmdline`` quoting of
#: paths with spaces. POSIX ARG_MAX is far larger — the budget only means
#: fewer, bigger batches there.
_COMMAND_LINE_CHAR_BUDGET = 16_000

# ── Flavour auto-detection ──────────────────────────────────────────────

logger = get_logger(__name__)


# ── DocumentRenderer ────────────────────────────────────────────────────


class DocumentRenderer:
    """Renders :class:`Document` to a specific Markdown flavor.

    Parameters
    ----------
    flavor:
        Target Markdown flavor (a :class:`~kompend.core.config.Flavor` member).
        Auto-detected when ``None``.
    output_dir:
        Output directory (used for auto-detection).
    rumdl_config:
        Deprecated — no longer used. Kept for backward compatibility.
    """

    def __init__(
        self,
        flavor: Flavor | None = None,
        output_dir: Path | None = None,
        rumdl_config: Path | None = None,
    ) -> None:
        self.flavor = detect_flavor(output_dir=output_dir, configured=flavor)

    def render(self, doc: Document) -> str:
        """Render a single :class:`Document` to a Markdown string."""
        return self._assemble(doc)

    def render_all(self, docs: list[Document], title: str = "Compiled Wiki") -> str:
        """Render multiple documents as a single Markdown file.

        Each document becomes a section with the appropriate heading level.
        """
        lines: list[str] = [f"# {title}\n"]
        for i, doc in enumerate(docs):
            if i > 0:
                lines.append("")  # spacing between sections
            rendered = self._assemble(doc)
            lines.append(rendered)

        return "\n".join(lines)

    @staticmethod
    def _assemble(doc: Document) -> str:
        """Assemble a single document as canonical Markdown.

        Produces YAML frontmatter (if any), a heading, and the body.
        No flavor-specific transformations.
        """
        parts: list[str] = []

        # Frontmatter
        fm = render_frontmatter(doc.frontmatter)
        if fm:
            parts.append(fm)

        # Heading via PageBuilder for consistent formatting
        b = PageBuilder()
        b.heading(doc.heading_level, doc.title)
        lines = [""] + b.render().split("\n")
        parts.append("\n".join(lines))

        # Body
        if doc.body:
            parts.append(doc.body)
            if not doc.body.endswith("\n"):
                parts.append("")

        return "\n".join(parts).strip() + "\n"


def detect_flavor(output_dir: Path | None = None, configured: Flavor | str | None = None) -> Flavor:
    """Auto-detect the best Markdown flavor for the output directory.

    Precedence:
    1. Explicitly configured flavor (highest) — resolved through
       :func:`~kompend.core.config.resolve_flavor`, which raises on unknown
       names instead of silently degrading the render.
    2. Auto-detection from files in the output directory.
    3. ``Flavor.OBSIDIAN`` (default).
    """
    if configured:
        return resolve_flavor(configured)

    if output_dir and output_dir.exists():
        # Obsidian vault
        if (output_dir / ".obsidian").is_dir():
            return Flavor.OBSIDIAN
        # MkDocs project
        if (output_dir / "mkdocs.yml").is_file():
            return Flavor.MKDOCS
        # Quarto project
        if (output_dir / "_quarto.yml").is_file():
            return Flavor.QUARTO
        if list(output_dir.rglob("*.qmd")):
            return Flavor.QUARTO

    return Flavor.OBSIDIAN


# ── Frontmatter helpers ─────────────────────────────────────────────────


def render_frontmatter(frontmatter: dict[str, Any]) -> str:
    """Render a dict as YAML frontmatter (``---`` delimited).

    Returns an empty string if the dict is empty.
    """
    if not frontmatter:
        return ""
    yaml_str = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True).strip()
    return f"---\n{yaml_str}\n---\n\n"


def format_output(
    output_dir: Path,
    *,
    flavor: Flavor | str | None = None,
    rumdl_config: Path | None = None,
    extra_files: list[Path] | None = None,
    check_only: bool = False,
) -> list[Path]:
    """Run ``rumdl fmt`` over generated markdown output.

    Formats every ``.md`` file under *output_dir* plus *extra_files* (e.g. the
    vault-root ``index.md``). Returns the files that changed (or, with
    *check_only*, the files that *would* change). Skips silently when rumdl
    is not installed; never raises. File arguments are batched
    (:func:`_command_line_batches`) so large vault sweeps stay under the
    Windows command-line limit instead of failing to spawn.
    """
    if not _check_rumdl():
        return []

    targets: list[Path] = [output_dir, *(extra_files or [])]
    files = _md_files_under(targets)
    before = {p: p.read_bytes() for p in files}
    if not files:
        return []

    # Pass explicit file paths, not directories: rumdl filters directory
    # scans through ignore files (.gitignore etc.) and would silently skip
    # generated output.
    base_cmd = ["rumdl", "fmt", "--check"] if check_only else ["rumdl", "fmt", "--silent"]
    if flavor:
        base_cmd.extend(["--flavor", str(flavor)])
    if rumdl_config:
        base_cmd.extend(["--config", str(rumdl_config)])

    would_change = False
    for batch in _command_line_batches(files, base_cmd):
        try:
            result = subprocess.run(  # noqa: S603 — internally constructed cmd list, no user input
                [*base_cmd, *(str(p) for p in batch)],
                capture_output=True,
                encoding="utf-8",
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("rumdl fmt failed (%s) — skipping output formatting", exc, exc_info=is_verbose())
            break
        if result.returncode not in (0, 1):
            logger.warning("rumdl fmt exited %s: %s", result.returncode, result.stderr.strip()[:_STDERR_PREVIEW_CHARS])
            break
        would_change |= result.returncode == 1

    if check_only:
        # ``rumdl fmt --check`` exits 1 when changes would be made.
        return list(files) if would_change else []
    # Diff after every batch, so a mid-sweep failure still reports the pages
    # earlier batches did format.
    return [p for p in files if p.read_bytes() != before.get(p)]


# ── Helper ───────────────────────────────────────────────────────────────


def _check_rumdl() -> bool:
    """Check if ``rumdl`` is available on PATH."""
    return rumdl_available()


def _md_files_under(targets: list[Path]) -> list[Path]:
    """All ``.md`` files directly under *targets* (dirs scanned recursively)."""
    files: list[Path] = []
    for target in targets:
        if target.is_file():
            files.append(target)
        elif target.is_dir():
            files.extend(p for p in target.rglob("*.md") if p.is_file())
    return sorted(set(files))


def _command_line_batches(files: list[Path], base_cmd: list[str]) -> Iterator[list[Path]]:
    """Split *files* into per-invocation batches under a command-line budget.

    Windows caps the command line of a spawned process at ~32,767 characters,
    so one ``rumdl fmt`` over a large vault (hundreds of page paths) fails to
    spawn with ``FileNotFoundError: [WinError 206] The filename or extension
    is too long``. Batching keeps every invocation well inside
    :data:`_COMMAND_LINE_CHAR_BUDGET`; correctness is unchanged — each batch
    formats its own files and results aggregate across batches.
    """
    budget = max(_COMMAND_LINE_CHAR_BUDGET - len(" ".join(base_cmd)) - 1, 1)
    batch: list[Path] = []
    used = 0
    for path in files:
        size = len(str(path)) + 1
        if batch and used + size > budget:
            yield batch
            batch, used = [], 0
        batch.append(path)
        used += size
    if batch:
        yield batch
