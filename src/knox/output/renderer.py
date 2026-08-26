"""Render :class:`Document` to one or more Markdown flavors.

``DocumentRenderer`` is the single path for all Markdown output in KNOX.
It produces canonical Markdown (frontmatter + heading + body). The heading
is assembled via :class:`~knox.core.page_builder.PageBuilder` for
consistent AST-driven output.

Flavor-specific formatting via ``rumdl fmt`` has been replaced by wenmode's
AST renderer, which produces consistent formatting without a subprocess.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from knox.core.page_builder import PageBuilder
from knox.logging import get_logger
from knox.output.models import Document

# ── Flavour auto-detection ──────────────────────────────────────────────

FLAVOR_ALIASES: dict[str, str] = {
    "obsidian": "obsidian",
    "gfm": "gfm",
    "github": "gfm",
    "mkdocs": "mkdocs",
    "mdx": "mdx",
    "quarto": "quarto",
    "myst": "myst",
    "standard": "standard",
    "pandoc": "pandoc",
    "kramdown": "kramdown",
    "azure_devops": "azure_devops",
}

logger = get_logger(__name__)


# ── DocumentRenderer ────────────────────────────────────────────────────


class DocumentRenderer:
    """Renders :class:`Document` to a specific Markdown flavor.

    Parameters
    ----------
    flavor:
        Target Markdown flavor (``"obsidian"``, ``"gfm"``, etc.).
        Auto-detected when ``None``.
    output_dir:
        Output directory (used for auto-detection).
    rumdl_config:
        Deprecated — no longer used. Kept for backward compatibility.
    """

    def __init__(
        self,
        flavor: str | None = None,
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


def detect_flavor(output_dir: Path | None = None, configured: str | None = None) -> str:
    """Auto-detect the best Markdown flavor for the output directory.

    Precedence:
    1. Explicitly configured flavor (highest).
    2. Auto-detection from files in the output directory.
    3. ``"obsidian"`` (default).
    """
    if configured:
        canonical = FLAVOR_ALIASES.get(configured.lower())
        if canonical:
            return canonical
        logger.warning("Unknown flavor %r — falling back to obsidian", configured)
        return "obsidian"

    if output_dir and output_dir.exists():
        # Obsidian vault
        if (output_dir / ".obsidian").is_dir():
            return "obsidian"
        # MkDocs project
        if (output_dir / "mkdocs.yml").is_file():
            return "mkdocs"
        # Quarto project
        if (output_dir / "_quarto.yml").is_file():
            return "quarto"
        if list(output_dir.rglob("*.qmd")):
            return "quarto"

    return "obsidian"


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
    flavor: str | None = None,
    rumdl_config: Path | None = None,
    extra_files: list[Path] | None = None,
    check_only: bool = False,
) -> list[Path]:
    """Run ``rumdl fmt`` over generated markdown output.

    Formats every ``.md`` file under *output_dir* plus *extra_files* (e.g. the
    vault-root ``index.md``). Returns the files that changed (or, with
    *check_only*, the files that *would* change). Skips silently when rumdl
    is not installed; never raises.
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
    cmd = ["rumdl", "fmt", "--silent"] if not check_only else ["rumdl", "fmt", "--check"]
    if flavor:
        cmd.extend(["--flavor", flavor])
    if rumdl_config:
        cmd.extend(["--config", str(rumdl_config)])
    cmd.extend(str(p) for p in files)

    try:
        result = subprocess.run(  # noqa: S603 — internally constructed cmd list, no user input
            cmd,
            capture_output=True,
            encoding="utf-8",
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("rumdl fmt failed — skipping output formatting", exc_info=True)
        return []
    if result.returncode not in (0, 1):
        logger.warning("rumdl fmt exited %s: %s", result.returncode, result.stderr.strip()[:300])
        return []

    if check_only:
        # ``rumdl fmt --check`` exits 1 when changes would be made.
        return list(files) if result.returncode == 1 else []
    return [p for p in files if p.read_bytes() != before.get(p)]


# ── Helper ───────────────────────────────────────────────────────────────


def _check_rumdl() -> bool:
    """Check if ``rumdl`` is available on PATH."""
    return shutil.which("rumdl") is not None


def _md_files_under(targets: list[Path]) -> list[Path]:
    """All ``.md`` files directly under *targets* (dirs scanned recursively)."""
    files: list[Path] = []
    for target in targets:
        if target.is_file():
            files.append(target)
        elif target.is_dir():
            files.extend(p for p in target.rglob("*.md") if p.is_file())
    return sorted(set(files))
