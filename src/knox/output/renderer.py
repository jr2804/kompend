"""Render :class:`Document` to one or more Markdown flavors.

``DocumentRenderer`` is the single path for all Markdown output in KNOX.
It produces canonical Markdown (frontmatter + body) and then optionally
pipes through ``rumdl fmt --flavor <FLAVOR>`` for flavor-specific formatting.

Anti-pattern enforced: no hand-rolled Markdown converters. All flavor
handling goes through rumdl. If a flavor is not supported, contribute
upstream to ``rvben/rumdl``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from knox.output.models import Document

logger = logging.getLogger(__name__)

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
        # Check for .qmd files
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
        Path to a custom ``.rumdl.toml`` config file to pass to ``rumdl fmt``.
    """

    def __init__(
        self,
        flavor: str | None = None,
        output_dir: Path | None = None,
        rumdl_config: Path | None = None,
    ) -> None:
        self.flavor = detect_flavor(output_dir=output_dir, configured=flavor)
        self.rumdl_config = rumdl_config
        self._rumdl_available = _check_rumdl()

    def render(self, doc: Document) -> str:
        """Render a single :class:`Document` to a Markdown string.

        Uses ``rumdl fmt --flavor <FLAVOR>`` when available; otherwise
        returns the canonical Markdown.
        """
        raw = self._assemble(doc)
        if self._rumdl_available:
            return self._format_via_rumdl(raw)
        return raw

    def render_all(self, docs: list[Document], title: str = "Compiled Wiki") -> str:
        """Render multiple documents as a single Markdown file.

        Each document becomes a section with the appropriate heading level.
        The whole file is optionally formatted via rumdl.
        """
        lines: list[str] = [f"# {title}\n"]
        for i, doc in enumerate(docs):
            if i > 0:
                lines.append("")  # spacing between sections
            # Render the individual document
            rendered = self._assemble(doc)
            lines.append(rendered)

        result = "\n".join(lines)
        if self._rumdl_available:
            return self._format_via_rumdl(result)
        return result

    def _assemble(self, doc: Document) -> str:
        """Assemble a single document as canonical Markdown.

        Produces YAML frontmatter (if any), a heading, and the body.
        No flavor-specific transformations.
        """
        parts: list[str] = []

        # Frontmatter
        fm = render_frontmatter(doc.frontmatter)
        if fm:
            parts.append(fm)

        # Heading
        prefix = "#" * min(doc.heading_level, 6)
        parts.append(f"{prefix} {doc.title}\n")

        # Body
        if doc.body:
            parts.append(doc.body)
            if not doc.body.endswith("\n"):
                parts.append("")

        return "\n".join(parts).strip() + "\n"

    def _format_via_rumdl(self, markdown: str) -> str:
        """Pipe *markdown* through ``rumdl fmt --flavor <FLAVOR>``.

        Falls back to raw markdown if rumdl is not installed.
        """
        if not self._rumdl_available:
            return markdown

        cmd = ["rumdl", "fmt", "--flavor", self.flavor]
        if self.rumdl_config:
            cmd.extend(["--config", str(self.rumdl_config)])

        try:
            result = subprocess.run(
                cmd,
                input=markdown,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
            if result.stderr:
                logger.debug("rumdl stderr: %s", result.stderr.strip())
            return markdown
        except FileNotFoundError:
            logger.warning("rumdl not found — disabling flavor formatting")
            self._rumdl_available = False
            return markdown
        except subprocess.TimeoutExpired:
            logger.warning("rumdl timed out — falling back to raw markdown")
            return markdown
        except OSError as exc:
            logger.warning("rumdl error: %s — falling back to raw markdown", exc)
            return markdown


# ── Helper ───────────────────────────────────────────────────────────────


def _check_rumdl() -> bool:
    """Check if ``rumdl`` is available on PATH."""
    return shutil.which("rumdl") is not None
