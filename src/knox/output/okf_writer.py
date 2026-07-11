"""OKF (Open Knowledge Format) v0.1 output writer (default format).

Produces wiki output compatible with the Google Open Knowledge Format spec:
https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md

This is the **default** output format. Use ``--format markdown`` or
``compile.output.format = "markdown"`` to select the legacy Obsidian vault mode.

OKF conformance requirements (§9):
1. Every .md file has parseable YAML frontmatter.
2. Every frontmatter has a non-empty ``type`` field.
3. Reserved filenames (``index.md``, ``log.md``) follow §6/§7 structure.

Key differences from MarkdownWriter (Obsidian vault mode):
- Frontmatter uses OKF fields: ``type``, ``title``, ``description``, ``resource``,
  ``tags``, ``timestamp``. KNOX-specific fields (``kind``, ``compiled_at``,
  ``sources``) are preserved as extensions (§4.1 permits additional keys).
- Cross-links use absolute bundle-relative paths: ``[Title](/concepts/title.md)``
  instead of ``[[Title]]`` (§5.1 recommends absolute links).
- ``index.md`` has grouped sections with descriptions (§6). Root index declares
  ``okf_version: "0.1"`` — the only frontmatter permitted in an index (§11).
- ``log.md`` uses ISO date headings with bold action prefix (§7).
- ``# Citations`` section with numbered ``[N]`` refs from provenance (§8).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import yaml

from knox.core.models import CompiledWiki, MetadataField
from knox.core.paths import INDEX_FILE, LOG_FILE
from knox.core.timeutil import iso_local
from knox.output.markdown import MarkdownWriter
from knox.output.models import Document

# OKF spec version this writer targets.
_OKF_VERSION = "0.1"


# Custom YAML Dumper that never emits anchors/aliases (avoids &id001/*id001
# for repeated list values, which is valid YAML but confuses some consumers).
class _NoAliasDumper(yaml.Dumper):
    """YAML Dumper that disables anchor/alias emission."""

    def ignore_aliases(self, data: object) -> bool:  # type: ignore[override]  # noqa: PLR6301
        return True


def _okf_frontmatter(frontmatter: dict[str, Any]) -> str:
    """Render frontmatter as YAML without anchors/aliases."""
    yaml_str = yaml.dump(
        frontmatter,
        Dumper=_NoAliasDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=True,
    ).strip()
    return f"---\n{yaml_str}\n---\n\n"


# Map KNOX node_type values to OKF ``type`` values.
# OKF types are free-form strings (§4.1); these are descriptive and
# self-explanatory per the spec's recommendation.
_KIND_TO_TYPE: dict[str, str] = {
    "concept": "Concept",
    "comparison": "Comparison",
    "entity": "Entity",
    "overview": "Overview",
    "procedure": "Procedure",
    "section": "Section",
}
_DEFAULT_TYPE = "Concept"

# Regex to find [[wikilinks]] or [[wikilinks|alias]] in body text.
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def _okf_type(node_type: str | None) -> str:
    """Map a KNOX node_type to an OKF type string."""
    if not node_type:
        return _DEFAULT_TYPE
    return _KIND_TO_TYPE.get(node_type, node_type.title() or _DEFAULT_TYPE)


def _okf_description(node: Any) -> str:
    """Extract a one-line description from a wiki node.

    Uses the first non-empty line of content, truncated to ~120 chars.
    """
    content = (node.content or "").strip()
    if not content:
        return ""
    first_line = content.split("\n")[0].strip()
    # Strip [[wikilinks]] to plain text (use alias or target name).
    first_line = _WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), first_line)
    if len(first_line) > 120:
        return first_line[:117].rstrip() + "…"
    return first_line


def convert_wikilinks(body: str, slug_to_path: dict[str, str]) -> str:
    """Convert ``[[wikilinks]]`` to OKF absolute markdown links.

    ``slug_to_path`` maps page slugs/titles to bundle-relative paths
    (e.g. ``"my-page" -> "/concepts/my-page.md"``).
    Unknown targets are left as plain text (OKF §5.3: broken links are valid).
    """

    def _replace(match: re.Match[str]) -> str:
        target = match.group(1)
        alias = match.group(2) or target
        path = slug_to_path.get(target)
        if path is not None:
            return f"[{alias}]({path})"
        return alias

    return _WIKILINK_RE.sub(_replace, body)


def _format_citations(provenance: list[str]) -> str:
    """Format a ``# Citations`` section from provenance sources (§8).

    Each source becomes a numbered ``[N]`` entry. URLs are linked directly;
    bare filenames are rendered as plain text.
    """
    if not provenance:
        return ""
    lines = ["# Citations\n"]
    for i, src in enumerate(provenance, 1):
        src_str = str(src)
        if src_str.startswith("http"):
            lines.append(f"[{i}] [{src_str}]({src_str})")
        else:
            lines.append(f"[{i}] {Path(src_str).name}")
    return "\n".join(lines) + "\n"


class OkfWriter(MarkdownWriter):
    """Write compiled wiki as an OKF v0.1 knowledge bundle.

    Produces the same directory structure as vault-mode MarkdownWriter
    (concepts/ subfolder, root index), but with OKF-compliant frontmatter,
    cross-links, index, log, and citations.
    """

    name: ClassVar[str] = "okf"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(vault_mode=True, **kwargs)

    def _wiki_to_documents(self, wiki: CompiledWiki) -> list[Document]:  # noqa: PLR6301
        """Build Documents with OKF-compliant frontmatter."""
        docs: list[Document] = []
        compiled_at = iso_local()

        for node in wiki.nodes:
            okf_type = _okf_type(node.node_type)
            description = _okf_description(node)
            prov = node.metadata.get(MetadataField.PROVENANCE) or []
            source_list = sorted({Path(str(p)).stem for p in prov if p}) if prov else []

            frontmatter: dict[str, Any] = {
                "type": okf_type,
                "title": node.title or "Untitled",
            }
            if description:
                frontmatter["description"] = description
            if source_list:
                frontmatter["tags"] = source_list
            frontmatter["timestamp"] = compiled_at
            # KNOX extensions (§4.1 permits additional keys)
            frontmatter["kind"] = node.node_type or "page"
            frontmatter["compiled_at"] = compiled_at
            if source_list:
                frontmatter["sources"] = source_list

            doc = Document(
                title=node.title or "Untitled",
                body=node.content or "",
                frontmatter=frontmatter,
                heading_level=node.metadata.get("heading_level", 1),
                source_chunks=node.source_chunks,
                page_slug=node.id,
                node_type=node.node_type,
            )
            docs.append(doc)

        return docs

    def _write_vault(self, docs: list[Document], wiki: CompiledWiki, output_dir: Path) -> Path:
        """Write OKF bundle: concept pages + index.md + log.md."""
        # Clear stale output (same logic as MarkdownWriter).
        for stale in output_dir.glob("*.md"):
            stale.unlink()
        for folder in set(self._KIND_FOLDERS.values()):
            sub = output_dir / folder
            if sub.is_dir():
                for stale in sub.glob("*.md"):
                    stale.unlink()

        # Build slug/title -> bundle-relative path mapping for link conversion.
        slug_to_path: dict[str, str] = {}
        for doc in docs:
            folder = self._folder_for_kind(doc.node_type)
            path = f"/{folder}/{doc.page_slug or 'untitled'}.md" if folder else f"/{doc.page_slug or 'untitled'}.md"
            slug_to_path[doc.page_slug] = path
            slug_to_path[doc.title] = path

        # Build related edges (reuse parent's logic).
        node_map = {doc.page_slug: doc.title for doc in docs if doc.page_slug}
        related_edges = self._build_related_edges(wiki.edges, node_map)

        for doc in docs:
            folder = self._folder_for_kind(doc.node_type)
            target_dir = output_dir / folder if folder else output_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{doc.page_slug or 'untitled'}.md"

            # Render frontmatter + heading + body (no flavor formatting).
            rendered = _okf_frontmatter(doc.frontmatter)
            prefix = "#" * min(doc.heading_level, 6)
            rendered += f"{prefix} {doc.title}\n\n"

            # Convert wikilinks to OKF absolute links.
            body = convert_wikilinks(doc.body, slug_to_path)
            rendered += body
            if body and not body.endswith("\n"):
                rendered += "\n"

            # Related section with standard markdown links.
            entries = related_edges.get(doc.page_slug, [])
            if entries:
                rendered += "\n## Related\n\n"
                for title, rel_type, confidence in entries[: self.max_related_links]:
                    path = slug_to_path.get(title, "")
                    link = f"[{title}]({path})" if path else title
                    annotation = f" ({rel_type})" if rel_type else ""
                    if confidence is not None:
                        annotation += f" — confidence {confidence:.2f}"
                    rendered += f"- {link}{annotation}\n"

            # Citations from provenance (§8).
            prov = next(
                (n.metadata.get(MetadataField.PROVENANCE) for n in wiki.nodes if n.id == doc.page_slug),
                None,
            )
            if prov:
                citations = _format_citations(sorted({str(p) for p in prov if p}))
                if citations:
                    rendered += "\n" + citations

            (target_dir / filename).write_text(rendered, encoding="utf-8")

        # OKF-compliant index and log.
        self._write_okf_index(docs, output_dir, wiki)
        self._write_okf_log(output_dir, wiki)
        return output_dir / INDEX_FILE

    def _write_okf_index(
        self,
        docs: list[Document],
        output_dir: Path,
        wiki: CompiledWiki,
    ) -> None:
        """Write ``index.md`` per §6: grouped sections with descriptions.

        Root index MAY declare ``okf_version`` in frontmatter (§11 exception).
        """
        # Group docs by OKF type for sectioned listing.
        by_type: dict[str, list[Document]] = {}
        for doc in docs:
            okf_type = doc.frontmatter.get("type", _DEFAULT_TYPE)
            by_type.setdefault(okf_type, []).append(doc)

        # Root index.md with okf_version frontmatter (§11).
        lines: list[str] = []
        lines.append("---")
        lines.append(f'okf_version: "{_OKF_VERSION}"')
        lines.append("---")
        lines.append("")
        title = wiki.metadata.get("title", "Knowledge Bundle")
        lines.append(f"# {title}\n")

        for type_name in sorted(by_type):
            group = by_type[type_name]
            lines.append(f"# {type_name}\n")
            for doc in sorted(group, key=lambda d: d.title):
                folder = self._folder_for_kind(doc.node_type)
                rel_path = f"{folder}/{doc.page_slug}.md" if folder else f"{doc.page_slug}.md"
                description = doc.frontmatter.get("description", "")
                if description:
                    lines.append(f"- [{doc.title}](/{rel_path}) - {description}")
                else:
                    lines.append(f"- [{doc.title}](/{rel_path})")
            lines.append("")

        (output_dir / INDEX_FILE).write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _write_okf_log(output_dir: Path, wiki: CompiledWiki) -> None:
        """Write ``log.md`` per §7: ISO date headings, bold action prefix.

        Creates or updates ``log.md`` at the bundle root (not hidden).
        """
        log_path = output_dir / LOG_FILE
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        article_count = len(wiki.nodes)
        source_count = wiki.metadata.get(MetadataField.SOURCE_COUNT, 0)

        entry = f"**Creation**: Compiled {article_count} page(s) from {source_count} source(s).\n"

        existing = ""
        if log_path.exists():
            existing = log_path.read_text(encoding="utf-8")

        # Check if today's date heading already exists.
        date_heading = f"## {today}"
        if date_heading in existing:
            # Prepend entry under existing date heading.
            existing = existing.replace(date_heading, f"{date_heading}\n{entry}", 1)
            log_path.write_text(existing, encoding="utf-8")
        else:
            # Prepend new date section.
            new_section = f"# Bundle Update Log\n\n{date_heading}\n{entry}\n"
            if existing:
                # Strip old title if present, prepend new section before first date heading.
                stripped = re.sub(r"^# Bundle Update Log\n+", "", existing)
                log_path.write_text(new_section + stripped, encoding="utf-8")
            else:
                log_path.write_text(new_section, encoding="utf-8")
