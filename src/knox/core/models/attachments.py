"""Attachment (user-provided supplementary file) models.

Attachments are supplementary files (photos, binaries, source code, etc.)
that belong to a source but are not ingested by default. They live in
``assets/<source-stem>/`` and are tracked by an append-only JSONL manifest
at ``.knox/assets.jsonl``.

Each attachment has an explicit ``ingestible`` flag and an optional
``embed`` override for the global ``search.embed_assets`` config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

__all__ = ["Attachment", "AttachmentManifest"]

# ── Ingestible file types ────────────────────────────────────────────────

# Text/markdown extensions — parsed via standard parser pipeline
TEXT_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".md",
        ".txt",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".html",
        ".htm",
        ".rst",
        ".tex",
        ".log",
        ".ini",
        ".cfg",
        ".conf",
        ".py",
        ".js",
        ".ts",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".sh",
        ".bat",
        ".ps1",
        ".sql",
        ".r",
        ".m",
        ".swift",
        ".kt",
        ".scala",
    }
)

# Image extensions — ingested via VLM caption (if VLM configured)
IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
    }
)

# All ingestible extensions
INGESTIBLE_EXTENSIONS: Final[frozenset[str]] = TEXT_EXTENSIONS | IMAGE_EXTENSIONS


@dataclass
class Attachment:
    """A registered attachment (supplementary file) for a source.

    Stored as one JSON line in the manifest (``.knox/assets.jsonl``).
    The manifest is append-only; to update an entry, append a new line
    with the same ``path``. When reading, the latest entry wins.
    """

    path: str
    """Relative to ``assets_dir`` (i.e., ``<slug>/<filename>``)."""

    source: str
    """Source file stem (the canonical link to parent source)."""

    ingestible: bool = False
    """If true, processed as a \"special source\" during compilation.
    Text/markdown via parser; PNG/JPG via VLM caption."""

    embed: bool | None = None
    """Overrides global ``search.embed_assets`` for this attachment.
    ``None`` = inherit global default."""

    registered_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    """ISO timestamp of registration."""

    deleted: bool = False
    """If true, this attachment has been removed (tombstone entry)."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "path": self.path,
            "source": self.source,
            "ingestible": self.ingestible,
            "embed": self.embed,
            "registered_at": self.registered_at,
            "deleted": self.deleted,
        }

    @property
    def filename(self) -> str:
        """The attachment filename (last component of path)."""
        return Path(self.path).name

    @property
    def slug(self) -> str:
        """The source slug (first component of path)."""
        return Path(self.path).parent.as_posix()

    @property
    def extension(self) -> str:
        """Lowercase file extension."""
        return Path(self.path).suffix.lower()

    @property
    def is_image(self) -> bool:
        """Whether this attachment is an ingestible image type."""
        return self.extension in IMAGE_EXTENSIONS

    @property
    def is_text(self) -> bool:
        """Whether this attachment is a text/markdown type."""
        return self.extension in TEXT_EXTENSIONS

    def effective_embed(self, global_embed_assets: bool) -> bool:
        """Resolve the effective embed flag considering global config.

        Args:
            global_embed_assets: The ``search.embed_assets`` config value.

        Returns:
            ``True`` if this attachment should be embedded.
        """
        if self.embed is not None:
            return self.embed
        return global_embed_assets

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Attachment:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            path=d["path"],
            source=d["source"],
            ingestible=d.get("ingestible", False),
            embed=d.get("embed"),
            registered_at=d.get("registered_at", ""),
            deleted=d.get("deleted", False),
        )


class AttachmentManifest:
    """Append-only JSONL manifest tracking registered attachments.

    Each line is a JSON-serialized :class:`Attachment`. To update an entry,
    append a new line with the same ``path``. When reading, the latest
    entry for each ``path`` wins (tombstone entries mark deletions).
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """The manifest file path."""
        return self._path

    def exists(self) -> bool:
        """Check if the manifest file exists."""
        return self._path.exists()

    def read_all(self) -> dict[str, Attachment]:
        """Read all active (non-deleted) attachments, keyed by path.

        Returns:
            Dict mapping attachment path to the latest :class:`Attachment`.
        """
        if not self._path.exists():
            return {}
        entries: dict[str, Attachment] = {}
        for raw_line in self._path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                d = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            att = Attachment.from_dict(d)
            if att.deleted:
                entries.pop(att.path, None)
            else:
                entries[att.path] = att
        return entries

    def read_for_source(self, source_stem: str) -> list[Attachment]:
        """Read all active attachments for a specific source."""
        return [a for a in self.read_all().values() if a.source == source_stem]

    def append(self, attachment: Attachment) -> None:
        """Append an attachment entry to the manifest.

        Args:
            attachment: The attachment to record.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(attachment.to_dict()) + "\n")

    def remove(self, attachment_path: str) -> bool:
        """Mark an attachment as deleted (tombstone entry).

        Args:
            attachment_path: The relative path of the attachment to remove.

        Returns:
            ``True`` if the attachment existed and was marked deleted.
        """
        existing = self.read_all()
        if attachment_path not in existing:
            return False
        tombstone = Attachment(
            path=attachment_path,
            source=existing[attachment_path].source,
            deleted=True,
        )
        self.append(tombstone)
        return True

    def list_sources(self) -> list[str]:
        """List all unique source stems with registered attachments."""
        return sorted({a.source for a in self.read_all().values()})
