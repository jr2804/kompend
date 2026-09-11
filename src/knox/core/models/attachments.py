"""Attachment (user-provided supplementary file) models.

Attachments are supplementary files (photos, binaries, source code, etc.)
that belong to a source but are not ingested by default. They live in
``attachments/<source-stem>/`` and are tracked by an append-only JSONL
manifest at ``.knox/attachments.jsonl``.

Each attachment has an explicit ``ingestible`` flag and an optional
``embed`` override for the global ``search.embed_attachments`` config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from knox.core.paths import VaultPaths, attachment_slugs_for_source, is_protected_source

__all__ = ["Attachment", "AttachmentManifest", "adopt_orphan_attachments"]

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

    Stored as one JSON line in the manifest (``.knox/attachments.jsonl``).
    The manifest is append-only; to update an entry, append a new line
    with the same ``path``. When reading, the latest entry wins.
    """

    path: str
    """Relative to ``attachments_dir`` (i.e., ``<slug>/<filename>``)."""

    source: str
    """Source file stem (the canonical link to parent source)."""

    ingestible: bool = False
    """If true, processed as a \"special source\" during compilation.
    Text/markdown via parser; PNG/JPG via VLM caption."""

    embed: bool | None = None
    """Overrides global ``search.embed_attachments`` for this attachment.
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

    def effective_embed(self, global_embed_attachments: bool) -> bool:
        """Resolve the effective embed flag considering global config.

        Args:
            global_embed_attachments: The ``search.embed_attachments`` config value.

        Returns:
            ``True`` if this attachment should be embedded.
        """
        if self.embed is not None:
            return self.embed
        return global_embed_attachments

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

    def remove(self, attachment_file: str) -> bool:
        """Mark an attachment as deleted (tombstone entry).

        Args:
            attachment_file: The relative path of the attachment to remove.

        Returns:
            ``True`` if the attachment existed and was marked deleted.
        """
        existing = self.read_all()
        if attachment_file not in existing:
            return False
        tombstone = Attachment(
            path=attachment_file,
            source=existing[attachment_file].source,
            deleted=True,
        )
        self.append(tombstone)
        return True

    def set_ingestible(self, attachment_file: str, ingestible: bool) -> bool:
        """Update the ingestible flag of an attachment (append-only update).

        Appends a new entry with the same ``path`` and the new flag; the
        latest entry wins on read. Independent of the global
        ``attachments.ingest_adopted`` gate — that gate is applied at
        compile time on top of this per-attachment flag.

        Returns:
            ``True`` if the attachment existed and was updated.
        """
        existing = self.read_all()
        if attachment_file not in existing:
            return False
        att = existing[attachment_file]
        updated = Attachment(
            path=att.path,
            source=att.source,
            ingestible=ingestible,
            embed=att.embed,
            registered_at=att.registered_at,
        )
        self.append(updated)
        return True

    def list_sources(self) -> list[str]:
        """List all unique source stems with registered attachments."""
        return sorted({a.source for a in self.read_all().values()})

    def deleted_paths(self) -> set[str]:
        """Paths whose latest manifest entry is a tombstone.

        Auto-adoption must skip these: a tombstone is an explicit user
        removal, and re-adopting the file would silently undo it.
        """
        if not self._path.exists():
            return set()
        last_deleted: dict[str, bool] = {}
        for raw_line in self._path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                d = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            p = str(d.get("path", ""))
            if not p:
                continue
            last_deleted[p] = bool(d.get("deleted"))
        return {p for p, deleted in last_deleted.items() if deleted}


def adopt_orphan_attachments(base_dir: Path, *, ingest_adopted: bool = True) -> list[Attachment]:
    """Register orphaned attachment files found on disk (auto-adoption).

    Scans ``attachments/<slug>/`` for files without a manifest entry and
    registers them, deriving the flags from the current setup:

    - ``ingestible`` — with *ingest_adopted* set (config key
      ``attachments.ingest_adopted``), text/image extensions are ingestible
      (text via the parser router, images via VLM at compile); binaries are
      never ingestible. Without it, adoption only registers the file.
    - ``embed`` — ``None`` (inherits the global ``attachments.embed``).

    Only subdirectories whose slug matches a live source in ``sources/``
    are adopted. Tombstoned paths (explicit ``knox attachments remove``)
    are never re-adopted. Runs at compile start so recovery scenarios
    (deleted manifest) and manually dropped files are picked up without
    re-ingest.

    Returns the list of newly registered attachments (empty when nothing
    was adopted).
    """
    vp = VaultPaths(base_dir)
    attachments_root = vp.attachments_dir
    if not attachments_root.is_dir():
        return []

    manifest = AttachmentManifest(vp.attachments_manifest_file)
    known = set(manifest.read_all())
    tombstoned = manifest.deleted_paths()

    sources_dir = vp.sources_dir
    live_slugs: set[str] = set()
    if sources_dir.is_dir():
        for p in sources_dir.rglob("*"):
            if p.is_file() and not is_protected_source(p, sources_dir):
                # Match both slug forms (full relative path + bare filename)
                # so sources under subdirectories are adoptable — see
                # attachment_slugs_for_source docstring.
                live_slugs.update(attachment_slugs_for_source(p.relative_to(sources_dir).as_posix()))

    adopted: list[Attachment] = []
    for slug_dir in sorted(attachments_root.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name not in live_slugs:
            continue
        for f in sorted(slug_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(attachments_root).as_posix()
            if rel in known or rel in tombstoned:
                continue
            att = Attachment(
                path=rel,
                source=slug_dir.name,
                ingestible=ingest_adopted and f.suffix.lower() in INGESTIBLE_EXTENSIONS,
            )
            manifest.append(att)
            known.add(rel)
            adopted.append(att)
    return adopted
