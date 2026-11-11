"""Attachment (user-provided supplementary file) models.

Attachments are supplementary files (photos, binaries, source code, etc.)
that belong to a source but are not ingested by default. They live in
``attachments/<source-stem>/`` and are tracked in the ``attachment_manifest``
table of ``.kompend/kompend.db`` (the legacy ``attachments.jsonl`` is no longer
written or read).

Each attachment has an explicit ``ingestible`` flag and an optional
``embed`` override for the global ``search.embed_attachments`` config.
"""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants;
# no runtime value ever enters an identifier or statement.

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from kompend.core.paths import ATTACHMENTS_DIR_NAME, VaultPaths, attachment_slugs_for_source, is_protected_source
from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import ATTACHMENT_MANIFEST_TABLE

__all__ = [
    "Attachment",
    "AttachmentLiveness",
    "AttachmentManifest",
    "LiveSlugOutcome",
    "adopt_orphan_attachments",
    "attachments_relative",
    "compute_content_digest",
    "live_attachment_slugs",
]

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

    Stored as one row in the ``attachment_manifest`` table. Re-registering a
    path replaces its row (latest wins); a tombstone row marks a removal.
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

    content_digest: str = ""
    """Digest of the attachment bytes at registration (see
    :func:`compute_content_digest`). ``''`` when unknown (legacy rows).
    The cache identity of per-attachment artifacts (captions, embeddings)
    keys on it, so changed bytes are detected without re-hashing upstream."""

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
            "content_digest": self.content_digest,
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
            content_digest=d.get("content_digest", ""),
            registered_at=d.get("registered_at", ""),
            deleted=d.get("deleted", False),
        )


class AttachmentManifest:
    """Attachment manifest backed by the single vault store.

    One row per ``path`` in the ``attachment_manifest`` table; ``deleted = 1``
    is a tombstone recording an explicit user removal. This is read-equivalent
    to the legacy append-only JSONL ("latest entry per path wins"): re-appending
    a tombstoned path clears the tombstone.

    Paths are stored **attachments-relative** (``<slug>/<file>``). The mutating
    methods (:meth:`append`, :meth:`remove`, :meth:`set_ingestible`) also accept a
    vault-relative ``attachments/<slug>/<file>`` path and normalise it via
    :func:`attachments_relative`, so the two coordinate systems cannot diverge.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._db_path = vault_db_path(path)

    @property
    def path(self) -> Path:
        """The path this manifest was constructed with (legacy artifact path)."""
        return self._path

    def exists(self) -> bool:
        """Whether the store currently holds any attachment row."""
        if not self._db_path.exists():
            return False
        with open_store(self._db_path) as conn:
            row = conn.execute(f"SELECT 1 FROM {ATTACHMENT_MANIFEST_TABLE} LIMIT 1").fetchone()
        return row is not None

    def read_all(self, *, include_deleted: bool = False) -> dict[str, Attachment]:
        """Read attachments keyed by path.

        By default tombstoned rows are excluded; ``include_deleted=True``
        returns them too (the ``--all`` view), with ``Attachment.deleted`` set.
        """
        if not self._db_path.exists():
            return {}
        where = "" if include_deleted else " WHERE deleted = 0"
        with open_store(self._db_path) as conn:
            rows = conn.execute(
                f"SELECT path, source, ingestible, embed, content_digest, registered_at, deleted FROM {ATTACHMENT_MANIFEST_TABLE}{where}",
            ).fetchall()
        return {row["path"]: _attachment_from_row(row) for row in rows}

    def read_for_source(self, source_stem: str) -> list[Attachment]:
        """Read all active attachments for a specific source."""
        return [a for a in self.read_all().values() if a.source == source_stem]

    def append(self, attachment: Attachment) -> None:
        """Upsert an attachment row (re-registering a path clears its tombstone)."""
        self._upsert(attachment)

    def remove(self, attachment_file: str) -> bool:
        """Mark an attachment as deleted (tombstone row).

        *attachment_file* may be vault-relative (``attachments/<slug>/<file>``)
        or attachments-relative (``<slug>/<file>``); see :func:`attachments_relative`.

        Returns:
            ``True`` if the attachment existed and was marked deleted.
        """
        key = attachments_relative(attachment_file)
        existing = self.read_all()
        if key not in existing:
            return False
        self._upsert(Attachment(path=key, source=existing[key].source, deleted=True))
        return True

    def set_ingestible(self, attachment_file: str, ingestible: bool) -> bool:
        """Update the ingestible flag of an attachment.

        Accepts the same path forms as :meth:`remove`.

        Returns:
            ``True`` if the attachment existed and was updated.
        """
        key = attachments_relative(attachment_file)
        existing = self.read_all()
        if key not in existing:
            return False
        att = existing[key]
        self._upsert(
            Attachment(
                path=att.path, source=att.source, ingestible=ingestible, embed=att.embed, content_digest=att.content_digest, registered_at=att.registered_at
            )
        )
        return True

    def list_sources(self) -> list[str]:
        """List all unique source stems with registered attachments."""
        return sorted({a.source for a in self.read_all().values()})

    def deleted_paths(self) -> set[str]:
        """Paths whose latest manifest entry is a tombstone.

        Auto-adoption must skip these: a tombstone is an explicit user
        removal, and re-adopting the file would silently undo it.
        """
        if not self._db_path.exists():
            return set()
        with open_store(self._db_path) as conn:
            rows = conn.execute(f"SELECT path FROM {ATTACHMENT_MANIFEST_TABLE} WHERE deleted = 1").fetchall()
        return {row["path"] for row in rows}

    def _upsert(self, attachment: Attachment) -> None:
        key = attachments_relative(attachment.path)
        with open_store(self._db_path) as conn:
            conn.execute(
                f"""INSERT INTO {ATTACHMENT_MANIFEST_TABLE}(path, source, ingestible, embed, content_digest, registered_at, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    source = excluded.source,
                    ingestible = excluded.ingestible,
                    embed = excluded.embed,
                    content_digest = excluded.content_digest,
                    registered_at = excluded.registered_at,
                    deleted = excluded.deleted""",
                (
                    key,
                    attachment.source,
                    int(attachment.ingestible),
                    None if attachment.embed is None else int(attachment.embed),
                    attachment.content_digest,
                    attachment.registered_at,
                    int(attachment.deleted),
                ),
            )


def attachments_relative(path: str | Path) -> str:
    """Normalise a path to the manifest's key form: ``<slug>/<file>``.

    Two coordinate systems meet here and only here: the orphan lint rule and
    :func:`kompend.lint.structural.orphan_attachment_dirs` emit **vault-relative**
    paths (``attachments/<slug>/<file>``), while ``attachments add`` and
    :func:`adopt_orphan_attachments` store **attachments-relative** keys
    (``<slug>/<file>``). Accepts either form; strips at most one leading
    ``attachments/`` segment and normalises backslashes.
    """
    normalized = str(path).replace("\\", "/").lstrip("/")
    prefix = f"{ATTACHMENTS_DIR_NAME}/"
    return normalized[len(prefix) :] if normalized.startswith(prefix) else normalized


def _attachment_from_row(row: Any) -> Attachment:
    """Build an :class:`Attachment` from an ``attachment_manifest`` row."""
    embed = row["embed"]
    return Attachment(
        path=row["path"],
        source=row["source"],
        ingestible=bool(row["ingestible"]),
        embed=None if embed is None else bool(embed),
        content_digest=row["content_digest"],
        registered_at=row["registered_at"],
        deleted=bool(row["deleted"]),
    )


class LiveSlugOutcome(StrEnum):
    """Why an attachment directory's slug counts as live — or does not."""

    LIVE = "live"
    #: The slug belongs to a source under a protected, tool-managed subdir
    #: (``sources/_captured/`` and friends). Retiring it is a user decision, so
    #: no rule may delete or flag the files.
    PROTECTED = "protected"
    DEAD = "dead"


@dataclass(frozen=True, slots=True)
class AttachmentLiveness:
    """Which attachment slugs map to a source, and which map only to a protected one.

    Both destructive rules and auto-adoption ask the same question — "is this
    attachment directory still addressable from ``sources/``?" — and used to
    answer it with three separate copies of the scan, two of which skipped
    protected sources and one of which did not. That divergence let
    ``orphan-attachments`` report (and, opted in, delete) the figures of a
    captured source while ``unreferenced-attachments`` kept them.
    """

    live: frozenset[str]
    protected: frozenset[str]

    def outcome(self, slug: str) -> LiveSlugOutcome:
        """How *slug* is classified. Live wins: a normal source outranks a protected one."""
        if slug in self.live:
            return LiveSlugOutcome.LIVE
        if slug in self.protected:
            return LiveSlugOutcome.PROTECTED
        return LiveSlugOutcome.DEAD


def live_attachment_slugs(sources_dir: Path) -> AttachmentLiveness:
    """Classify every attachment slug derivable from the sources tree.

    The one implementation of "which slugs are live". ``live`` holds slugs of
    ordinary sources; ``protected`` holds slugs whose only source sits under a
    protected subdir (:func:`is_protected_source`) — the compile lifecycle
    owns those notes, so their attachments are neither orphaned nor prunable.
    Matches both slug forms through :func:`attachment_slugs_for_source`.
    """
    sources_root = Path(sources_dir)
    live: set[str] = set()
    protected: set[str] = set()
    if not sources_root.is_dir():
        return AttachmentLiveness(live=frozenset(), protected=frozenset())
    for path in sources_root.rglob("*"):
        if not path.is_file():
            continue
        slugs = attachment_slugs_for_source(path.relative_to(sources_root).as_posix())
        if is_protected_source(path, sources_root):
            protected |= slugs
        else:
            live |= slugs
    return AttachmentLiveness(live=frozenset(live), protected=frozenset(protected - live))


def adopt_orphan_attachments(base_dir: Path, *, ingest_adopted: bool = True) -> list[Attachment]:
    """Register orphaned attachment files found on disk (auto-adoption).

    Scans ``attachments/<slug>/`` for files without a manifest entry and
    registers them, deriving the flags from the current setup:

    - ``ingestible`` — with *ingest_adopted* set (config key
      ``attachments.ingest_adopted``), text/image extensions are ingestible
      (text via the parser, images via VLM at compile); binaries are
      never ingestible. Without it, adoption only registers the file.
    - ``embed`` — ``None`` (inherits the global ``attachments.embed``).

    Only subdirectories whose slug matches a live source in ``sources/``
    are adopted. A slug that belongs only to a protected source
    (``sources/_captured/`` and friends) is left alone: the lifecycle owns
    that note, and adoption must not mint manifest rows for it. Tombstoned
    paths (explicit ``kompend attachments remove``) are never re-adopted. Runs
    at compile start so recovery scenarios (deleted manifest) and manually
    dropped files are picked up without re-ingest.

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

    live_slugs = live_attachment_slugs(vp.sources_dir).live

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
                content_digest=compute_content_digest(f.read_bytes()),
            )
            manifest.append(att)
            known.add(rel)
            adopted.append(att)
    return adopted


def compute_content_digest(data: bytes) -> str:
    """Content digest of an attachment's bytes (sha256, first 16 hex chars).

    Same shape as the figure-image naming in ``preprocessing/images.py``:
    identical bytes → identical digest, changed bytes → new digest, while
    the attachment's path stays put.
    """
    return hashlib.sha256(data).hexdigest()[:16]
