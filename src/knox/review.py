"""N8 review queue for human-in-the-loop wiki approval.

Storage layout inside ``.knox/review/``:

- ``<slug>.md`` — pending page (text)
- ``<slug>.approved.md`` — approved
- ``<slug>.rejected.md`` — rejected (optionally with reason)
- ``<slug>.locked`` — empty marker file preventing future re-compile

The queue is deliberately file-based so it can be diffed, edited in
any text editor, and committed to git for traceability.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from slugify import slugify as _slugify_impl

from knox.core.models import WikiNode
from knox.core.utils.time import iso_local
from knox.logging import get_logger

logger = get_logger(__name__)


def slugify(title: str) -> str:
    """Convert a title into a filesystem-safe slug."""
    result = _slugify_impl(title)
    return result if result else "untitled"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ReviewEntry:
    """A single review queue entry."""

    slug: str
    status: ReviewStatus
    title: str
    reason: str = ""
    source_node_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "status": self.status,
            "title": self.title,
            "reason": self.reason,
            "source_node_id": self.source_node_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewEntry:
        return cls(
            slug=str(data.get("slug", "")),
            status=ReviewStatus(str(data.get("status", "pending"))),
            title=str(data.get("title", "")),
            reason=str(data.get("reason", "")),
            source_node_id=str(data.get("source_node_id", "")),
        )


class ReviewQueue:
    """File-backed review queue for compiled wiki nodes.

    Parameters
    ----------
    review_dir:
        Directory where pending/approved/rejected pages are stored.
    """

    STATE_FILENAME = "_state.json"

    def __init__(self, review_dir: Path) -> None:
        self.review_dir = review_dir

    # ------------------------------------------------------------------
    # Enqueue / approve / reject
    # ------------------------------------------------------------------

    def enqueue(self, node: WikiNode) -> ReviewEntry:
        """Add a wiki node to the pending queue."""
        self.review_dir.mkdir(parents=True, exist_ok=True)
        slug = slugify(node.title)
        pending_path = self._pending_path(slug)
        if not pending_path.exists():
            pending_path.write_text(node.content, encoding="utf-8")
        entry = ReviewEntry(
            slug=slug,
            status="pending",
            title=node.title,
            source_node_id=node.id,
        )
        self._upsert_state(entry)
        return entry

    def approve(self, slug: str, approve_all: bool = False) -> list[ReviewEntry]:
        """Move pending entries to approved. ``approve_all=True`` approves everything pending."""
        state = self._load_state()
        approved: list[ReviewEntry] = []
        for entry in state.values():
            if entry.status != "pending":
                continue
            if not approve_all and entry.slug != slug:
                continue
            self._move(entry.slug, pending=True, approved=True)
            entry.status = "approved"
            state[entry.slug] = entry
            approved.append(entry)
        self._save_state(state)
        return approved

    def reject(self, slug: str, reason: str = "") -> list[ReviewEntry]:
        """Move pending entries to rejected."""
        state = self._load_state()
        rejected: list[ReviewEntry] = []
        for entry in state.values():
            if entry.status != "pending" or entry.slug != slug:
                continue
            self._move(entry.slug, pending=True, rejected=True)
            entry.status = "rejected"
            entry.reason = reason
            state[entry.slug] = entry
            rejected.append(entry)
        self._save_state(state)
        return rejected

    def lock(self, slug: str) -> bool:
        """Mark an approved page as locked — re-compile will not overwrite it."""
        if not self._approved_path(slug).exists():
            return False
        lock_path = self.review_dir / f"{slug}.locked"
        lock_path.touch()
        return True

    def is_locked(self, slug: str) -> bool:
        return (self.review_dir / f"{slug}.locked").exists()

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_entries(self, status: str | None = None) -> list[ReviewEntry]:
        """Return entries, optionally filtered by status."""
        if status is not None and not isinstance(status, ReviewStatus):
            status = ReviewStatus(status)
        return [e for e in self._load_state().values() if status is None or e.status == status]

    # ------------------------------------------------------------------
    # Internal path helpers
    # ------------------------------------------------------------------

    def _pending_path(self, slug: str) -> Path:
        return self.review_dir / f"{slug}.md"

    def _approved_path(self, slug: str) -> Path:
        return self.review_dir / f"{slug}.approved.md"

    def _rejected_path(self, slug: str) -> Path:
        return self.review_dir / f"{slug}.rejected.md"

    def _state_path(self) -> Path:
        return self.review_dir / self.STATE_FILENAME

    def _load_state(self) -> dict[str, ReviewEntry]:
        if not self._state_path().exists():
            return {}
        try:
            data = json.loads(self._state_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {k: ReviewEntry.from_dict(v) for k, v in data.get("entries", {}).items()}

    def _save_state(self, state: dict[str, ReviewEntry]) -> None:
        self.review_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": iso_local(),
            "entries": {k: v.to_dict() for k, v in state.items()},
        }
        self._state_path().write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _upsert_state(self, entry: ReviewEntry) -> None:
        state = self._load_state()
        state[entry.slug] = entry
        self._save_state(state)

    def _move(self, slug: str, *, pending: bool = False, approved: bool = False, rejected: bool = False) -> None:
        """Move a single file from pending to its destination."""
        src = self._pending_path(slug)
        if not src.exists():
            return
        if approved:
            shutil.move(str(src), str(self._approved_path(slug)))
        elif rejected:
            shutil.move(str(src), str(self._rejected_path(slug)))


def approved_pages(review_dir: Path) -> Iterable[Path]:
    """Yield paths of all approved pages in a review directory."""
    if not review_dir.exists():
        return
    yield from sorted(review_dir.glob("*.approved.md"))
