"""Privacy redaction, private flag, lifecycle states for compiled output."""

# ruff: noqa: S608 — SQL is assembled only from Final[str] table-name constants.

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from kompend.core.models import (
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_DRAFT,
    LIFECYCLE_STALE,
    VALID_LIFECYCLE_STATES,
    CompiledWiki,
    WikiNode,
)
from kompend.core.store import open_store, vault_db_path
from kompend.core.store.schema import PRIVATE_PAGES_TABLE
from kompend.core.utils.time import iso_local, now_local
from kompend.logging import get_logger

_PRIVACY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
)
_PRIVACY_REDACTION_TOKEN: str = "[REDACTED]"  # noqa: S105 - redaction placeholder, not a secret

logger = get_logger(__name__)


@dataclass
class LifecycleEntry:
    node_id: str
    state: str = LIFECYCLE_DRAFT
    last_transition: str = ""
    review_notes: str = ""


@dataclass
class LifecycleState:
    entries: dict[str, LifecycleEntry] = field(default_factory=dict)


class LifecycleStore:
    """JSON-backed lifecycle store."""

    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file

    def load(self) -> LifecycleState:
        if not self.state_file.exists():
            return LifecycleState()
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return LifecycleState()
        if not isinstance(data, dict):
            return LifecycleState()
        entries = data.get("entries", {})
        return LifecycleState(
            entries={
                k: LifecycleEntry(
                    node_id=entry.get("node_id", k),
                    state=entry.get("state", LIFECYCLE_DRAFT),
                    last_transition=entry.get("last_transition", ""),
                    review_notes=entry.get("review_notes", ""),
                )
                for k, entry in entries.items()
                if isinstance(entry, dict)
            }
        )

    def save(self, state: LifecycleState) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(
                {
                    "entries": {
                        k: {"node_id": v.node_id, "state": v.state, "last_transition": v.last_transition, "review_notes": v.review_notes}
                        for k, v in state.entries.items()
                    }
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def transition(
        state: LifecycleState,
        node_id: str,
        new_state: str,
        notes: str = "",
    ) -> LifecycleEntry:
        """Move a page to a new lifecycle state."""
        if new_state not in VALID_LIFECYCLE_STATES:
            msg = f"Invalid lifecycle state {new_state!r}. Valid: {VALID_LIFECYCLE_STATES}"
            raise ValueError(msg)
        now = iso_local()
        existing = state.entries.get(node_id) or LifecycleEntry(node_id=node_id)
        entry = LifecycleEntry(
            node_id=node_id,
            state=new_state,
            last_transition=now,
            review_notes=notes or existing.review_notes,
        )
        state.entries[node_id] = entry
        return entry


def has_sensitive_content(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PRIVACY_PATTERNS)


def filter_wiki(wiki: CompiledWiki) -> tuple[CompiledWiki, int]:
    """Apply filter_sensitive to every node. Returns (filtered, total_redactions)."""
    total = 0
    new_nodes: list[WikiNode] = []
    for node in wiki.nodes:
        sanitised, n = filter_sensitive(node.content)
        total += n
        if n:
            new_nodes.append(
                WikiNode(
                    id=node.id,
                    title=node.title,
                    content=sanitised,
                    node_type=node.node_type,
                    source_chunks=node.source_chunks,
                    metadata={**(node.metadata or {}), "privacy_redactions": n},
                )
            )
        else:
            new_nodes.append(node)
    return (
        CompiledWiki(
            nodes=new_nodes,
            edges=list(wiki.edges),
            toc=list(wiki.toc),
            quality_score=wiki.quality_score,
            iteration=wiki.iteration,
            metadata={**wiki.metadata, "privacy_filtered": total > 0},
        ),
        total,
    )


def filter_sensitive(text: str) -> tuple[str, int]:
    """Strip secrets/PII, return (sanitised, count)."""
    if not text:
        return text, 0
    count = 0
    for pattern in _PRIVACY_PATTERNS:
        text, n = pattern.subn(_PRIVACY_REDACTION_TOKEN, text)
        count += n
    return text, count


def mark_wiki_private(wiki: CompiledWiki, base_dir: Path) -> None:
    """Record the private-page index in the ``private_pages`` table.

    Snapshot semantics: the previous index is replaced.
    """
    saved_at = iso_local()
    with open_store(vault_db_path(base_dir)) as conn:
        conn.execute(f"DELETE FROM {PRIVATE_PAGES_TABLE}")
        conn.executemany(
            f"INSERT OR REPLACE INTO {PRIVATE_PAGES_TABLE} (id, title, saved_at) VALUES (?, ?, ?)",
            [(n.id, n.title, saved_at) for n in wiki.nodes],
        )


def mark_stale(
    state: LifecycleState,
    *,
    stale_after_days: int = 90,
    now: datetime | None = None,
) -> int:
    """Auto-mark pages as stale after the configured days. Returns count changed."""
    now = now or now_local()
    changed = 0
    for entry in state.entries.values():
        if entry.state in (LIFECYCLE_STALE, LIFECYCLE_ARCHIVED):
            continue
        if not entry.last_transition:
            continue
        try:
            last_dt = datetime.fromisoformat(entry.last_transition)
        except ValueError:
            continue
        if now - last_dt >= timedelta(days=stale_after_days):
            entry.state = LIFECYCLE_STALE
            state.entries[entry.node_id] = entry
            changed += 1
    return changed
