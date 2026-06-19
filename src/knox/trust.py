"""Q21 output trust: privacy redaction, private flag, lifecycle states."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from knox.core.models import CompiledWiki, WikiNode
from knox.core.paths import VaultPaths
from knox.provenance import (
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_DRAFT,
    LIFECYCLE_STALE,
    VALID_LIFECYCLE_STATES,
)

logger = logging.getLogger(__name__)

# Patterns used by the privacy filter. Conservative on purpose — false
# positives (over-redaction) are safer than false negatives.
_PRIVACY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # email
)
_PRIVACY_REDACTION_TOKEN: str = "[REDACTED]"  # noqa: S105 — sentinel token, not a credential


def filter_sensitive(text: str) -> tuple[str, int]:
    """Strip known secret/PII patterns from *text*.

    Returns the sanitised text and the number of substitutions.
    """
    if not text:
        return text, 0
    count = 0
    for pattern in _PRIVACY_PATTERNS:
        text, n = pattern.subn(_PRIVACY_REDACTION_TOKEN, text)
        count += n
    return text, count


def has_sensitive_content(text: str) -> bool:
    """Return True if *text* matches any privacy pattern."""
    return any(pattern.search(text) for pattern in _PRIVACY_PATTERNS)


def filter_wiki(wiki: CompiledWiki) -> tuple[CompiledWiki, int]:
    """Apply :func:`filter_sensitive` to every node's content.

    Returns the filtered wiki and the total number of substitutions.
    """
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


def mark_wiki_private(wiki: CompiledWiki, output_dir: Path) -> None:
    """Write a ``.knox/private.json`` index of private pages.

    Private pages are excluded from default search/queries unless the
    caller passes ``--include-private``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    private_index = VaultPaths(output_dir.parent).private_path
    private_index.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pages": [{"id": n.id, "title": n.title} for n in wiki.nodes],
    }
    private_index.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


@dataclass
class LifecycleEntry:
    """Lifecycle metadata for a single wiki page."""

    node_id: str
    state: str = LIFECYCLE_DRAFT
    last_transition: str = ""
    review_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "state": self.state,
            "last_transition": self.last_transition,
            "review_notes": self.review_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LifecycleEntry:
        state = str(data.get("state", LIFECYCLE_DRAFT))
        if state not in VALID_LIFECYCLE_STATES:
            state = LIFECYCLE_DRAFT
        return cls(
            node_id=str(data.get("node_id", "")),
            state=state,
            last_transition=str(data.get("last_transition", "")),
            review_notes=str(data.get("review_notes", "")),
        )


@dataclass
class LifecycleState:
    """The full lifecycle record for a wiki output directory."""

    entries: dict[str, LifecycleEntry] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"entries": {k: v.to_dict() for k, v in self.entries.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LifecycleState:
        return cls(entries={k: LifecycleEntry.from_dict(v) for k, v in data.get("entries", {}).items()})


class LifecycleStore:
    """JSON-backed store for per-page lifecycle state."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path

    def load(self) -> LifecycleState:
        if not self.state_path.exists():
            return LifecycleState()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return LifecycleState()
        if not isinstance(data, dict):
            return LifecycleState()
        return LifecycleState.from_dict(data)

    def save(self, state: LifecycleState) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state.to_dict(), indent=2, default=str), encoding="utf-8")

    def transition(
        self,
        state: LifecycleState,
        node_id: str,
        new_state: str,
        notes: str = "",
    ) -> LifecycleEntry:
        """Move a page to a new lifecycle state."""
        _ = self  # noqa: PLR6301 — instance method for future per-store config
        if new_state not in VALID_LIFECYCLE_STATES:
            msg = f"Invalid lifecycle state {new_state!r}. Valid: {VALID_LIFECYCLE_STATES}"
            raise ValueError(msg)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        existing = state.entries.get(node_id) or LifecycleEntry(node_id=node_id)
        entry = LifecycleEntry(
            node_id=node_id,
            state=new_state,
            last_transition=now,
            review_notes=notes or existing.review_notes,
        )
        state.entries[node_id] = entry
        return entry


def mark_stale(
    state: LifecycleState,
    *,
    stale_after_days: int = 90,
    now: datetime | None = None,
) -> int:
    """Auto-mark pages as ``stale`` after the configured number of days.

    Returns the number of entries that changed state.
    """
    now = now or datetime.now(UTC)
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
