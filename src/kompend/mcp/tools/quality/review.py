"""MCP review tool - the review queue for compiled pages."""

from __future__ import annotations

from typing import Any

from kompend.cli.settings import build_cli_overrides
from kompend.core.config import resolve_config
from kompend.core.paths import REVIEW_DIR, vault_paths
from kompend.review import ReviewQueue


def review(  # noqa: PLR0917, PLR0911 — MCP tool with many kwargs
    wiki_dir: str | None = None,
    action: str = "list",
    slug: str | None = None,
    reason: str | None = None,
    approve_all: bool = False,
    json_output: bool = False,
) -> dict[str, Any]:
    """Manage the review queue for compiled pages.

    Actions: list, approve, reject, lock, unlock
    """
    vp = vault_paths(wiki_dir)
    resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())
    queue = ReviewQueue(review_dir=vp.kompend_dir / REVIEW_DIR)

    if action == "list":
        entries = queue.list_entries()
        if json_output:
            return {"status": "ok", "entries": [e.to_dict() for e in entries]}
        if not entries:
            return {"status": "ok", "message": "No pages in the review queue."}
        return {"status": "ok", "entries_text": "\n".join(f"  [{e.status:9s}] {e.slug}  ({e.title})" for e in entries)}

    if action == "lock":
        if not slug:
            return {"status": "error", "message": "Lock requires a slug."}
        if queue.lock(slug):
            return {"status": "ok", "message": f"Locked: {slug}"}
        return {"status": "error", "message": f"Cannot lock {slug!r}: no approved page found."}

    if action == "unlock":
        if not slug:
            return {"status": "error", "message": "Unlock requires a slug."}
        lock_file = queue.review_dir / f"{slug}.locked"
        if lock_file and lock_file.exists():
            lock_file.unlink()
            return {"status": "ok", "message": f"Unlocked: {slug}"}
        return {"status": "error", "message": f"No lock for {slug!r}."}

    if action == "approve":
        approved = queue.approve(slug or "", approve_all=approve_all)
        if not approved:
            return {"status": "error", "message": "No matching pages to approve."}
        return {"status": "ok", "approved": [e.slug for e in approved]}

    if action == "reject":
        if not slug:
            return {"status": "error", "message": "Reject requires a slug."}
        rejected = queue.reject(slug, reason=reason or "")
        if not rejected:
            return {"status": "error", "message": f"No pending page with slug {slug!r}."}
        return {"status": "ok", "rejected": [e.slug for e in rejected]}

    return {"status": "error", "message": f"Unknown review action: {action!r}. Use list|approve|reject|lock|unlock."}
