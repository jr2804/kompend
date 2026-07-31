"""The ``knox review`` command (review queue management)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated

import typer

from knox.cli._shared import build_cli_overrides, vault_paths_from_ctx
from knox.cli.app import app
from knox.cli.args import JsonOutput
from knox.core.config import resolve_config
from knox.core.paths import REVIEW_DIR
from knox.logging import get_console, get_err_console
from knox.review import ReviewQueue


@app.command(name="review")
def review_cmd(
    ctx: typer.Context,
    action: Annotated[
        str,
        typer.Argument(help="Action: list | approve | reject | lock | unlock"),
    ],
    slug: Annotated[
        str | None,
        typer.Argument(help="Page slug (required for approve/reject/lock/unlock)."),
    ] = None,
    reason: Annotated[
        str | None,
        typer.Option("--reason", help="Rejection reason (for 'reject')."),
    ] = None,
    approve_all: Annotated[
        bool,
        typer.Option("--all", help="Approve every pending page."),
    ] = False,
    json_output: JsonOutput = False,
) -> None:
    """Manage the review queue for compiled pages."""
    vp = vault_paths_from_ctx(ctx)
    resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())
    queue = ReviewQueue(review_dir=vp.knox_dir / REVIEW_DIR)
    _dispatch_review_action(queue, action, slug, reason, approve_all, json_output)


def _dispatch_review_action(
    queue: ReviewQueue,
    action: str,
    slug: str | None,
    reason: str | None,
    approve_all: bool,
    json_output: bool,
) -> None:
    """Dispatch a review command action via a handler map."""
    handlers: dict[str, Callable[[], None]] = {
        "list": lambda: _review_list(queue, json_output),
        "approve": lambda: _review_approve(queue, slug, approve_all),
        "reject": lambda: _review_reject(queue, slug, reason),
        "lock": lambda: _review_lock(queue, slug),
        "unlock": lambda: _review_unlock(queue, slug),
    }
    handler = handlers.get(action)
    if handler is None:
        get_err_console().print(f"Unknown review action: {action!r}. Use list|approve|reject|lock|unlock.")
        raise typer.Exit(1)
    handler()


def _review_list(queue: ReviewQueue, json_output: bool) -> None:
    entries = queue.list_entries()
    if json_output:
        get_console().print(json.dumps([e.to_dict() for e in entries], indent=2, default=str))
        return
    if not entries:
        get_console().print("No pages in the review queue.")
        return
    for e in entries:
        get_console().print(f"  [{e.status:9s}] {e.slug}  ({e.title})", markup=False)


def _review_lock(queue: ReviewQueue, slug: str | None) -> None:
    if not slug:
        get_err_console().print("Lock requires a slug.")
        raise typer.Exit(1)
    if queue.lock(slug):
        get_console().print(f"Locked: {slug}")
        return
    get_err_console().print(f"Cannot lock {slug!r}: no approved page found.")
    raise typer.Exit(1)


def _review_unlock(queue: ReviewQueue, slug: str | None) -> None:
    lock_file = queue.review_dir / f"{slug}.locked" if slug else None
    if lock_file and lock_file.exists():
        lock_file.unlink()
        get_console().print(f"Unlocked: {slug}")
        return
    get_err_console().print(f"No lock for {slug!r}.")
    raise typer.Exit(1)


def _review_approve(queue: ReviewQueue, slug: str | None, approve_all: bool) -> None:
    approved = queue.approve(slug or "", approve_all=approve_all)
    if not approved:
        get_err_console().print("No matching pages to approve.")
        raise typer.Exit(1)
    for e in approved:
        get_console().print(f"Approved: {e.slug}")


def _review_reject(queue: ReviewQueue, slug: str | None, reason: str | None) -> None:
    if not slug:
        get_err_console().print("Reject requires a slug.")
        raise typer.Exit(1)
    rejected = queue.reject(slug, reason=reason or "")
    if not rejected:
        get_err_console().print(f"No pending page with slug {slug!r}.")
        raise typer.Exit(1)
    for e in rejected:
        get_console().print(f"Rejected: {e.slug}")
