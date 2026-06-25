"""The ``knox review`` command (review queue management)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated

import typer

from knox.cli._shared import build_cli_overrides, resolve_project_paths
from knox.cli.app import app
from knox.cli.args import JsonOutput
from knox.core.config import resolve_config
from knox.core.paths import REVIEW_DIR
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
    wiki_dir_override = ctx.obj.get("wiki_dir") if ctx.obj else None
    cfg = resolve_config(cli_overrides=build_cli_overrides(wiki_dir=wiki_dir_override))
    _sources_dir, _output_dir, _cache_dir, state_dir = resolve_project_paths(cfg)
    queue = ReviewQueue(review_dir=state_dir / REVIEW_DIR)
    _dispatch_review_action(queue, action, slug, reason, approve_all, json_output)


def _review_list(queue: ReviewQueue, json_output: bool) -> None:  # ty: ignore
    entries = queue.list_entries()  # ty: ignore
    if json_output:
        typer.echo(json.dumps([e.to_dict() for e in entries], indent=2, default=str))
        return
    if not entries:
        typer.echo("No pages in the review queue.")
        return
    for e in entries:
        typer.echo(f"  [{e.status:9s}] {e.slug}  ({e.title})")


def _review_lock(queue: ReviewQueue, slug: str | None) -> None:  # ty: ignore
    if not slug:
        typer.echo("Lock requires a slug.", err=True)
        raise typer.Exit(1)
    if queue.lock(slug):  # ty: ignore
        typer.echo(f"Locked: {slug}")
        return
    typer.echo(f"Cannot lock {slug!r}: no approved page found.", err=True)
    raise typer.Exit(1)


def _review_unlock(queue: ReviewQueue, slug: str | None) -> None:  # ty: ignore
    lock_file = queue.review_dir / f"{slug}.locked" if slug else None  # ty: ignore
    if lock_file and lock_file.exists():
        lock_file.unlink()
        typer.echo(f"Unlocked: {slug}")
        return
    typer.echo(f"No lock for {slug!r}.", err=True)
    raise typer.Exit(1)


def _review_approve(queue: ReviewQueue, slug: str | None, approve_all: bool) -> None:  # ty: ignore
    approved = queue.approve(slug or "", approve_all=approve_all)  # ty: ignore
    if not approved:
        typer.echo("No matching pages to approve.", err=True)
        raise typer.Exit(1)
    for e in approved:
        typer.echo(f"Approved: {e.slug}")


def _review_reject(queue: ReviewQueue, slug: str | None, reason: str | None) -> None:  # ty: ignore
    if not slug:
        typer.echo("Reject requires a slug.", err=True)
        raise typer.Exit(1)
    rejected = queue.reject(slug, reason=reason or "")  # ty: ignore
    if not rejected:
        typer.echo(f"No pending page with slug {slug!r}.", err=True)
        raise typer.Exit(1)
    for e in rejected:
        typer.echo(f"Rejected: {e.slug}")


def _dispatch_review_action(
    queue: ReviewQueue,  # ty: ignore
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
        typer.echo(f"Unknown review action: {action!r}. Use list|approve|reject|lock|unlock.", err=True)
        raise typer.Exit(1)
    handler()
