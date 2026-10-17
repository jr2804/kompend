"""``kompend references`` — manage bibliographic references.

Subcommands:
- ``list``: Display references stored in the project.
- ``export``: Export references to BibTeX or CSL JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kompend.agents.reference_resolver import ReferenceResolverAgent
from kompend.cli._shared import build_cli_overrides, vault_paths_from_ctx
from kompend.cli.app import app
from kompend.cli.args import (
    ReferencesExportOutput,
    ReferencesExportSource,
    ReferencesFetchDryRun,
    ReferencesFetchSource,
    ReferencesListSource,
)
from kompend.compilation.flows.references import resolve_unresolved_references
from kompend.core.config import resolve_config
from kompend.core.models.references import SHORT_LABEL_TITLE_MAX
from kompend.core.pipeline.references import ReferenceStore
from kompend.logging import get_console, get_err_console, get_logger
from kompend.output.bibtex import references_to_bibtex

logger = get_logger(__name__)

references_app = typer.Typer(
    name="references",
    help="Manage extracted bibliographic references.",
    no_args_is_help=True,
)
app.add_typer(references_app)


@references_app.command("list", help="List all extracted references.")
def references_list(
    ctx: typer.Context,
    source: ReferencesListSource = None,
) -> None:
    """List all references (or those from a specific source)."""
    store = _resolve_ref_store(ctx)
    if store is None:
        get_console().print("No references found. Run a compile first to extract references.")
        raise typer.Exit(0)

    if source:
        refs = store.get_source(source)
        if not refs:
            get_console().print(f"No references found for source: {source}")
            raise typer.Exit(0)
        get_console().print(f"[bold]References from {source}[/bold] ({len(refs)}):\n")
    else:
        refs = store.all_references()
        if not refs:
            get_console().print("No references found.")
            raise typer.Exit(0)
        get_console().print(f"[bold]All references[/bold] ({len(refs)}):\n")

    for i, ref in enumerate(refs, 1):
        label = ref.short_label() or ref.id or "(untitled)"
        title = ref.title or ref.raw[:SHORT_LABEL_TITLE_MAX] if ref.raw else ""
        doi_or_url = ref.doi or ref.url or ""
        get_console().print(f"  [cyan]{i:3d}.[/cyan] {label}")
        if title:
            get_console().print(f"        {title}", markup=False)
        if doi_or_url:
            get_console().print(f"        {doi_or_url}", markup=False)


@references_app.command("export", help="Export references to BibTeX (.bib) or CSL JSON (.json).")
def references_export(
    ctx: typer.Context,
    output: ReferencesExportOutput = Path("references.bib"),
    source: ReferencesExportSource = None,
) -> None:
    """Export references to ``.bib`` (BibTeX) or ``.json`` (CSL JSON).

    The output format is determined by the file extension:
    ``.bib`` → BibTeX, ``.json`` → CSL JSON.
    """
    store = _resolve_ref_store(ctx)
    if store is None:
        get_err_console().print("No references found. Run a compile first to extract references.")
        raise typer.Exit(1)

    if source:
        refs = store.get_source(source)
        if not refs:
            get_err_console().print(f"No references found for source: {source}")
            raise typer.Exit(1)
    else:
        refs = store.all_references()

    if not refs:
        get_err_console().print("No references to export.")
        raise typer.Exit(1)

    suffix = output.suffix.lower()
    if suffix == ".bib":
        content = references_to_bibtex(refs)
    elif suffix == ".json":
        content = json.dumps([r.to_csl() for r in refs], indent=2, ensure_ascii=False)
    else:
        get_err_console().print(
            f"Unknown format '{suffix}'. Use .bib (BibTeX) or .json (CSL JSON).",
        )
        raise typer.Exit(1)

    output.write_text(content, encoding="utf-8")
    get_console().print(f"Exported {len(refs)} reference(s) to {output}")


def _resolve_ref_store(ctx: typer.Context) -> ReferenceStore | None:
    """Resolve the project ``ReferenceStore``, or ``None`` if missing."""
    vp = vault_paths_from_ctx(ctx)
    store = ReferenceStore(vp.references_file)
    return store if store.total_count() > 0 else None


@references_app.command(
    "fetch",
    help="Resolve DOIs/URLs for references missing them, using the configured LLM.",
)
def references_fetch(
    ctx: typer.Context,
    source: ReferencesFetchSource = None,
    dry_run: ReferencesFetchDryRun = False,
) -> None:
    """Resolve missing DOIs/URLs for references that lack them.

    For every stored reference that has neither a DOI nor a URL (optionally
    limited to a single *source*), the configured LLM is asked to recover a
    canonical identifier from the citation text. Resolved identifiers are
    written back to the references store so the wiki's reference links
    become resolvable.

    Requires an LLM model to be configured (``KOMP_LLM_DEFAULT_MODEL``).
    """
    vp = vault_paths_from_ctx(ctx)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())
    store = ReferenceStore(vp.references_file)
    if store.total_count() == 0:
        get_console().print("No references found. Run a compile first to extract references.")
        raise typer.Exit(0)

    pending = store.unresolved_indices(source)
    if not pending:
        get_console().print("All references already have a DOI or URL.")
        raise typer.Exit(0)

    model = cfg.kompend.compile.llm.default_model
    if not model:
        get_err_console().print(
            "No LLM model configured. Set KOMP_LLM_DEFAULT_MODEL or run 'kompend config set'.",
        )
        raise typer.Exit(1)

    timeout = float(cfg.kompend.compile.llm.timeout)
    try:
        agent = ReferenceResolverAgent(model_string=model, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        get_err_console().print(f"Could not initialise resolver agent: {exc}")
        raise typer.Exit(1) from exc

    get_console().print(f"Resolving {len(pending)} reference(s) via {model}…")
    resolved = resolve_unresolved_references(
        store,
        agent,
        timeout,
        source=source,
        dry_run=dry_run,
    )

    if dry_run:
        get_console().print(
            f"[dry-run] {resolved} reference(s) would be resolved. No changes saved.",
        )
    elif resolved:
        remaining = len(store.unresolved_indices(source))
        get_console().print(
            f"Resolved {resolved} reference(s). {remaining} still unresolved.",
        )
    else:
        get_console().print("No references could be resolved.")
