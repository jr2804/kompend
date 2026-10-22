"""Reference management MCP tools."""

import json
from pathlib import Path
from typing import Any

from kompend.agents.reference_resolver import ReferenceResolverAgent
from kompend.cli._shared import build_cli_overrides
from kompend.compilation.flows.references import resolve_unresolved_references
from kompend.core.config import resolve_config
from kompend.core.models.references import SHORT_LABEL_TITLE_MAX
from kompend.core.paths import vault_paths
from kompend.core.pipeline.references import ReferenceStore
from kompend.logging import get_logger
from kompend.output.bibtex import references_to_bibtex

logger = get_logger(__name__)


def references_list(
    wiki_dir: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """List all references (or those from a specific source)."""
    vp = vault_paths(wiki_dir)
    ref_file = vp.references_file
    if not ref_file.exists():
        return {"status": "ok", "references": [], "count": 0, "message": "No references found. Run a compile first."}

    store = ReferenceStore(ref_file)

    if source:
        refs = store.get_source(source)
        if not refs:
            return {"status": "ok", "references": [], "count": 0, "message": f"No references found for source: {source}"}
    else:
        refs = store.all_references()
        if not refs:
            return {"status": "ok", "references": [], "count": 0, "message": "No references found."}

    ref_dicts = []
    for i, ref in enumerate(refs, 1):
        label = ref.short_label() or ref.id or "(untitled)"
        title = ref.title or ref.raw[:SHORT_LABEL_TITLE_MAX] if ref.raw else ""
        doi_or_url = ref.doi or ref.url or ""
        ref_dicts.append(
            {
                "index": i,
                "label": label,
                "title": title,
                "doi_or_url": doi_or_url,
            }
        )

    return {"status": "ok", "references": ref_dicts, "count": len(ref_dicts)}


def references_export(
    wiki_dir: str | None = None,
    output: str = "references.bib",
    source: str | None = None,
) -> dict[str, Any]:
    """Export references to .bib (BibTeX) or .json (CSL JSON)."""
    vp = vault_paths(wiki_dir)
    ref_file = vp.references_file
    if not ref_file.exists():
        return {"status": "error", "message": "No references found. Run a compile first."}

    store = ReferenceStore(ref_file)

    if source:
        refs = store.get_source(source)
        if not refs:
            return {"status": "error", "message": f"No references found for source: {source}"}
    else:
        refs = store.all_references()

    if not refs:
        return {"status": "error", "message": "No references to export."}

    out_file = Path(output)
    suffix = out_file.suffix.lower()
    if suffix == ".bib":
        content = references_to_bibtex(refs)
    elif suffix == ".json":
        content = json.dumps([r.to_csl() for r in refs], indent=2, ensure_ascii=False)
    else:
        return {"status": "error", "message": f"Unknown format '{suffix}'. Use .bib (BibTeX) or .json (CSL JSON)."}

    out_file.write_text(content, encoding="utf-8")
    return {"status": "ok", "count": len(refs), "message": f"Exported {len(refs)} reference(s) to {output}"}


def references_fetch(
    wiki_dir: str | None = None,
    source: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Resolve missing DOIs/URLs for references, using the configured LLM."""
    vp = vault_paths(wiki_dir)
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())
    ref_file = vp.references_file
    if not ref_file.exists():
        return {"status": "ok", "count": 0, "message": "No references found. Run a compile first."}

    store = ReferenceStore(ref_file)
    pending = store.unresolved_indices(source)
    if not pending:
        return {"status": "ok", "count": 0, "message": "All references already have a DOI or URL."}

    model = cfg.kompend.compile.llm.default_model
    if not model:
        return {"status": "error", "message": "No LLM model configured (KOMP_LLM_DEFAULT_MODEL)."}

    timeout = float(cfg.kompend.compile.llm.timeout)
    try:
        agent = ReferenceResolverAgent(
            model_string=model,
            timeout=timeout,
            llm=cfg.kompend.compile.llm,
            budget=cfg.kompend.compile.budget,
        )
    except Exception as exc:
        return {"status": "error", "message": f"Could not initialise resolver agent: {exc}"}

    resolved = resolve_unresolved_references(
        store,
        agent,
        timeout,
        source=source,
        dry_run=dry_run,
    )

    if dry_run:
        return {"status": "ok", "count": resolved, "dry_run": True, "message": f"[dry-run] {resolved} reference(s) would be resolved."}
    if resolved:
        remaining = len(store.unresolved_indices(source))
        return {"status": "ok", "count": resolved, "remaining": remaining, "message": f"Resolved {resolved} reference(s). {remaining} still unresolved."}
    return {"status": "ok", "count": 0, "message": "No references could be resolved."}
