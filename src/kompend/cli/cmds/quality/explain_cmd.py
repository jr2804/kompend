"""``kompend explain`` - what the pipeline did to one source file."""

from __future__ import annotations

import json
import sys

from kompend.cli.args import JsonOutput
from kompend.cli.args.quality import ExplainRun, ExplainSource
from kompend.cli.settings import build_cli_overrides, vault_paths_from_ctx
from kompend.core.config import resolve_config
from kompend.core.reporting.explain import AmbiguousSourceError, build_explain_report, render_explain
from kompend.logging import get_console, get_err_console


def explain_cmd(
    source: ExplainSource,
    *,
    run_id: ExplainRun = None,
    json_output: JsonOutput = False,
) -> None:
    """Explain what the pipeline did to one source file.

    Stage-by-stage ledger rows of the chosen run, where the wall-clock time
    went, the attempt's facts (parser, backend, judge model, cache identity)
    and every degradation with its reason code - no ``--verbose`` needed.

    Examples:
        kompend explain sources/notes/stuck.md
        kompend explain stuck.md --run r-20261107-3 --json
    """
    vp = vault_paths_from_ctx()
    resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())
    try:
        report = build_explain_report(vp.base_dir, source, run_id=run_id)
    except AmbiguousSourceError as exc:
        get_err_console().print(str(exc))
        get_err_console().print("candidates: " + ", ".join(exc.candidates))
        raise SystemExit(1) from None
    if report is None:
        get_err_console().print(f"No ledger rows found for {source!r}. Nothing ingested, or the name does not match.")
        raise SystemExit(1)
    if json_output:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    get_console().print(render_explain(report), markup=False)
