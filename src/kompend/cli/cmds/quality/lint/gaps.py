"""Gap analysis: turn the latest lint run into knowledge gaps and source recommendations."""

from __future__ import annotations

import json
import re
from typing import Any

from kompend.agents.remediate import make_agentic_remediation_agent
from kompend.cli.args import JsonOutput, LintGapsFix, LintSemantic
from kompend.cli.settings import build_cli_overrides, config_file_from_ctx, vault_paths_from_ctx
from kompend.core.cache.wiki import write_wiki_cache
from kompend.core.config import ResolvedConfig, resolve_config
from kompend.core.models import load_cached_wiki
from kompend.core.paths import LINT_HISTORY_FILE, VaultPaths
from kompend.lint.base import LintReport, RemediationStatus
from kompend.lint.gap_analysis import analyze_gaps, gaps_to_text, load_degree_map
from kompend.lint.history import load_lint_history, report_from_dict
from kompend.lint.remediation import remediate, remediation_to_text
from kompend.lint.semantic_gaps import verify_topics
from kompend.logging import get_console, get_err_console

__all__ = ["lint_gaps"]


def lint_gaps(
    *,
    json_output: JsonOutput = False,
    semantic: LintSemantic = True,
    fix: LintGapsFix = False,
) -> None:
    """Analyze the latest lint run to identify missing knowledge and recommend sources.

    Reads the most recent entry from ``.kompend/lint_history.jsonl`` and extracts
    actionable gap signals: missing topics (broken wikilinks), coverage gaps,
    poor-quality pages, empty pages, orphan topics, and contradictions.

    Missing-topic findings are coverage-weighted: referenced by many pages or
    by a hub page → HIGH (knowledge-core hole); a single leaf-page reference
    → LOW (fringe term).

    With the embedding model configured, missing topics are additionally
    verified against the source chunks: a topic the sources already discuss
    is reclassified from "add a source" to "covered — generate the page"
    (run ``kompend lint remediate`` or pass ``--fix``).

    Examples:
        kompend lint gaps              # analyze latest lint run
        kompend lint gaps --json       # machine-readable output
        kompend lint gaps --fix        # generate pages for source-covered topics
    """
    vp = vault_paths_from_ctx()
    cfg = resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(),
        config_file=config_file_from_ctx(),
    )
    state_dir = vp.kompend_dir

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=1)
    if not entries:
        get_err_console().print("No lint history found. Run `kompend lint run` or compile with --lint first.")
        raise SystemExit(1)

    report_payload = entries[-1].report
    report = report_from_dict(report_payload)

    # Coverage weighting: node degree from the cached wiki graph.
    degree_map = load_degree_map(vp.base_dir)
    recs = analyze_gaps(report, degree_map, cfg.kompend.display)

    # Semantic source-coverage verification for missing topics.
    embed_model = cfg.kompend.compile.embedding.model
    covered_topics: dict[str, Any] = {}
    if semantic and embed_model:
        missing = [r for r in recs if r.resolution == "missing" and r.category == "Missing topic"]
        topics: list[str] = []
        for r in missing:
            m = re.match(r"'(.+?)'", r.finding)
            topics.append(m.group(1) if m else r.finding)
        coverage = verify_topics(topics, vp.base_dir, embed_model)
        for rec in missing:
            m = re.match(r"'(.+?)'", rec.finding)
            topic = m.group(1) if m else rec.finding
            cov = coverage.get(topic)
            if cov is not None and cov.covered:
                covered_topics[topic] = {"score": cov.best_score, "source": cov.best_source}
                rec.category = "Covered in sources (no page)"
                rec.resolution = "source-covered"
                rec.action = (
                    f"Sources already cover this topic (best match: {cov.best_source}, "
                    f"score {cov.best_score:.2f}) — run `kompend lint remediate` to generate the page."
                )
    elif semantic and not embed_model:
        get_err_console().print("Note: semantic coverage check skipped — no compile.embedding.model configured.\n")

    if fix:
        _run_remediation_for_gaps(vp, cfg, report)

    if json_output:
        payload: list[dict[str, Any]] = [
            {"priority": r.priority, "category": r.category, "finding": r.finding, "action": r.action, "resolution": r.resolution} for r in recs
        ]
        if covered_topics:
            payload.append({"category": "_semantic_coverage", "topics": covered_topics})
        get_console().print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        ts = entries[-1].timestamp
        score = report_payload.health_score
        get_console().print(f"\nBased on lint run from {ts} (score: {score:.0f}%)\n")
        get_console().print(gaps_to_text(recs))


def _run_remediation_for_gaps(vp: VaultPaths, cfg: ResolvedConfig, report: LintReport) -> None:
    """Run the remediation flow (same as `kompend lint remediate`) after analysis."""
    wiki = load_cached_wiki(vp.base_dir)
    if wiki is None:
        get_err_console().print("--fix: no compiled wiki found — run 'kompend compile' first.")
        return
    default_model = cfg.kompend.compile.llm.default_model
    if not default_model:
        get_err_console().print("--fix: no compile.llm.default_model configured — cannot generate pages.")
        return
    agent = make_agentic_remediation_agent(
        model_string=default_model,
        timeout=float(cfg.kompend.compile.llm.timeout),
        llm=cfg.kompend.compile.llm,
        memory_dir=vp.agent_memory_dir,
    )
    actions = remediate(
        wiki,
        report,
        base_dir=vp.base_dir,
        output_dir=vp.output_dir,
        agent=agent,
        dry_run=False,
    )
    applied = [a for a in actions if a.status == RemediationStatus.APPLIED]
    if applied:
        write_wiki_cache(wiki, vp.base_dir)
    get_console().print(remediation_to_text(actions))
