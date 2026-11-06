"""The lint passes: the deterministic ``check``, the LLM ``semantic`` pass and ``all``."""

from __future__ import annotations

import json
import time

from kompend.cli.args import (
    JsonOutput,
    LintFix,
    LintFixDestructive,
    LintQuiet,
    LintRule,
    LintVerbose,
)
from kompend.cli.cmds.quality.lint.reporting import print_report
from kompend.cli.settings import (
    build_cli_overrides,
    config_file_from_ctx,
    resolve_lint_output,
    vault_paths_from_ctx,
)
from kompend.core.config import resolve_config
from kompend.core.config._configs import LintConfig
from kompend.core.config.rumdl import resolve_rumdl_config
from kompend.core.models import load_cached_wiki
from kompend.lint.flow import (
    CHECK_STEP_RULES,
    _make_lint_agent,
    persist_lint_fixes,
    run_full_lint_flow,
    run_runner_with_progress,
)
from kompend.lint.flow import (
    _run_pass as _run_rules_pass,
)
from kompend.lint.history import LintLlmMeta, persist_history
from kompend.lint.llm_rules import LLM_RULE_BY_NAME
from kompend.lint.runner import LintRunner
from kompend.logging import get_console, get_err_console
from kompend.output.pages.log import log_stage

__all__ = ["lint_all", "lint_check", "lint_semantic"]


def lint_check(
    *,
    fix: LintFix = True,
    fix_destructive: LintFixDestructive = False,
    rule: LintRule = None,
    json_output: JsonOutput = False,
    verbose_lint: LintVerbose = False,
    quiet_lint: LintQuiet = False,
) -> None:
    """Run the deterministic structural rules against the compiled wiki.

    Covers: broken-wikilinks, image-embed, kompend-reference, missing-frontmatter,
    orphan-page, duplicate-title, empty-page, heading-skip, stale-page,
    source-not-found, orphan-attachments, duplicate-references.
    Formatting (rumdl) is the separate `kompend lint format` step.

    With --fix (default), repairable findings are fixed in place: alias
    links are rewritten to resolved markdown links, image embeds corrected,
    missing frontmatter filled. Destructive fixes (file deletion, e.g.
    orphan-attachments) require an explicit --fix-destructive — plain
    --fix never deletes files.

    Examples:
        kompend lint check                  # structural rules, fix what's possible
        kompend lint check --no-fix         # report only
        kompend lint check --fix-destructive  # also delete orphan attachments
        kompend lint check --rule broken-wikilinks
    """
    vp = vault_paths_from_ctx()
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file_from_ctx())
    wiki = load_cached_wiki(vp.base_dir)
    if wiki is None:
        get_err_console().print(f"No compiled wiki found in {vp.output_dir}. Run 'kompend compile' first.")
        raise SystemExit(1)

    start = time.monotonic()
    runner = LintRunner(
        rules=rule or CHECK_STEP_RULES,
        enable_llm=False,
        lint_agent=None,
        fix=fix,
        fix_destructive=fix_destructive,
        sources_dir=str(vp.sources_dir) if vp.sources_dir.exists() else None,
        lint_config=LintConfig(
            fuzzy_threshold=float(cfg.kompend.lint.fuzzy_threshold),
            llm_timeout=float(cfg.kompend.lint.llm_timeout),
        ),
        rumdl_config=resolve_rumdl_config(vp.base_dir, cfg.kompend.compile.output.rumdl_config),
        data_format=cfg.kompend.compile.llm.data_format,
    )
    report = run_runner_with_progress(runner, wiki, show_progress=None)
    elapsed = time.monotonic() - start
    if fix and any(r.fix_applied for r in report.results):
        updated = persist_lint_fixes(wiki, vp)
        get_console().print(f"Persisted fixes to {updated} file(s).")

    persist_history(
        report,
        vp=vp,
        config=cfg.raw,
        context="check",
        elapsed=elapsed,
        fix=fix,
        node_count=len(wiki.nodes),
        llm=None,
    )
    print_report(report, json_output, vp, resolve_lint_output(cfg.kompend.lint.output, verbose=verbose_lint, quiet=quiet_lint))


def lint_semantic(
    *,
    json_output: JsonOutput = False,
    verbose_lint: LintVerbose = False,
    quiet_lint: LintQuiet = False,
) -> None:
    """Run the LLM-powered lint rules against the compiled wiki.

    Requires an LLM: compile.lint.model, falling back to
    compile.llm.default_model. Read-only — findings feed `kompend lint gaps`
    and `kompend lint remediate`.

    Examples:
        kompend lint semantic               # LLM rules only
    """
    vp = vault_paths_from_ctx()
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file_from_ctx())
    wiki = load_cached_wiki(vp.base_dir)
    if wiki is None:
        get_err_console().print(f"No compiled wiki found in {vp.output_dir}. Run 'kompend compile' first.")
        raise SystemExit(1)

    lint_agent, model = _make_lint_agent(cfg)
    if lint_agent is None:
        get_err_console().print(
            "LLM lint rules require a model. Set compile.lint.model or compile.llm.default_model in config.",
        )
        raise SystemExit(1)

    report, elapsed = _run_rules_pass(wiki, vp, cfg, rules=list(LLM_RULE_BY_NAME.keys()), enable_llm=True, lint_agent=lint_agent, fix=False)
    llm_meta = LintLlmMeta(model=model, usage=lint_agent.usage.to_dict()) if model is not None and lint_agent.usage is not None else None
    persist_history(
        report,
        vp=vp,
        config=cfg.raw,
        context="semantic",
        elapsed=elapsed,
        fix=False,
        node_count=len(wiki.nodes),
        llm=llm_meta,
    )
    print_report(report, json_output, vp, resolve_lint_output(cfg.kompend.lint.output, verbose=verbose_lint, quiet=quiet_lint))


def lint_all(
    *,
    fix: LintFix = True,
    fix_destructive: LintFixDestructive = False,
    json_output: JsonOutput = False,
    verbose_lint: LintVerbose = False,
    quiet_lint: LintQuiet = False,
) -> None:
    """Run every lint step in one flow with a single weighted progress bar.

    Steps (deterministic steps weight 1/item, LLM steps weight 100/item):

    | Step      | What                                    | LLM |
    |-----------|-----------------------------------------|-----|
    | check     | structural rules, auto-fix              | no  |
    | format    | rumdl formatting                        | no  |
    | semantic  | factual/clarity/coverage/contradiction  | yes |
    | gaps      | gap analysis + semantic source check    | no  |
    | remediate | generate missing pages from extractions | yes |

    Destructive fixes (file deletion, e.g. orphan-attachments) require an
    explicit --fix-destructive — plain --fix never deletes files.

    LLM steps are skipped gracefully (with a note) when no model is
    configured (compile.lint.model / compile.llm.default_model).

    Examples:
        kompend lint all                    # everything, auto-fix on
        kompend lint all --no-fix           # report only
        kompend lint all --json
    """
    vp = vault_paths_from_ctx()
    cfg = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides(), config_file=config_file_from_ctx())
    lint_output = resolve_lint_output(cfg.kompend.lint.output, verbose=verbose_lint, quiet=quiet_lint)
    merged_dict = run_full_lint_flow(vp, cfg, fix=fix, fix_destructive=fix_destructive, lint_output=lint_output)
    if json_output:
        get_console().print(json.dumps(merged_dict, indent=2, ensure_ascii=False))
    else:
        get_console().print(
            f"Lint complete: health {merged_dict.get('health_score', 0.0):.1f}% "
            f"({merged_dict.get('passes', 0)} pass / {merged_dict.get('failures', 0)} fail / "
            f"{merged_dict.get('skips', 0)} skip)"
        )
        log_stage(
            vp,
            action="Lint",
            detail=f"Health score {merged_dict.get('health_score', 0.0):.1f}% ({merged_dict.get('passes', 0)} checks passed)",
        )
