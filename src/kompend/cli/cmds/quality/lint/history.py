"""The lint history log: querying past runs, reading one report, trimming the log."""

from __future__ import annotations

from kompend.cli.args import (
    JsonOutput,
    LintHistoryLast,
    LintHistoryTrend,
    LintPruneAgeDays,
    LintPruneKeep,
    LintShowRun,
)
from kompend.cli.settings import build_cli_overrides, config_file_from_ctx, vault_paths_from_ctx
from kompend.core.config import resolve_config
from kompend.core.paths import LINT_HISTORY_FILE
from kompend.lint.history import history_to_json, history_to_text, load_lint_history, prune_lint_history
from kompend.lint.log import latest_lint_log, lint_log_dir, read_lint_log
from kompend.logging import get_console, get_err_console

__all__ = ["lint_history", "lint_prune", "lint_show"]


def lint_history(
    *,
    last: LintHistoryLast = 10,
    trend: LintHistoryTrend = False,
    json_output: JsonOutput = False,
) -> None:
    """Query the lint history log.

    Examples:
        kompend lint history              # last 10 runs
        kompend lint history -n 5         # last 5 runs
        kompend lint history --trend      # with score sparkline
        kompend lint history --json       # machine-readable
    """
    vp = vault_paths_from_ctx()
    resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(),
        config_file=config_file_from_ctx(),
    )
    state_dir = vp.kompend_dir

    entries = load_lint_history(state_dir / LINT_HISTORY_FILE, last_n=last)

    if json_output:
        get_console().print(history_to_json(entries))
    else:
        get_console().print(history_to_text(entries, show_trend=trend))


def lint_show(
    *,
    run: LintShowRun = None,
    json_output: JsonOutput = False,
) -> None:
    """Read a per-run lint report from ``.kompend/lint-log/``.

    The terminal summary is bounded, so the full finding list lives here: the
    Markdown report for humans and the JSON entry (same payload as the
    ``lint_runs`` row) for tools.
    """
    vp = vault_paths_from_ctx()
    directory = lint_log_dir(vp.base_dir)
    if not directory.is_dir():
        get_err_console().print("No lint report found - run 'kompend lint all' or 'kompend compile' first.")
        raise SystemExit(1)
    if run:
        path = directory / f"{run}.{'json' if json_output else 'md'}"
        if not path.exists():
            get_err_console().print(f"No lint report named '{run}' in {directory}.")
            raise SystemExit(1)
    else:
        markdown = latest_lint_log(vp.base_dir)
        if markdown is None:
            get_err_console().print(f"No lint report found in {directory}.")
            raise SystemExit(1)
        path = markdown.with_suffix(".json") if json_output else markdown
    get_console().print(read_lint_log(path))


def lint_prune(
    *,
    keep_last: LintPruneKeep = None,
    max_age_days: LintPruneAgeDays = None,
) -> None:
    """Remove old entries from ``.kompend/lint_history.jsonl``.

    At least one of --keep-last or --max-age-days must be provided.

    Examples:
        kompend lint prune --keep-last 50        # keep only 50 newest
        kompend lint prune --max-age-days 30     # remove entries older than 30 days
    """
    if keep_last is None and max_age_days is None:
        get_err_console().print("At least one of --keep-last or --max-age-days must be provided.")
        raise SystemExit(1)

    vp = vault_paths_from_ctx()
    resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(),
        config_file=config_file_from_ctx(),
    )
    state_dir = vp.kompend_dir

    removed = prune_lint_history(
        state_dir / LINT_HISTORY_FILE,
        keep_last=keep_last,
        max_age_days=max_age_days,
    )
    get_console().print(f"Removed {removed} entr{'y' if removed == 1 else 'ies'}.")
