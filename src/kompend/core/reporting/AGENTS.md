# DOX — kompend.core.reporting

<!-- on entry: loaded when agent enters src/kompend/core/reporting -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

On-demand report builders for CLI observability. Aggregates structured
stores into human-readable markdown — no LLM calls.

## Modules

| Module | Contents |
|--------|----------|
| `log.py` | `build_log_report` - the engine behind `kompend log` |
| `handle.py` | `RunHandle` - the run identity the CLI holds between start and end |
| `runlog.py` | `log_run_start`, `log_run_end` - append-only audit trail in the `runs` table |

## Rules

- Import: `from kompend.core.reporting.log import build_log_report`,
  `from kompend.core.reporting.runlog import log_run_start`.
- These are CLI-facing presentation modules, not pipeline core logic.
- `runlog.py` inserts `start`/`end` rows into the `runs` table of `.kompend/kompend.db`
  (via `core.store.open_store`) and prunes to a soft cap. `log.py` reads `runs`
  and `lint_runs`. No JSONL — the store is the single opener.
- The run log is written for **every** command, so it must never fail because
  of a config value: the display limits (`cli_arg_preview`, `log_error_chars`)
  are read from the *merged* config (`cfg.get(...)`), not from the validated
  model — `cfg.kompend` raises on a vault whose config does not validate yet
  (`kompend config path` must still work there).
