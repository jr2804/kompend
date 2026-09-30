# DOX — knox.core.reporting

<!-- on entry: loaded when agent enters src/knox/core/reporting -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

On-demand report builders for CLI observability. Aggregates structured
stores into human-readable markdown — no LLM calls.

## Modules

| Module | Contents |
|--------|----------|
| `log.py` | `build_log_report` — the engine behind `knox log` |
| `runlog.py` | `RunHandle`, `log_run_start`, `log_run_end` — append-only audit trail in the `runs` table |

## Rules

- Import: `from knox.core.reporting.log import build_log_report`,
  `from knox.core.reporting.runlog import log_run_start`.
- These are CLI-facing presentation modules, not pipeline core logic.
- `runlog.py` inserts `start`/`end` rows into the `runs` table of `.knox/knox.db`
  (via `core.store.open_store`) and prunes to a soft cap. `log.py` reads `runs`
  and `lint_runs`. No JSONL — the store is the single opener.
