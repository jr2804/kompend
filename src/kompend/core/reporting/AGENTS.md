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
| `handle.py` | `RunHandle` - the run identity the CLI holds between start and end; the `_CURRENT_RUN_ID` ContextVar + `set_current_run_id`/`current_run_id` |
| `runlog.py` | `log_run_start`, `log_run_end` - append-only audit trail in the `runs` table; attaches/detaches the flight recorder (see Rules) |
| `ledger.py` | `StageLedger`, `stage_ledger` - per-item stage ledger in the `stages` table; documented queries (`slowest_items`, `failure_histogram`, `cache_hit_rate`) |

## Rules

- Import: `from kompend.core.reporting.log import build_log_report`,
  `from kompend.core.reporting.runlog import log_run_start`.
- These are CLI-facing presentation modules, not pipeline core logic.
- `runlog.py` inserts `start`/`end` rows into the `runs` table of `.kompend/kompend.db`
  (via `core.store.open_store`) and prunes to a soft cap. `log.py` reads `runs`
  and `lint_runs`. No JSONL — the store is the single opener.
- `ledger.py` writes one row per (run, source, stage, attempt) into the `stages`
  table: outcome (`StageOutcome`), `duration_ms`, `reason_code` (`ReasonCode`),
  a capped detail excerpt, and the attempt's facts (parser/backend/judge
  model/cache key/sha/bytes). Rows correlate with `runs` through the
  `current_run_id()` ContextVar (set by `log_run_start`), so no run id is
  threaded through the pipeline. It is an observer: every write failure is
  swallowed (`sqlite3.Error` → debug log); each row commits on its own so a
  crash keeps what was written; pruning reuses the runlog soft-cap pattern
  (`_SOFT_CAP_ROWS`/`_KEEP_ROWS`). Callers pass `StageLedger` down or open it
  with `stage_ledger(db_path)`; `record()` never raises.
- The run log is written for **every** command, so it must never fail because
  of a config value: the display limits (`cli_arg_preview`, `log_error_chars`)
  are read from the *merged* config (`cfg.get(...)`), not from the validated
  model — `cfg.kompend` raises on a vault whose config does not validate yet
  (`kompend config path` must still work there).
- The **flight recorder** (`kompend.logging.RunLogHandler`) is attached by
  `log_run_start` (root logger, run id minted there) and detached by
  `log_run_end`: INFO+ straight to `.kompend/logs/run-<id>.log`, DEBUG held
  in a bounded ring dumped into the same file only when `status != "ok"`.
  `RunHandle.log_file` carries the path. The handler lives in
  `kompend.logging` (the module that owns logging configuration); root runs
  at DEBUG and the terminal handler's own level gates the console — never
  the other way round.
