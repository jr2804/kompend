# DOX — knox.core.reporting

## Purpose

On-demand report builders for CLI observability. Aggregates structured
stores into human-readable markdown — no LLM calls.

## Modules

| Module | Contents |
|--------|----------|
| `log.py` | `build_log_report` — the engine behind `knox log` |
| `runlog.py` | `RunHandle`, `log_run_start`, `log_run_end` — append-only JSONL audit trail |

## Rules

- Import: `from knox.core.reporting.log import build_log_report`,
  `from knox.core.reporting.runlog import log_run_start`.
- These are CLI-facing presentation modules, not pipeline core logic.
- `runlog.py` writes to `.knox/runs.jsonl` and prunes to a soft cap.
