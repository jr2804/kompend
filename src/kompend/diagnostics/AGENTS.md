# DOX — kompend.diagnostics

<!-- on entry: loaded when agent enters src/kompend/diagnostics -->
<!-- parent: ../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/CHECKS.md -->

## Purpose

Host and configuration diagnostics behind `kompend doctor` and the
`doctor` MCP tool. Each check answers one question about the machine or the
configuration — it never runs the pipeline, never writes to the vault, and
never raises.

## Ownership

- `src/kompend/diagnostics/` — the checks, the result type, the run order
- Not here: rendering (`src/kompend/cli/cmds/quality/doctor.py` prints text and
  JSON), command wiring (`doctor_cmd.py`), MCP exposure (`mcp/tools/quality.py`)

## Architecture

| Module | Contents |
|--------|----------|
| `base.py` | `CheckResult` — the frozen result record every check returns |
| `environment.py` | Sources/output directories, git, disk headroom, the parser build — the parts that need no model |
| `providers.py` | Model catalogue, `[providers.<name>]` endpoints, Ollama offload, `nvidia-smi` |
| `models.py` | Connectivity pings, structured-output probes, context window discovery, window fit, server window, output ceiling |
| `capabilities.py` | The structured-output probes themselves (`probe_*`, `measure_compact_formats`) |
| `runner.py` | `run_doctor` — which checks run, in which order |

## Local Contracts

- **A check returns a result; it never raises.** A failure is a fact the doctor
  prints (`CheckResult(passed=False, message=…)`), so an unreachable provider,
  a missing wheel or an empty catalogue is reported, not thrown.
- **Order is part of the contract** (`runner.py` docstring): the catalogue
  first, then the free workspace checks, then per-model blocks so a model is
  never swapped out between two of its own checks, then the embedder last —
  loading it evicts the completion models from VRAM.
- **`CheckResult.name` is kebab-case and stable** — the JSON output and the MCP
  tool expose it, so renaming one is a user-visible change.
- **One side effect, one place.** `providers.check_catalogue()` calls
  `catalog.ensure_fresh()` (TTL-gated) and is the only refresher of the
  models.dev cache in the process; no other check writes anything, and none
  writes into the vault.
- Model-string parsing (`provider_id_of`/`model_slug`) is not owned here — it lives
  in `../llm/model_id.py` (see `../llm/AGENTS.md`). The doctor imports it, so its
  provider lookups and Ollama calls cannot disagree with the build path.

## Verification

- `uv run pytest tests/unit/test_cli_doctor_status.py tests/unit/test_capability_probes.py -q`
- `uv run python scripts/checks/check_field_consistency.py` (a renamed result
  field breaks the JSON contract silently otherwise)
