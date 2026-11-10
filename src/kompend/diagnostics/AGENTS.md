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
  JSON), command wiring (`doctor_cmd.py`), MCP exposure (`mcp/tools/quality/doctor.py`)

## Architecture

| Module | Contents |
|--------|----------|
| `base.py` | `CheckResult` — the frozen result record every check returns |
| `environment.py` | Sources/output directories, git, disk headroom, the parser build — the parts that need no model |
| `providers.py` | Model catalogue, `[providers.<name>]` endpoints, Ollama offload + tag identity (`ollama_show_lookup`), `nvidia-smi` |
| `models.py` | Connectivity pings, structured-output probes, context window discovery, window fit, server window, output ceiling |
| `decision.py` | The two decision-model checks: the offline result-type grammar (`decision-schema`) and the live SystemOne ping (`connectivity-decision`) |
| `capabilities.py` | The structured-output probes themselves (`probe_*`, `measure_compact_formats`) |
| `runner.py` | `run_doctor` — which checks run, in which order |

## Local Contracts

- **A check returns a result; it never raises.** A failure is a fact the doctor
  prints (`CheckResult(passed=False, message=…)`), so an unreachable provider,
  a missing wheel or an empty catalogue is reported, not thrown.
- **Order is part of the contract** (`runner.py` docstring): the catalogue
  first, then the free workspace checks, then the decision checks, then
  per-model blocks so a model is never swapped out between two of its own
  checks, then the embedder last — loading it evicts the completion models
  from VRAM. The decision checks sit before the llm block because a decision
  model speaks a different endpoint family (SystemOne): running it first keeps
  a locally hosted decision model from evicting the completion model those
  measurements just loaded.
- **`CheckResult.name` is kebab-case and stable** — the JSON output and the MCP
  tool expose it, so renaming one is a user-visible change. The decision checks
  are `decision-schema` and `connectivity-decision`.
- **A decision model is optional, its verdict is not.** `connectivity-decision`
  is a WARNING when `[decision]` is disabled or has no model — the pipeline
  keeps its deterministic fallbacks — and a FAILURE when a configured model
  cannot answer. The offline schema check never touches the network: it builds
  the request the grammar would build for every type in
  `llm.decision.DECISION_RESULT_TYPES` and for one representative of the dynamic
  split-boundary family (`boundary_verdict_model`, validated at the six-page
  `SourceSplitter.min_pages` floor), judged against the caps the shipped
  `TypeSafeModel` declares rather than against no limits at all — a cap copied
  into this package would be the first thing to go stale. If the pydantic-ai
  internals it reads move, it reports that as a warning instead of passing
  silently, and an unimportable splitting package is a warning too.
- **One side effect, one place.** `providers.check_catalogue()` calls
  `catalog.ensure_fresh()` (TTL-gated) and is the only refresher of the
  models.dev cache in the process; no other check writes anything, and none
  writes into the vault.
- Model-string parsing (`provider_id_of`/`model_slug`) is not owned here — it lives
  in `../llm/model_id.py` (see `../llm/AGENTS.md`). The doctor imports it, so its
  provider lookups and Ollama calls cannot disagree with the build path.
- **The embed check reads the manifest, it never writes.** `doctor_cmd.py` hands
  `run_doctor` the embedding string recorded in `kompend.db`
  (`rag.recorded_embedding_model`), and `ping_embed_model` resolves both tags to
  digests to flag an alias-tag rebuild that would produce byte-identical
  vectors. A digest it cannot resolve degrades to plain tag reporting — never
  a guess, never a write.
- **The last-run section is not a check.** `core.reporting.last_run` builds it
  from the ledger and the run log and `print_doctor_results` renders it; the
  checks stay host/model probes and never read the ledger. The section is
  read-only and prints an explicit empty state for a vault with no runs.

## Verification

- `uv run pytest tests/unit/test_cli_doctor_status.py tests/unit/test_doctor_decision.py tests/unit/test_capability_probes.py -q`
- `uv run python scripts/checks/check_field_consistency.py` (a renamed result
  field breaks the JSON contract silently otherwise)
