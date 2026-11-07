# 2026-11-07 — Pydantic AI durable-execution candidates (knox-rxgu)

**Bead:** knox-rxgu (P3, parent epic knox-cj1i). **Method:** installed wheels only —
pydantic-ai-slim 2.66.0, pydantic-ai-harness 0.66.0, pydantic-graph 2.66.0 under
`.venv/lib/python3.13/site-packages/`. No website copy. No dependencies installed, no
`src/kompend/` changes (research deliverable).

## 1. What the installed stack actually offers

**No `dbos`, `prefect`, or `temporalio` runtime package is installed.** All three are
optional extras of pydantic-ai-slim 2.66.0 (`pydantic_ai_slim-2.66.0.dist-info/METADATA`):

```toml
:62  Provides-Extra: dbos
:63  Requires-Dist: dbos>=2.10.0; extra == 'dbos'
:105 Provides-Extra: prefect
:106 Requires-Dist: prefect>=3.9.0; extra == 'prefect'
:121 Provides-Extra: temporal
:122 Requires-Dist: temporalio>=1.27.0; extra == 'temporal'
```

**`pydantic_ai/capabilities/` exists** in 2.66.0 with ~30 modules (abstract, capability,
citations, combined, content_filter, deferred_tool_handler, **durable_operation**,
file_understanding, hooks, image_generation, instrumentation, local_workspace, mcp,
model_fallback, native_tool, prefix_tools, prepare_tools, process_event_stream,
process_history, reinject_system_prompt, resolve_model_id, select_model,
set_tool_metadata, **thread_executor**, toolset, web_fetch, web_search, wrapper,
x_search, …). Relevant here:

- `capabilities/durable_operation.py` (3 lines) re-exports
  `pydantic_ai.durable_exec._capability_operation.durable_operation`.
- `capabilities/thread_executor.py` — `UseThreadExecutor`: a bounded
  `ThreadPoolExecutor` for sync callbacks ("sync tool functions and other sync
  callbacks … run in threads using `anyio.to_thread.run_sync`"). Thread hygiene, not
  durability or parallel LLM work.

**`pydantic_ai/durable_exec/`** carries the full durability framework
(`BaseDurabilityCapability`, `DurabilityEngineSpec`, `DurableOperation(Backend)`,
`IDENTITY_CODEC`/`JSON_CODEC`, `DurableRunContext`) plus three engine subpackages.
`durable_exec/__init__.py` docstring: *"Each subpackage adds durability for one
durable-execution platform via a capability you attach to an `Agent`"* —
`TemporalDurability`, `DBOSDurability`, `PrefectDurability`.

**DBOS** (`durable_exec/dbos/__init__.py`) exports `DBOSAgent`, `DBOSDurability`,
`DBOSModel`, `DBOSOperationNamer`, `DBOSParallelExecutionMode`, `StepConfig`; raises
`ImportError` when the extra is absent — the wheel's own message directs installing
the `dbos` package via the `pydantic-ai-slim[dbos]` extra group. Core docstring
(`dbos/_durability.py`, `DBOSDurability`):
*"Capability that makes an agent durable by routing I/O through DBOS steps. … routes
model requests, MCP I/O, and optionally event-stream handling through DBOS steps when
the agent runs inside a DBOS workflow. … outside a workflow the capability is
transparent and the run is a normal, non-durable agent run."*

**Prefect** (`durable_exec/prefect/__init__.py`) exports `PrefectAgent` (deprecated),
`PrefectDurability`, `PrefectModel`, `PrefectMCPToolset` (deprecated),
`PrefectFunctionToolset` (deprecated), `TaskConfig`, `DEFAULT_PYDANTIC_AI_CACHE_POLICY`.
`PrefectDurability` (`prefect/_durability.py`): *"makes an agent durable by routing I/O
through Prefect tasks"* inside a `@flow`; its `engine_spec` sets
`unsupported_runtime_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'})` and
`wrapped_toolset_kinds` — **kompend's plain function tools would be wrapped as Prefect
tasks (`toolset_lifecycles={'function': 'enter-always', …}`)**.

**In-process parallel tool calling is already the pydantic-ai default**, no engine
involved: `pydantic_ai/tool_manager.py:51`
`ParallelExecutionMode = Literal['parallel', 'sequential', 'parallel_ordered_events']`,
default `'parallel'` (`pydantic_ai/agent/abstract.py:2077`
`parallel_execution_mode(mode: tool_manager.ParallelExecutionMode = 'parallel')`).
Under durability it *narrows*: `dbos/_agent.py:69-70`
`DBOSParallelExecutionMode = Literal['sequential', 'parallel_ordered_events']` — *"This
is a subset of the ParallelExecutionMode because 'parallel' cannot guarantee
deterministic ordering."*

**Harness `step_persistence`** (`pydantic_ai_harness/step_persistence/README.md`) is
what kompend already uses: snapshots at settled steps with `complete`/`interrupted`
states, an append-only step-event trail, and a tool-effect ledger; stores "from memory,
files, SQLite, MongoDB, or your own store". Its own boundary statement: *"A snapshot
holds the run's message history, not everything around it … For recovery inside a
step, run the agent on durable execution, which `StepPersistence` works alongside."*

## 2. kompend's current state — the bead's premise is stale

The consolidated bead text says *"checkpointing is currently manual"*. That stopped
being true on 2026-10-27 (HISTORY.md row `harness-transcripts`): kompend attaches
harness `StepPersistence` to every agent (`src/kompend/agents/agentic_harness.py:69,120`)
and persists generation transcripts into `SqliteStepStore(paths.step_db_file)`
(`src/kompend/compilation/flows/generation.py:495`). The conversation id is the
generation deps key (extraction set, model, sizes, pinned facts), so it is
content-stable across processes; each pass gets a deterministic `...-segNNN-aNNpNN`
run id (`src/kompend/compilation/AGENTS.md:59-66`).

Recovery today (`src/kompend/compilation/flows/generation.py:675-745`): a finished
segment *"replays its pages without an LLM call"*; an interrupted segment resumes its
own conversation via `recovery.history`; only a segment with no persisted conversation
starts fresh. Extraction results and the assembled wiki are additionally cached at the
compile level (`two_phase.compile()`), and the per-item stage ledger (knox-ty8f.2,
store lane) is landing separately.

Also relevant: `rg "asyncio.gather|TaskGroup" src/kompend` — **zero hits**. Segment and
source processing is sequential today; kompend has no agent-internal parallelism gap to
close, and its generation tools (`read_segment`, `fetch_concepts`, `list_known_titles`)
are sub-millisecond local reads where LLM latency dominates every turn.

## 3. Candidate 1 — DBOS durable execution: **NOT WORTH IT (now)**

**Concrete meaning for kompend:** attach `DBOSDurability` to the compilation agents,
wrap `compile()` (or each segment run) in a `@DBOS.workflow`, launch DBOS at CLI start,
add `dbos>=2.10.0`. Persisted state: every model request and toolset/MCP I/O as DBOS
steps in DBOS's system database — per DBOS's own documentation that is PostgreSQL
(external knowledge; the `dbos` package is not installed, so this cannot be verified
from the wheel and must be confirmed before any adoption decision).

**Interruption semantics:** re-run the same workflow id and DBOS replays completed
steps deterministically instead of re-executing them.

**Cost:** a new dependency (policy: a decision, not an install) plus an always-on
Postgres system database — architecturally alien to a local-first vault tool that
deliberately keeps its transcript store as a separate SQLite file precisely because
`kompend.db` is recreated on schema bumps (`AGENTS.md:59-66`). Beyond the dependency:
durable boundaries constrain — plain `'parallel'` tool execution is unavailable
(`'parallel_ordered_events'` only), `'mcp'`/`'dynamic'` toolsets are wrapped, and
`RunContext` fields not carried across the boundary read as `None`/defaults
(`pydantic_ai/capabilities/AGENTS.md`, durable-boundary rules). kompend's deps are rich
in-process objects (`GraphCompiler`, concept brokers, usage trackers) not designed to
survive serialization — wiring them through durable contexts would be most of the
migration.

**Why the verdict:** `StepPersistence` already delivers the same user-visible outcome —
an interrupted compile resumes without re-paying completed LLM work — at segment and
conversation granularity, on one SQLite file. The marginal delta of DBOS is intra-step
replay (mid-model-request), and a model request is one HTTP exchange: the cheapest unit
in the flow to lose. Postgres infra to save one HTTP call per crash is a bad trade.

**What would flip it:** (a) kompend becomes a long-running service with many
concurrent compiles and SLAs — then DBOS/Temporal queueing plus replay earns its
database; (b) evidence that StepPersistence actually loses work (segments repeatedly
dying mid-tool-cycle with unusable transcripts — none observed today); (c) a SQLite- or
file-backed durability engine appearing in the wheel (check on each pydantic-ai upgrade).

## 4. Candidate 2 — Prefect ("parallel tool calling"): **NOT WORTH IT**

**Concrete meaning for kompend:** attach `PrefectDurability`, run `agent.run()` inside
a `@flow`, add `prefect>=3.9.0` plus Prefect's backend (server or ephemeral local API —
external, confirm), and accept that kompend's function toolsets get wrapped per-task
(`enter-always`).

**Why the verdict:**

1. The advertised benefit already exists for free: parallel execution of multiple tool
   calls within one model turn is pydantic-ai's default (`'parallel'`,
   `tool_manager.py:51` / `abstract.py:2077`). kompend's tools are local reads; the
   loop's bottleneck is the LLM round-trip, which no tool executor can parallelize.
2. Durability *restricts* parallelism rather than adding it — the engines only offer
   deterministic ordered variants (`DBOSParallelExecutionMode` literal proves the
   design constraint).
3. The integration shape the docs page demonstrates (`PrefectAgent`) is **deprecated**
   in the installed wheel — `prefect/_agent.py:75-86`: *"`PrefectAgent` is deprecated
   in favor of the `PrefectDurability` capability"* — anything written against the
   page's API is already legacy.
4. kompend's real concurrency opportunity is **across segments/sources** (zero
   `asyncio.gather` in `src/`). That is compile-orchestrator work (a bounded
   `TaskGroup` over independent segments, no new deps, measure first) — Prefect
   durability journals a single agent run and does not parallelize segments for you.

**What would flip it:** (a) kompend grows cross-machine scheduled workflows needing
Prefect-the-platform (a product decision, not an agent capability); (b) agents acquire
genuinely slow tools (network fetchers, heavy sub-agents) where intra-turn tool
parallelism outgrows pydantic-ai's in-process default.

## 5. Page-vs-installed contradictions (the valuable part)

1. **Stale premise:** the bead's "checkpointing is currently manual" contradicts
   kompend's own history — `harness-transcripts` (2026-10-27) replaced manual
   checkpoints with harness `StepPersistence`, and the recovery mechanics live at
   `flows/generation.py:675-745`.
2. **Deprecated on its own page:** `PrefectAgent` and `PrefectFunctionToolset`/
   `PrefectMCPToolset` carry `@deprecated` markers in installed 2.66.0; the capability
   (`PrefectDurability`) is the supported path.
3. **Framing inversion:** "parallel tool calling via Prefect" — parallel tool calls are
   a built-in default; durable engines only admit deterministic ordered parallelism.
4. **Extra-gated:** all three advertised engines need new runtime dependencies
   (METADATA lines quoted in §1); the docs present them as plain capabilities.
5. **Side finding (deferred compaction follow-up, bead comment 2026-11-03):** harness
   0.66.0 now has `SummarizingCompaction.retries` —
   `pydantic_ai_harness/compaction/_summarizing_compaction.py:653`
   `retries: int = field(default=1, kw_only=True)`, forwarded at `:1136`. The deferral's
   first condition (harness floor containing pydantic/pydantic-ai#12315) is satisfied;
   the default is still 1, so the second condition (prefer summarising over trimming)
   remains the operator's call.

## 6. Follow-ups

- Close knox-rxgu with this report; revisit both candidates only on the flip
  conditions above.
- Possible future kompend-side bead (no new deps): bounded segment-level concurrency in
  the compile orchestrator — **measure first**; usage tracker and window assumptions
  must be audited for interleaved segments before any implementation.
- Housekeeping: update the knox-rxgu consolidated text (checkpointing premise) when the
  bead is closed.
