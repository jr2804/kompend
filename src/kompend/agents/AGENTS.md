# DOX — kompend.agents

<!-- on entry: loaded when agent enters src/kompend/agents -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

PydanticAI-based agents for every LLM call in the kompend pipeline. Each
pipeline stage is a structured-output agent returning a typed Pydantic model.
This package is the **only** layer that talks to pydantic-ai `Agent` directly.

## Ownership

- `src/kompend/agents/` — all PydanticAI agent classes, their dependency
  dataclasses, and the `KompendModel` wrapper
- No pipeline orchestration belongs here — agents are built from their spec
  at the call site (e.g. `compilation/two_phase.py`)

## Architecture

### Class hierarchy

Every agent is a registered `AgentSpec`; the typed class is its facade and
extends `KompendAgentBase[ResultT, DepsT]`. The 36 specs cover every pipeline
stage from extraction to write.

### KompendAgentBase responsibilities

Each concrete agent declares one `SPEC: AgentSpec` (name, `result_type`,
`deps_type`, templates, delta-retry flag, prompt hooks); `build_agent("name",
...)` materialises it and the class is only the typed facade. Add a `SPEC` and
a registry entry in `agents/specs.py` instead of hand-wiring a class.

Prompt or validation behaviour a template cannot express belongs on the spec
as `user_prompt` / `system_prompt` / `validate_output`. The hook receives
`(agent, deps)`, so it reads `agent.rlm`, `agent.skills` and
`agent._template_loader` as the old override did; the base installs it as the
method. Never override `render_*` or `validate_output` in a concrete class.

### The `base/` package

`KompendAgentBase` and its machinery, split by concern. The import path
`kompend.agents.base` is stable — `base/__init__.py` re-exports the public
surface, so importers never name a submodule:

| Module | Scope |
|--------|-------|
| `base/agent.py` | the `KompendAgentBase` dataclass: construction defaults (`_apply_config_defaults`), run entry points, prompt rendering, lazy `_build_agent()` |
| `base/spec.py` | `AgentSpec` (data-only agent definition), `agent_class()`, `install_hooks()` and `build_agent()`; the agent modules' `SPEC` constants are the definitions |
| `base/deps.py` | `AgentDeps` protocol, `ResultT`/`DepsT` typevars, `resolve_retry_budget()`, the `UNSET_*` sentinels |
| `base/models.py` | top-level model assembly (failover chain) and output-ceiling arithmetic |
| `base/context.py` | context-manager and RLM capability assembly, `_CONTEXT_PRESETS` |
| `base/retries.py` | transient-retry loop, failure mapping, cache-rejection sniffing, session limits |
| `base/delta_retry.py` | the delta-retry salvage hook and its six helpers |

The base class provides:

- **Model bridge** — `KompendModel` wraps a native pydantic-ai `Model`
  built by `build_model()` from a `provider/model` string
- **Lazy agent construction** — the `pydantic_ai.Agent` is built on
  first `run()` call, so tests can import the class without a running
  model
- **Context management** — `build_fallback_compaction()`
  (`context_manager.py`) is attached from the `context_manager` config
  preset (`"default"`/`"aggressive"`/`"sliding"`/`false`): a
  `FallbackCompaction` of `SummarizingCompaction` (at a fraction of the
  discovered `context_window.max_input`), then a token-triggered
  `SlidingWindowCompaction` keeping half the trigger, used only when the
  summarizer cannot answer (`UnexpectedModelBehavior`, e.g. one unknown-tool
  call exhausting the harness's implicit budget of 1,
  pydantic/pydantic-ai#12307); other errors burn. Disabled at the code-API
  level, enabled by config.
- **RLM (code exploration)** — `RLMCapability` (`rlm.py`, optional
  `[rlm]` extra) attaches from the `compile.llm.rlm` group (`RlmConfig` in
  `core/config/_configs.py`; `enable` is the only switch, explicit fields
  override the preset baseline). It registers an `execute_code` tool
  (sandboxed REPL, deps context pre-loaded from `rlm_context_field`) plus
  RLM instructions; the sub-model powers `llm_query()` in the REPL and is
  usage-tracked via `KompendModel`. With
  `max_depth > 0` the REPL also exposes `rlm_query(prompt, slice)` — a
  child agent (same main model, own REPL) recursing to the depth budget,
  guarded by a per-root-run node budget (default 16) and cycle detection on
  (depth, prompt, context fingerprint). The per-run REPL cache is keyed by
  `id(deps)` and a lookup must confirm the entry still belongs to that deps
  object (ids are recycled; a stale REPL would run against the previous
  run's namespace, guard and model). When RLM is enabled the
  `rlm_context_field` deps value is replaced in rendered prompts by
  `RLM_CONTEXT_PLACEHOLDER` (the deps keep the full value for the REPL).
  Disabled by default (`rlm=False`).
- **Skill injection** — user-defined skills from `compile.skills` are
  appended to the system prompt under `## Skill: <name>` headings
- **Output validation + retry** — PydanticAI validates against
  `result_type`; `ModelRetry` triggers a retry with feedback
- **Usage tracking** — token usage recorded per call
- **Usage limits** — `run()`/`run_with_messages()` pass
  `_usage_limits_for(deps)` (default: the RLM preset limits, else None →
  pydantic-ai's implicit 50-request cap). Every session agent overrides the
  hook through `_session_limits(units=…, per_unit=…)` — one formula: planned
  turns + a fixed 2-request session cost + `_retry_requests()` (resolved
  tool+output budgets — unbudgeted retries abort on the first retried call),
  scaled by `headroom_frac`. Only the unit count and per-unit cost are per
  agent (nodes and groups from `budget.*`, the rest class constants). A
  session counts progress only for ids in its own payload; `summarize_node`
  rejects unknown ids with `ModelRetry`. Unsummarized nodes are re-requested
  caller-side (`NodeSummarizer`, per `SummarizerSessionPolicy` from
  `budget.summary_session_*`).
- **Reasoning effort** — config string mapped to provider thinking
  controls (e.g. `think=False` for Ollama)
- **VLM captioning** — `VlmAgent` sends a LIST user prompt: the rendered
  template plus the image bytes as `BinaryContent`. Its deps carry the
  image path and the figure's surroundings, never a ready-made prompt.

### Construction contract

Agents take their limits from the config, not from the call site. A
construction site hands over the two config groups and gets every limit it
did not spell out:

| Kwarg | Source |
|-------|--------|
| `llm` | the `compile.llm` group — timeout, retries, reasoning effort, context manager, context window, fallback chain, delta-retry switch, data format |

`_apply_config_defaults()` (in `base/agent.py`) fills every gap from its
`compile.llm` twin (the tool/output retry split collapses to 0 when
`output_retries_enabled` is false). An explicitly passed value always wins.

There is no `max_tokens` and no `budget` kwarg: an answer is bounded by the
model's context window minus its prompt (`llm/AGENTS.md`). The exception is
protocol-driven: Anthropic rejects an unset `max_tokens`, so there the agent
passes a callable `model_settings` computing window - prompt - margin per
request, clamped by the model's catalog output limit; every other protocol
sends nothing.

`agent.data_format` is the resolved `llm.data_format` (see `llm/AGENTS.md`):
prompt-building code reads it off the agent instead of taking a parameter, so
the format cannot drift from the config the agent already carries
(`CategorizeAgent` → `_build_files_payload`).

A **transient** failure (timeout, dropped connection, 429/5xx) is retried in
the base's `_invoke` — `compile.llm.transient_retries` times with
`transient_backoff` doubling. That is the only whole-request retry: stages must
not add backoff loops of their own. Tool, output-schema and session-progress
retries are other axes.

### Failure vocabulary (`failure.py`)

`classify_agent_failure(exc)` maps any raised error onto an `AgentFailureKind`
(`empty_answer`, `schema_mismatch`, `context_overflow`, `transient`, `other`);
`is_retry_pointless(kind)` says when re-sending the request cannot help.
An empty answer never spends the output budget: `KompendModel` raises
`EmptyAnswerError` on the first blank response, for every output mode except
`text`. Stages call `record_unit_failure(...)` per failed unit (one log line
naming unit, kind and request count) and `UnitFailureLog.check()` raises once
more than `compile.llm.batch_failure_threshold_ratio` of the units failed. A
stage never swallows a failure into a smaller result set: record it and go on,
or stop with the kinds in the message.

`resolve_retry_budget(llm, retries=…, output_retries=…)` derives the
tool/output retry split from the same config — the agentic factories use it so
a multi-turn agent does not share the single-shot budget. The halves are
separate because the remedies differ (correction attempt vs regeneration).
`KompendAgentBase.tool_retry_budget()` exposes the tool half to the
capabilities that need it (the context manager's summarizer agent).

The agentic harness offers the memory capability only on request
(`AgenticHarness(memory=True)`). The notebook is a `read_memory`/`write_memory`
tool pair: an invented file name gets `ModelRetry`, and once the tool budget
is spent the run dies with the node left unsummarized.
`compile.llm.agent_memory` (default on) decides which stages request it. The
notebook is owned by ONE agent instance, so it only helps a stage where one
instance handles MANY outputs (summarisation, verification, remediation,
distillation); fresh-agent-per-unit stages gain nothing. `compile.llm.memory_store`
(default `vault`) picks the backend: `vault` roots the notebook at
`.kompend/agent-memory/<agent_name>/MEMORY.md` via `DirectoryMemoryStore`
(`vault_memory.py`), resolved by `build_notebook_store(llm, memory_dir)`. The
notebook is agent-written prompt context, not a cache input: it is not part
of the summary cache key.

Three more harness capabilities ride on `AgenticHarness` (`agentic_harness.py`):
**StepPersistence** (`step_store`) saves every run's transcript after each turn,
so a hard kill resumes from the last complete snapshot (keys in
`compilation/step_recovery.py`). **ConversationSearch** (`search_history`) adds
a BM25 tool over the same store, scoped to the conversation. **SystemReminders**
(`reminder_content` + `reminder_interval`) appends a cache-safe tail reminder
rendered from a template.

The delta-retry salvage hook (`base/delta_retry.py`) is opt-in per class: a
subclass sets `supports_delta_retry: ClassVar[bool] = True`, and it fires only
when `llm.delta_retry_enabled` is on. It must be a `ClassVar` — a plain
attribute would `__init__` and never take effect.

### KompendModel (`kompend_model.py`)

Thin delegate around any native pydantic-ai `Model`. Adds:

- `_merge_reasoning()` — maps `reasoning_effort` to provider settings
- `_track_usage()` — records input/output tokens
- `_fail_on_empty_answer()` — raises `EmptyAnswerError` instead of letting a
  blank response burn four output retries
- `_save_debug()` — writes request/response to LLM debug log
- `_repair_tool_call_args()` / `_repair_text_part()` — `json_repair`
  before the strict validator (fixes malformed JSON from small local models)

History compression lives on the Agent layer (`build_fallback_compaction`).

### Dependency types (`types.py`, `deps.py`)

| Deps class          | Used by                                                                               |
| ------------------- | ------------------------------------------------------------------------------------- |
| `PipelineAgentDeps` | extraction, search, capture, lint, verification, categorize, summarize, context-chunk |
| `QueryDeps`         | QueryAgent (adds `retrieved_context`, `query`, `mode`)                                |
| `Wicer*` deps       | the three WICER agents (probe, evaluate, diagnose)                                    |
| `DoctorDeps`        | DoctorAgent                                                                           |
| `SummarizerDeps`    | AgenticSummarizerAgent (payload: `SummaryNode`)                                       |
| `MemoryConsolidationDeps` | MemoryConsolidationAgent (`previous` + delta digest, `target_tokens`)      |
| `ContextChunkDeps`  | AgenticContextChunkAgent (payload: `ContextChunk`)                                    |
| `DistillAgentDeps`  | AgenticDistillAgent (payload: `DistillGroup`, `DistillPage`)                          |
| `VerificationDeps`  | AgenticVerificationAgent (payload: `VerificationClaim`, `VerificationSource`, `VerificationPage`) |
| `WriteDeps`         | AgenticWriteAgent (`source_texts`, served in `source_char_max` windows)                |
| `VlmDeps`          | VlmAgent (`image_file`, the figure's `context`)                                        |

Agentic-session payloads are `TypedDict`s in `types.py`, never bare
`dict[str, Any]` at the agent boundary. `DistillPage` is `total=False` because
it mirrors untrusted model output.

## Local Contracts

- **Never crop what the model sees or returns** — a window splits, it never truncates (`.agents/CODE.md` "LLM calls").
- **No text in code.** All prompts are Jinja2 templates in
  `src/kompend/templates/<module>/`. See parent doc.
- **No direct pydantic-ai `Agent` construction outside this package.**
  Pipeline code imports an agent class and calls `agent.run(deps)`.
- **No `litellm` imports.** Model construction goes through
  `kompend.llm.models.build_model()`.
- **Roles.** Every agent corresponds to an `LLMRole` enum value in
  `kompend.llm.roles`. Add new roles there when adding new agents.
- **Export new agents** from `__init__.py`; **result types** live in the
  consuming module (e.g. `kompend.compilation.schema.ExtractionOutput`).
  The package surface is attached **lazily** (`lazy_loader.attach`, mirroring
  `kompend.rag`): importing `kompend.agents` must not pull the pydantic-ai
  stack — add a new agent by extending its module's list in the
  `submod_attrs` map, never with a module-level import (measured: the eager
  facade made every CLI spawn pay ~4-5 s of vendor import floor).

## Relationship to other layers

Distinct from `kompend/acp/` (the editor protocol adapter). Stack: editor
request → `kompend/acp/` → MCP tools → pipeline → **these** PydanticAI
agents. See `docs/reference/pydantic-ai-agents.md` for per-role model
overrides, skills and how to write a new agent.
