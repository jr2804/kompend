# DOX — kompend.agents

<!-- on entry: loaded when agent enters src/kompend/agents -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

PydanticAI-based agents for every LLM call in the kompend pipeline. Each
pipeline stage is a structured-output agent returning a typed Pydantic model;
this package is the **only** layer that talks to pydantic-ai `Agent` directly.

## Ownership

- `src/kompend/agents/` — all PydanticAI agent classes, their dependency
  dataclasses, and the `KompendModel` wrapper
- No pipeline orchestration belongs here — agents are built from their spec
  at the call site (e.g. `compilation/two_phase.py`)

## Architecture

### Class hierarchy

Every agent is a registered `AgentSpec`; the typed class is its facade and
extends `KompendAgentBase[ResultT, DepsT]`. The 37 specs cover every pipeline
stage from extraction to write. Each concrete agent declares one
`SPEC: AgentSpec` (name, `result_type`, `deps_type`, templates, delta-retry
flag, prompt hooks); `build_agent("name", ...)` materialises it. Add a `SPEC`
or a registry entry in `agents/specs.py` instead of hand-wiring a class.

Prompt or validation behaviour a template cannot express belongs on the spec as
`user_prompt` / `system_prompt` / `validate_output`. The hook receives `(agent,
deps)`, so it reads `agent.rlm`, `agent.skills` and `agent._template_loader`; the
base installs it as the method. Never override `render_*` or `validate_output`.

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
| `base/context.py` | context-manager and RLM capability assembly, `_CONTEXT_PRESETS`, `resolve_compact_on_request` |
| `base/retries.py` | transient-retry loop, failure mapping, cache-rejection sniffing, session limits |
| `base/delta_retry.py` | the delta-retry salvage hook and its six helpers |
| `base/protocols.py` | `ContextHost`, `ModelHost`, `RetryHost`, `DeltaRetryHost` — the agent surface each helper reads |

A helper takes its agent as one of those protocols, never as `KompendAgentBase`:
`base/agent.py` imports the helpers, so naming the class would close the cycle.

The base class provides:

- **Model bridge** — `KompendModel` wraps a native pydantic-ai `Model` built by
  `build_model()` from a `provider/model` string; the `pydantic_ai.Agent` is
  built on first `run()`, so tests import the class without a running model
- **Context management** — `build_fallback_compaction()`
  (`context_manager.py`) comes from the `context_manager` preset
  (`"default"`/`"aggressive"`/`"sliding"`/`false`), tuned by
  `[compile.llm.compaction]` (`CompactionConfig`: `judge`, `keep_tokens`,
  `min_growth_tokens`). Default/aggressive is a `_PinningFallbackCompaction`:
  a `GrowthGatedCompaction` around a `TieredCompaction` (clear tool results →
  dedupe superseded reads via `_file_key` → `RelevanceCompaction` when a
  `judge` is set → doctrine summarizer; cheapest first, stopping when the
  history fits), then a token-triggered `SlidingWindowCompaction` for when the
  summarizer cannot answer (`UnexpectedModelBehavior`,
  pydantic/pydantic-ai#12307). The `sliding` preset is
  `build_sliding_compaction()`: a `_GuardedSlidingCompaction` trim on a token
  trigger (not just the 100-message count) with the same pin and gate, every
  figure using `conservative_tokens` (the shared density floor, see the guide).
- **Compaction invariants** — `_pin_first_user_turn` pins the task assignment
  (harness `pin()`, model-invisible) before any tier runs and every strategy
  re-injects it; the gate re-arms only after `min_growth_tokens` of growth since
  the last pass — a skip never moves the baseline, so growth accumulates (a fold
  invalidates the provider prompt cache from the rewrite point on); summaries
  come from `templates/compaction/doctrine.md.j2`; every trim is logged, and a
  retained message that alone exceeds the trigger is dropped (an echo of an
  already-counted response) — never the pin, never half a tool call/return pair.
  A silent trim is indistinguishable from no compaction; never add a path that
  skips the pin, crops instead of folding, or fires on message count alone.
  `compact_on_request: ClassVar[bool]` (default `False`) adds
  model-triggered `CompactOnRequest` for history-threading agents only.
- **RLM (code exploration)** — `RLMCapability` (`rlm.py`, optional
  `[rlm]` extra) attaches from the `compile.llm.rlm` group (`RlmConfig`;
  `enable` is the only switch, explicit fields override the preset baseline).
  It registers an `execute_code` tool (sandboxed REPL, deps context pre-loaded
  from `rlm_context_field`) plus RLM instructions; the sub-model powers
  `llm_query()` in the REPL and is usage-tracked via `KompendModel`. With
  `max_depth > 0` the REPL also exposes `rlm_query(prompt, slice)` — a child
  agent (same main model, own REPL) recursing to the depth budget, guarded by
  a per-root-run node budget (default 16) and cycle detection on (depth,
  prompt, context fingerprint). The per-run REPL cache is keyed by `id(deps)`
  and a lookup must confirm the entry still belongs to that deps object (ids
  are recycled; a stale REPL would run against the previous namespace, guard
  and model). When RLM is enabled the `rlm_context_field` deps value is
  replaced in rendered prompts by `RLM_CONTEXT_PLACEHOLDER` (the deps keep the
  full value for the REPL). Disabled by default (`rlm=False`).
- **Skill injection** — `compile.skills` entries are appended to the system
  prompt under `## Skill: <name>` headings
- **Output validation + retry** — PydanticAI validates against
  `result_type`; `ModelRetry` triggers a retry with feedback
- **Usage tracking** — token usage recorded per call
- **Usage limits** — `run()`/`run_with_messages()` pass `_usage_limits_for(deps)`
  (default: the RLM preset limits, else None → pydantic-ai's implicit 50-request
  cap). Session agents override `_session_limits(units=…, per_unit=…)`: planned
  turns + a fixed 2-request cost + `_retry_requests()`, scaled by
  `headroom_frac` (units/per-unit cost from `budget.*`, the rest class
  constants). A session counts progress only for ids in its own payload;
  `summarize_node` rejects unknown ids with `ModelRetry`, and `NodeSummarizer`
  re-requests unsummarized nodes per `SummarizerSessionPolicy`.
- **Reasoning effort** — config string mapped to provider thinking controls
  (e.g. `think=False` for Ollama)
- **VLM captioning** — `VlmAgent` sends a LIST user prompt: the rendered
  template plus the image bytes as `BinaryContent`. Its deps carry the
  image path and the figure's surroundings, never a ready-made prompt.

### Construction contract

Agents take their limits from the config, not from the call site: a
construction site hands over the `llm` group (`compile.llm` — timeout, retries,
reasoning effort, context manager, context window, fallback chain, delta-retry
switch, data format) and gets every limit it did not spell out.

`_apply_config_defaults()` (in `base/agent.py`) fills every gap from its
`compile.llm` twin (the tool/output retry split collapses to 0 when
`output_retries_enabled` is false); an explicitly passed value always wins.

There is no `max_tokens` and no `budget` kwarg: an answer is bounded by the
model's context window minus its prompt (`llm/AGENTS.md`). The exception is
protocol-driven: Anthropic rejects an unset `max_tokens`, so there the agent
passes a callable `model_settings` computing window - prompt - margin per
request, clamped by the catalog output limit; every other protocol sends
nothing.

`agent.data_format` is the resolved `llm.data_format` (`llm/AGENTS.md`):
prompt-building code reads it off the agent rather than taking a parameter, so
the format cannot drift from the config the agent already carries
(`CategorizeAgent` → `_build_files_payload`).

A **transient** failure (timeout, dropped connection, 429/5xx) is retried in
the base's `_invoke` — `compile.llm.transient_retries` times with
`transient_backoff` doubling. That is the only whole-request retry: stages must
not add backoff loops. Tool, output-schema and session-progress retries are
other axes.

### Failure vocabulary (`failure.py`)

`classify_agent_failure(exc)` maps any raised error onto an `AgentFailureKind`
(`empty_answer`, `schema_mismatch`, `context_overflow`, `transient`, `other`);
`is_retry_pointless(kind)` says when re-sending cannot help. What escapes
`run()`/`run_with_messages()` is always an `LLMOutputError` (or
`LLMTransientError`/`LLMContextOverflowError`) with `__cause__` set — callers
catch that type, not `Exception`. The two broad catches in `base/agent.py` and
the one in `base/retries.py` are THE mapping boundary, re-raising typed, and
the only ones allowed in `base/**`. An empty answer never spends the output
budget: `KompendModel` resamples a blank structured-output response once, then
raises `EmptyAnswerError` (every mode except `text`) — retry feedback cannot
fix a blank answer. Stages call
`record_unit_failure(...)` per failed unit (one log line naming unit, kind and
request count) and `UnitFailureLog.check()` raises once more than
`compile.llm.batch_failure_threshold_ratio` of the units failed. A stage never
swallows a failure into a smaller result set: record it and go on, or stop with
the kinds in the message.

`resolve_retry_budget(llm, retries=…, output_retries=…)` derives the
tool/output retry split from the same config — the agentic factories use it so
a multi-turn agent does not share the single-shot budget. The halves are
separate because the remedies differ (correction attempt vs regeneration).
`KompendAgentBase.tool_retry_budget()` exposes the tool half to the
capabilities that need it (the context manager's summarizer agent).

The agentic harness offers the memory capability only on request
(`AgenticHarness(memory=True)`). The notebook is a `read_memory`/`write_memory`
tool pair: an invented file name gets `ModelRetry`, and once the tool budget
is spent the run dies with the node left unsummarized. `compile.llm.agent_memory`
(default on) decides which stages request it; it only helps a stage where one
instance handles MANY outputs (summarisation, verification, remediation,
distillation) — fresh-agent-per-unit stages gain nothing. `compile.llm.memory_store`
(default `vault`) picks the backend: `vault` roots it at
`.kompend/agent-memory/<agent_name>/MEMORY.md` via `DirectoryMemoryStore`
(`vault_memory.py`), resolved by `build_notebook_store(llm, memory_dir)` — prompt
context, not a cache key.

Three harness capabilities ride on `AgenticHarness` (`agentic_harness.py`):
**StepPersistence** (`step_store`) saves each run's transcript per turn, so a
hard kill resumes from the last snapshot (keys in
`compilation/step_recovery.py`); **ConversationSearch** (`search_history`) adds
a BM25 tool over that store; **SystemReminders** (`reminder_content` +
`reminder_interval`) appends a cache-safe tail reminder from a template.

The delta-retry salvage hook (`base/delta_retry.py`) is opt-in per class via
`supports_delta_retry: ClassVar[bool] = True` (a plain attribute would `__init__`
and never take effect) and fires only with `llm.delta_retry_enabled`.

### KompendModel (`kompend_model.py`)

Thin delegate around any native pydantic-ai `Model`. Adds `_merge_reasoning()`
(reasoning effort → provider settings), `_track_usage()` (input/output tokens),
`_fail_on_empty_answer()` (`EmptyAnswerError` instead of letting a blank
response burn four output retries), `_save_debug()` (LLM debug log), and
`_repair_tool_call_args()`/`_repair_text_part()` (`json_repair` before the
strict validator, for malformed JSON from small local models).
History compression lives on the Agent layer (`build_fallback_compaction`,
`build_sliding_compaction`); the doctrine lives in `templates/compaction/`.

### Dependency types (`types.py`, `deps.py`)

| Deps class          | Used by                                                                               |
| ------------------- | ------------------------------------------------------------------------------------- |
| `PipelineAgentDeps` | extraction, search, capture, lint, verification, categorize, summarize, context-chunk |
| `QueryAgentDeps`    | QueryAgent session (question, context, retriever, sub-questions, findings)            |
| `Wicer*` deps       | the three WICER agents (probe, evaluate, diagnose)                                    |
| `DoctorDeps`        | DoctorAgent                                                                           |
| `DecisionProbeDeps` | DecisionProbeAgent (`decision_probe.py`, the `connectivity-decision` doctor ping: one typed yes/no through the SystemOne endpoint) |
| `SummarizerDeps`    | AgenticSummarizerAgent (payload: `SummaryNode`)                                       |
| `MemoryConsolidationDeps` | MemoryConsolidationAgent (`previous` + delta digest, `target_tokens`)      |
| `ContextChunkDeps`  | AgenticContextChunkAgent (payload: `ContextChunk`)                                    |
| `DistillAgentDeps`  | AgenticDistillAgent (payload: `DistillGroup`, `DistillPage`)                          |
| `VerificationDeps`  | AgenticVerificationAgent (payload: `VerificationClaim`, `VerificationSource`, `VerificationPage`) |
| `WriteDeps`         | AgenticWriteAgent (`source_texts`, served in `source_char_max` windows)                |
| `VlmDeps`          | VlmAgent (`image_file`, the figure's `context`)                                        |

Agentic-session payloads are `TypedDict`s in `types.py`; `DistillPage` is
`total=False` because it mirrors untrusted model output.

## Local Contracts

- **Never crop what the model sees or returns** — a window splits, it never truncates (`.agents/CODE.md` "LLM calls").
- **No text in code.** All prompts are Jinja2 templates in
  `src/kompend/templates/<module>/`. See parent doc.
- **No direct pydantic-ai `Agent` construction outside this package**, and no
  `litellm` imports: pipeline code imports an agent class and calls
  `agent.run(deps)`; models come from `kompend.llm.models.build_model()`.
- **Roles.** Every agent names an `LLMRole` value from `kompend.llm.roles`; add
  one there when adding an agent.
- **Export new agents** from `__init__.py`; **result types** live in the
  consuming module (e.g. `kompend.compilation.schema.ExtractionOutput`).
  The package surface is attached **lazily** (`lazy_loader.attach`, mirroring
  `kompend.rag`): importing `kompend.agents` must not pull the pydantic-ai
  stack — extend the `submod_attrs` map instead of adding a module-level import
  (the eager facade cost ~4-5 s per CLI spawn).

## Relationship to other layers

Distinct from `kompend/acp/` (the editor protocol adapter): editor request →
`kompend/acp/` → MCP tools → pipeline → **these** agents. Per-role model
overrides, skills and how to write a new agent: `docs/reference/pydantic-ai-agents.md`.
