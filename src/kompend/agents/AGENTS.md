# DOX — kompend.agents

<!-- on entry: loaded when agent enters src/kompend/agents -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

PydanticAI-based agents for every LLM call in the kompend pipeline. Each
pipeline stage (extraction, generation, query, WiCER, capture, lint,
search, verification, VLM, organize, doctor) is a single-turn
structured-output agent that returns a typed Pydantic model. This package
is the **only** layer that talks to pydantic-ai `Agent` directly.

## Ownership

- `src/kompend/agents/` — all PydanticAI agent classes, their dependency
  dataclasses, and the `KompendModel` wrapper
- No prompt text belongs here — templates live in `src/kompend/templates/`
- No pipeline orchestration belongs here — agents are instantiated at
  the call site (e.g. `compilation/two_phase.py`)

## Architecture

### Class hierarchy

All agents extend `KompendAgentBase[ResultT, DepsT]` (ABC):

- `ExtractionAgent`, `ExtractionReduceAgent`
- `AgenticGenerationAgent`
- `ReferenceExtractionAgent` (bibliographic reference parsing)
- `WicerProbeAgent`, `WicerEvaluateAgent`, `WicerDiagnoseAgent`
- `QueryAgent`
- `QueryExpansionAgent`, `CragGradeAgent`, `HydeAgent`
- `VlmAgent`
- `CaptureAgent`, `DistillAgent`
- `CategorizeAgent`
- `LintRuleAgent`, `RemediationAgent`
- `VerificationAgent`
- `SummarizerAgent`
- `ContextChunkAgent`
- `DoctorAgent`

### KompendAgentBase responsibilities

Each concrete agent declares:

- `template_name` — Jinja2 user-prompt template (in
  `src/kompend/templates/<module>/`)
- `system_template_name` — Jinja2 system-prompt template
- `result_type` — Pydantic model (or `str`) the LLM must return
- `deps_type` — typed context dataclass (default: `PipelineAgentDeps`)

The base class provides:

- **Model bridge** — `KompendModel` wraps a native pydantic-ai `Model`
  built by `build_model()` from a `provider/model` string
- **Lazy agent construction** — the `pydantic_ai.Agent` is built on
  first `run()` call, so tests can import the class without a running
  model
- **Context management** — `ContextManagerCapability`
  (`summarization-pydantic-ai`) is attached from the `context_manager`
  config preset (`"default"`/`"aggressive"`/`"sliding"`/`false`). It
  summarises message history when it nears the model's context budget
  (`max_tokens` = discovered `context_window.max_input`), replacing the
  old `_compact_messages` drop. `on_usage_update` feeds
  `UsageTracker.record_context_usage`; `on_before/after_compress` log
  compression events. Disabled by default at the code-API level
  (`context_manager=False`); enabled by config (`compile.llm.context_manager`).
- **RLM (code exploration)** — `RLMCapability` (`rlm.py`, optional
  `[rlm]` extra) attaches from the `compile.llm.rlm` group
  (`RlmConfig` in `core/config/_configs.py`:
  `enable`, `preset`, `max_tool_calls`, `code_timeout`, `sub_model`,
  `max_depth`, `node_budget` — `enable` is the only switch, explicit
  fields override the preset baseline; no scalar shorthand). It
  registers an `execute_code` tool (sandboxed REPL, deps context
  pre-loaded from `rlm_context_field`) and RLM mode instructions; the
  sub-model powers `llm_query()` inside the REPL and is usage-tracked
  via `KompendModel`. With `max_depth > 0` the REPL also exposes
  `rlm_query(prompt, slice)` — a child agent (same main model, own REPL)
  recursing up to the depth budget, guarded by a per-root-run node
  budget (default 16) and cycle detection on (depth, prompt, context
  fingerprint). The per-run REPL cache is keyed by `id(deps)`; a lookup
  must confirm the entry still belongs to that deps object — ids are
  recycled once a deps object is collected, and a stale REPL would run
  the next agent against the previous run's namespace, guard and model.
  Run
  tool-calls are bounded by `UsageLimits` from the preset. When RLM is
  enabled, the `rlm_context_field` deps value is replaced by a
  placeholder in rendered prompts (`RLM_CONTEXT_PLACEHOLDER`) so the
  model explores the context via code instead of reading it inline
  (the deps still carry the full value for the REPL). Composes with
  context management (compaction bounds the conversation, RLM keeps
  document data out of it). Disabled by default (`rlm=False`).
- **Skill injection** — user-defined skills from `compile.skills` are
  appended to the system prompt under `## Skill: <name>` headings
- **Output validation + retry** — PydanticAI validates against
  `result_type`; `ModelRetry` triggers a retry with feedback
- **Usage tracking** — token usage recorded per call
- **Usage limits** — `run()`/`run_with_messages()` pass
  `_usage_limits_for(deps)` (default: the RLM preset limits, else None →
  pydantic-ai's implicit 50-request cap). Multi-turn tool agents override the
  hook; `AgenticSummarizerAgent` derives `request_limit` from node count
  (`requests_per_node` — instance field set by the pipeline from
  `budget.summary_requests_per_node` — plus a fixed 2-request session cost —
  the "no more nodes" probe and the closing turn — scaled by `headroom_frac`),
  and `AgenticDistillAgent` from group count (~3 requests/group,
  `requests_per_group`), every factor a kwarg with a default. The derived
  budget is a floor, not a guarantee: unsummarized nodes are re-requested in a
  later session (caller-side, in `NodeSummarizer`, batched and stall-retried
  per `SummarizerSessionPolicy` from `budget.summary_session_*`). A session
  counts as progress only for ids in its own payload, and `summarize_node`
  rejects unknown ids with `ModelRetry` so a mangled id can never masquerade as
  progress
- **Reasoning effort** — config string mapped to provider thinking
  controls (e.g. `think=False` for Ollama)

### Construction contract

Agents take their limits from the config, not from the call site. A
construction site hands over the two config groups and gets every limit it
did not spell out:

| Kwarg | Source |
|-------|--------|
| `llm` | the `compile.llm` group — timeout, retries, reasoning effort, context manager, context window, fallback chain, delta-retry switch |
| `budget` | the `compile.budget` group — the response ceiling |

`KompendAgentBase._apply_config_defaults()` fills the gaps: `timeout` ←
`llm.timeout`, `retries` ← `llm.output_retries` (0 when
`output_retries_enabled` is false), `reasoning_effort` ←
`llm.reasoning_effort`, `context_manager` ← `llm.context_manager`,
`context_window` ← `llm.context_window`, `fallback_models` ←
`llm.fallback_chain`, `delta_retry_enabled` ← `llm.delta_retry_enabled`,
`max_tokens` ← `budget.<output_budget_key>`. An explicitly passed value
always wins over the config.

Each agent class declares `output_budget_key` — the `budget.*_output_max`
that bounds its response (`extract_output_max`, `gen_output_max`,
`reference_output_max`, `overview_output_max`, `verify_output_max`,
`lint_output_max`, `wicer_probe_output_max`, `wicer_eval_output_max`).
Agents without a dedicated key use the generic `budget.agent_output_max`.

`resolve_retry_budget(llm, retries=…, output_retries=…)` derives the
tool/output retry split from the same config — the agentic factories use it
so a multi-turn agent does not share the single-shot retry budget.

An agent built with only `model_string` gets the provider's default output
ceiling and the hardcoded `_DEFAULT_RETRIES` — the response is truncated by
the provider and the structured output fails with
`Exceeded maximum output retries`.

The delta-retry salvage hook is opt-in per class: a subclass sets
`supports_delta_retry: ClassVar[bool] = True`, and the hook fires only when
`delta_retry_enabled` (from `llm.delta_retry_enabled`) is on. It is a
`ClassVar` — a plain class attribute would be shadowed by the dataclass
`__init__` and never take effect.

### KompendModel (`kompend_model.py`)

Thin delegate around any native pydantic-ai `Model`. Adds:

- `_merge_reasoning()` — maps `reasoning_effort` to provider settings
- `_track_usage()` — records input/output tokens
- `_save_debug()` — writes request/response to LLM debug log
- `_repair_tool_call_args()` / `_repair_text_part()` — `json_repair`
  before the strict validator (fixes malformed JSON from small local models)

History compression is **not** handled here. It moved up to the
pydantic-ai Agent layer via `ContextManagerCapability` (see
`KompendAgentBase`); the old `_compact_messages()` workaround — which
dropped all `ModelResponse` on retry, defeating pydantic-ai's retry
feedback loop — was removed.

### Dependency types (`types.py`, `deps.py`)

| Deps class          | Used by                                                                               |
| ------------------- | ------------------------------------------------------------------------------------- |
| `PipelineAgentDeps` | extraction, search, capture, lint, verification, categorize, summarize, context-chunk |
| `QueryDeps`         | QueryAgent (adds `retrieved_context`, `query`, `mode`)                                |
| `WicerProbeDeps`    | WicerProbeAgent                                                                       |
| `WicerEvaluateDeps` | WicerEvaluateAgent                                                                    |
| `WicerDiagnoseDeps` | WicerDiagnoseAgent                                                                    |
| `DoctorDeps`        | DoctorAgent                                                                           |
| `SummarizerDeps`    | AgenticSummarizerAgent (payload: `SummaryNode`)                                       |
| `ContextChunkDeps`  | AgenticContextChunkAgent (payload: `ContextChunk`)                                    |
| `DistillAgentDeps`  | AgenticDistillAgent (payload: `DistillGroup`, `DistillPage`)                          |
| `VerificationDeps`  | AgenticVerificationAgent (payload: `VerificationClaim`, `VerificationSource`, `VerificationPage`) |
| `WriteDeps`         | AgenticWriteAgent (payload: `source_texts` — the full sources, served in windows of `source_char_max` characters) |

Agentic-session payload structures are `TypedDict`s declared in `types.py`
alongside the other module-level classes (the sorter groups them after the
deps dataclasses) — never bare `dict[str, Any]` at the agent boundary, so a
mistyped key is a `ty` error. `DistillPage` is `total=False` because it
mirrors untrusted model output; callers re-read every key defensively.

## Local Contracts

- **No text in code.** All prompts are Jinja2 templates in
  `src/kompend/templates/<module>/`. See parent doc.
- **No direct pydantic-ai `Agent` construction outside this package.**
  Pipeline code imports an agent class and calls `agent.run(deps)`.
- **No `litellm` imports.** Model construction goes through
  `kompend.llm.models.build_model()`.
- **Roles.** Every agent corresponds to an `LLMRole` enum value in
  `kompend.llm.roles`. Add new roles there when adding new agents.
- **Export new agents** from `__init__.py`.
- **Result types** are Pydantic models defined in the consuming module
  (e.g. `kompend.compilation.schema.ExtractionOutput`), not in this package.

## Relationship to other layers

This package (`agents/`, plural) is distinct from `kompend/acp/` (singular
`agent/`), which implements the ACP protocol adapter for editors. The
two layers stack:

1. Editor sends an ACP request → `kompend/acp/` dispatches to a
   TOML-defined conversational agent
2. Conversational agent calls MCP tools
3. MCP tools drive the pipeline
4. Pipeline internally uses **these** PydanticAI agents (`kompend.agents.*`)

See `docs/reference/pydantic-ai-agents.md` for the user-facing guide covering per-role model
overrides, skills configuration, and how to write a new agent.
