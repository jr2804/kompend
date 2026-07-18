# DOX — knox.agents

## Purpose

PydanticAI-based agents for every LLM call in the KNOX pipeline. Each
pipeline stage (extraction, generation, query, WiCER, capture, lint,
search, verification, VLM, organize, doctor) is a single-turn
structured-output agent that returns a typed Pydantic model. This package
is the **only** layer that talks to pydantic-ai `Agent` directly.

## Ownership

- `src/knox/agents/` — all PydanticAI agent classes, their dependency
  dataclasses, and the `KnoxModel` wrapper
- No prompt text belongs here — templates live in `src/knox/templates/`
- No pipeline orchestration belongs here — agents are instantiated at
  the call site (e.g. `compilation/two_phase.py`)

## Architecture

### Class hierarchy

All agents extend `KnoxAgentBase[ResultT, DepsT]` (ABC):

- `ExtractionAgent`, `ExtractionReduceAgent`
- `GenerationAgent`
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

### KnoxAgentBase responsibilities

Each concrete agent declares:

- `template_name` — Jinja2 user-prompt template (in
  `src/knox/templates/<module>/`)
- `system_template_name` — Jinja2 system-prompt template
- `result_type` — Pydantic model (or `str`) the LLM must return
- `deps_type` — typed context dataclass (default: `PipelineAgentDeps`)

The base class provides:

- **Model bridge** — `KnoxModel` wraps a native pydantic-ai `Model`
  built by `build_model()` from a `provider/model` string
- **Lazy agent construction** — the `pydantic_ai.Agent` is built on
  first `run()` call, so tests can import the class without a running
  model
- **Skill injection** — user-defined skills from `compile.skills` are
  appended to the system prompt under `## Skill: <name>` headings
- **Output validation + retry** — PydanticAI validates against
  `result_type`; `ModelRetry` triggers a retry with feedback
- **Usage tracking** — token usage recorded per call
- **Reasoning effort** — config string mapped to provider thinking
  controls (e.g. `think=False` for Ollama)

### KnoxModel (`knox_model.py`)

Thin delegate around any native pydantic-ai `Model`. Adds:

- `_merge_reasoning()` — maps `reasoning_effort` to provider settings
- `_compact_messages()` — drops `ModelResponse` before retries
- `_track_usage()` — records input/output tokens
- `_save_debug()` — writes request/response to LLM debug log

### Dependency types (`deps.py`)

| Deps class | Used by |
|------------|---------|
| `PipelineAgentDeps` | extraction, search, capture, lint, verification, categorize, summarize, context-chunk |
| `GenerationDeps` | GenerationAgent (adds `concepts_json`, `wicer_findings`) |
| `QueryDeps` | QueryAgent (adds `retrieved_context`, `query`, `mode`) |
| `WicerProbeDeps` | WicerProbeAgent |
| `WicerEvaluateDeps` | WicerEvaluateAgent |
| `WicerDiagnoseDeps` | WicerDiagnoseAgent |
| `DoctorDeps` | DoctorAgent |

## Local Contracts

- **No text in code.** All prompts are Jinja2 templates in
  `src/knox/templates/<module>/`. See parent doc.
- **No direct pydantic-ai `Agent` construction outside this package.**
  Pipeline code imports an agent class and calls `agent.run(deps)`.
- **No `litellm` imports.** Model construction goes through
  `knox.llm.models.build_model()`.
- **Roles.** Every agent corresponds to an `LLMRole` enum value in
  `knox.llm.roles`. Add new roles there when adding new agents.
- **Export new agents** from `__init__.py`.
- **Result types** are Pydantic models defined in the consuming module
  (e.g. `knox.compilation.schema.ExtractionOutput`), not in this package.

## Relationship to other layers

This package (`agents/`, plural) is distinct from `knox/acp/` (singular
`agent/`), which implements the ACP protocol adapter for editors. The
two layers stack:

1. Editor sends an ACP request → `knox/acp/` dispatches to a
   TOML-defined conversational agent
2. Conversational agent calls MCP tools
3. MCP tools drive the pipeline
4. Pipeline internally uses **these** PydanticAI agents (`knox.agents.*`)

See `docs/reference/pydantic-ai-agents.md` for the user-facing guide covering per-role model
overrides, skills configuration, and how to write a new agent.
