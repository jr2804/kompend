# PydanticAI Agents & Skills

Every LLM call in KNOX routes through a **PydanticAI agent** in
`knox.agents.*`. Each pipeline stage — extraction, generation, query,
WiCER, capture, lint, search — is a single-turn structured-output call
that returns a typed Pydantic model. PydanticAI handles JSON extraction,
validation, and automatic retry on schema mismatch, so pipeline code never
parses raw LLM output.

This guide covers:

- How agents are structured and wired into the pipeline
- Per-role model overrides (`compile.llm.role_models`)
- Writing and configuring skills (`compile.skills`)
- Agent architecture for developers

---

## Agent architecture

### Base class

All agents inherit from
[`KnoxAgentBase[ResultT, DepsT]`](https://codeberg.org/jr2804/knox/src/branch/main/src/knox/agents/base.py)
— a KNOX wrapper around `pydantic_ai.Agent` that adds:

- **Model bridge** — builds a [`KnoxModel`](#knoxmodel) from a
  `provider/model` string via `build_model()`.
- **Template-driven prompts** — system and user prompts are Jinja2
  templates loaded by `TemplateLoader`. No prompt text lives in Python.
- **Skill injection** — user-defined skills are appended to the system
  prompt at render time.
- **Output validation + retry** — PydanticAI validates the LLM response
  against `result_type`; on mismatch, it sends actionable feedback and
  retries up to `output_retries` times.
- **Usage tracking** — token usage is recorded per call for
  observability.
- **Reasoning effort** — maps `reasoning_effort` config to the
  provider's thinking controls (e.g. `think=False` for Ollama).

### Available agents

| Agent | Stage | Result type | Role |
|-------|-------|-------------|------|
| `ExtractionAgent` | Compilation T3 | `ExtractionOutput` | `compile.extract` |
| `ExtractionReduceAgent` | Compilation T3 (reduce) | `ExtractionOutput` | `compile.extract` |
| `GenerationAgent` | Compilation T4 | `GenerationOutput` | `compile.generate` |
| `WicerProbeAgent` | Compilation T4 (WiCER probe) | `WicerProbeOutput` | `wicer.probe` |
| `WicerEvaluateAgent` | Compilation T4 (WiCER judge) | `WicerEvaluateOutput` | `wicer.judge` |
| `WicerDiagnoseAgent` | Compilation T4 (WiCER refine) | `WicerDiagnoseOutput` | `wicer.refine` |
| `QueryAgent` | Query / RAG | `str` | `synthesize` |
| `QueryExpansionAgent` | Search | `QueryExpansionOutput` | `search.expand` |
| `CragGradeAgent` | Search (CRAG) | `CragGradeOutput` | `search.crag` |
| `HydeAgent` | Search (HyDE) | `str` | `search.hyde` |
| `VlmAgent` | Resolution (VLM) | `str` | `vlm` |
| `CaptureAgent` | Capture | `CaptureOutput` | `capture` |
| `DistillAgent` | Capture (distill) | `DistillOutput` | `capture.distill` |
| `CategorizeAgent` | Organize | `CategoryTree` | `organize.categorize` |
| `LintRuleAgent` | Lint | `dict[str, Any]` | `lint.*` |
| `RemediationAgent` | Lint (remediate) | `str` | `lint.remediate-concept` |
| `VerificationAgent` | Trust | `dict[str, Any]` | `verify` |
| `SummarizerAgent` | Summarize | `str` | `synthesize` |
| `ContextChunkAgent` | Chunking | `str` | `chunking.contextual` |
| `DoctorAgent` | Doctor | `str` | `doctor` |

### KnoxModel

`KnoxModel` is a thin delegate that wraps any native pydantic-ai `Model`
and adds KNOX-specific concerns:

- **Reasoning effort mapping** — `_merge_reasoning()` translates the
  config string (`"none"`, `"low"`, `"medium"`, `"high"`, …) into the
  provider's thinking settings at request time.
- **Context compaction** — `_compact_messages()` drops prior
  `ModelResponse` messages before each retry, preserving full validation
  feedback while reducing token cost.
- **Usage tracking** — `_track_usage()` records input/output tokens for
  observability and budget enforcement.
- **Debug dumps** — `_save_debug()` writes the full request/response
  pair to the LLM debug log when enabled.

### How agents are wired into the pipeline

Agents are instantiated by the compilation orchestrator at the point of
use, not via a global registry. For example, `two_phase.py` creates an
`ExtractionAgent` per chunk batch:

```python
agent = ExtractionAgent(
    model_string=self._model or "",
    timeout=self._timeout,
    max_tokens=self._budget.extract_output_max,
    skills=self._config.skills,
    retries=self._config.llm.output_retries,
    reasoning_effort=self._config.llm.reasoning_effort,
)
result = agent.run(PipelineAgentDeps(text=chunk, source_id=stem))
```

All agents receive the same `skills` dict and `retries`/`reasoning_effort`
from `compile.llm`.

---

## Per-role model overrides

Different pipeline stages have different cost/quality trade-offs. You can
pin a specific model per LLM role using `compile.llm.role_models`:

```toml
[compile.llm]
default_model = "ollama/qwen3"

[compile.llm.role_models]
"compile.extract" = "ollama/gpt-oss:20b"
"compile.generate" = "openai/gpt-4o"
"vlm" = "openai/gpt-4o"
"wicer.judge" = "anthropic/claude-sonnet-4-20250514"
```

Role keys are the config identifiers under `[compile.llm.role_models]`. When a role is not listed, `default_model` is used.

Common roles:

| Role | Stage |
|------|-------|
| `compile.extract` | Entity/concept extraction |
| `compile.generate` | Wiki page generation |
| `wicer.probe` | WiCER diagnostic probe |
| `wicer.judge` | WiCER quality evaluation |
| `wicer.refine` | WiCER content refinement |
| `vlm` | Image captioning (VLM enricher) |
| `search.crag` | CRAG relevance grading |
| `search.expand` | Query expansion |
| `search.hyde` | HyDE hypothetical answer |
| `synthesize` | Query answer synthesis |
| `verify` | Trust verification |
| `capture` | Source capture |
| `capture.distill` | Capture distillation |
| `organize.categorize` | Page categorization |
| `chunking.contextual` | Contextual chunk enrichment |
| `doctor` | Doctor diagnostics |
| `agent` | Conversational ACP agent |

---

## Skills

Skills are user-defined instruction blocks injected into agent system
prompts. They let domain experts provide terminology, style guides, or
extraction hints **without touching Python code or Jinja2 templates**.

### How skills work

1. Define one or more skills under `[compile.skills."<name>"]`.
2. Each skill has a single `instructions` field (free-form text).
3. At system-prompt render time, each active skill is appended under a
   `## Skill: <name>` heading, separated by horizontal rules.
4. All PydanticAI agents receive the same set of skills — they are
   pipeline-global.

### Configuring skills

```toml
[compile.skills."domain-terms"]
instructions = '''
- VAD: Voice Activity Detection
- NR: Noise Reduction
- DOA: Direction of Arrival
- AEC: Acoustic Echo Cancellation
'''

[compile.skills."style-guide"]
instructions = '''
- Use sentence-case for page titles
- Cross-link related concepts with [[wikilinks]]
- Prefer active voice
- Keep summaries under 3 sentences
'''
```

### When to use skills vs. templates

| Need | Solution |
|------|----------|
| Domain terminology, abbreviations, style rules | **Skill** (`compile.skills`) |
| Change what the LLM is asked to do | **Template** (Jinja2 `.md.j2`) |
| Change the output schema | **Result type** (Pydantic model) |
| Change which model is used | **Role model** (`compile.llm.role_models`) |

Skills are additive — they augment the system prompt without replacing
it. Templates define the core instruction; skills provide additional
context.

---

## MCP integration (future)

PydanticAI's tool/capabilities system allows attaching MCP servers to
specific agents, enabling an agent to call external tools (web search,
databases, APIs) during its LLM round. This is planned for a future
phase. The current architecture is designed to accommodate this without
pipeline changes:

- `KnoxAgentBase._build_agent()` constructs the `pydantic_ai.Agent` —
  MCP tool servers would be registered here.
- The pipeline calls `agent.run(deps)` — tool calls happen transparently
  inside PydanticAI's agent loop.
- No changes to pipeline callers are needed.

---

## Writing a new agent

1. **Create the agent class** in `src/knox/agents/<name>.py`:

```python
from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps
from knox.compilation.schema import MyOutput  # or define a Pydantic model


class MyAgent(KnoxAgentBase[MyOutput, PipelineAgentDeps]):
    """One-line description of what this agent does."""

    template_name = "my_prompt.md.j2"
    system_template_name = "my_system.md.j2"
    result_type = MyOutput
```

2. **Create the Jinja2 templates** in
   `src/knox/templates/<module>/`:

   - `<name>_system.md.j2` — system prompt
   - `<name>_prompt.md.j2` — user prompt template (receives `deps`
     fields as Jinja2 variables)

3. **Register the agent** — wire it into the compilation pipeline at the
   call site (see the existing agents in `src/knox/agents/` for the
   pattern).

4. **Export** the agent from `src/knox/agents/__init__.py`.

5. **Route it to a model** in `src/knox/core/pipeline/config.py` if it
   needs a role override (see `compile.llm.role_models`).

See the [LLM Configuration guide](../guides/llm-config.md) for model
strings, fallback chains, and reasoning-effort controls.
