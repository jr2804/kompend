# PydanticAI Agents & Skills

Every LLM call in kompend routes through a **PydanticAI agent** in
`kompend.agents.*`. Agents come in two shapes:

- **Single-shot** — one structured-output call that returns a typed Pydantic
  model.
- **Agentic** — a multi-turn loop with tools (`fetch_gaps`, `read_page`, …)
  that iterates over work in a single session.

PydanticAI handles JSON extraction, validation, and automatic retry on schema
mismatch, so pipeline code never parses raw LLM output. Which actions are
agentic and which are deliberately single-shot is a recorded decision — see the
[decision log](../architecture/decisions.md#decision-which-llm-actions-stay-single-shot-by-design).

This guide covers:

- How agents are structured and wired into the pipeline
- The agent inventory (single-shot and agentic)
- Writing and configuring skills (`compile.skills`)
- Writing a new agent

---

## Agent architecture

### The spec registry

Every agent is declared as an `AgentSpec` — a frozen, data-only definition in
the agent's module (`SPEC: AgentSpec`) and registered in
`src/kompend/agents/specs.py`. `all_specs()` returns all 37 definitions,
`get_spec("extraction")` resolves one by name, and `AgentName` is the string
vocabulary used by the registry and tests.

The spec carries the name, result type, deps type, template names and prefix,
delta-retry opt-in, temperature, native-output and retry defaults, and the
optional prompt/validation hooks. `build_agent("name", model_string=...)`
materialises the spec into a typed `KompendAgentBase[ResultT, DepsT]` subclass
(cached by name) and applies the spec's scalar defaults only where the call
site did not pass a value. Call sites may keep instantiating the typed class
directly when that reads better; the spec is the declaration, not a
construction mandate.

```python
from kompend.agents.base import build_agent
from kompend.agents.deps import PipelineAgentDeps

agent = build_agent("extraction", model_string="ollama/gemma4:e2b")
result = agent.run(PipelineAgentDeps(text=chunk, source_id=stem))
```

### Base class

All agents inherit from `KompendAgentBase[ResultT, DepsT]` — a kompend wrapper around
`pydantic_ai.Agent` that adds:

- **Model bridge** — builds a [`KompendModel`](#kompendmodel) from a
  `provider/model` string.
- **Typed result and deps** — the `AgentSpec` supplies the output schema
  (`result_type`) and the run-context dataclass (`deps_type`); the base hands
  them to pydantic-ai as `output_type`. PydanticAI validates and retries on
  mismatch, so the spec never relies on its non-validating `output_schema`
  form.
- **Template-driven prompts** — system and user prompts are Jinja2
  templates loaded by `TemplateLoader`. No prompt text lives in Python.
- **Prompt hooks** — agents whose prompt is computed rather than templated
  declare `user_prompt` / `system_prompt` on the spec. The hook receives
  `(agent, deps)`, so it can read `agent.rlm`, `agent.skills` and
  `agent._template_loader`; `install_hooks()` installs it on the materialised
  class. Never override `render_*` or `validate_output` in a concrete class.
- **Skill injection** — user-defined skills are appended to the system
  prompt at render time.
- **Output validation + retry** — PydanticAI validates the LLM response
  against the output type; on mismatch, it sends actionable feedback and
  retries up to `output_retries` times. `validate_output` is installed from
  the spec as a static hook.
- **Usage tracking** — token usage is recorded per call for
  observability.
- **Reasoning effort** — maps `reasoning_effort` config to the
  provider's thinking controls (e.g. `think=False` for Ollama).
- **Tools and capabilities** — agentic agents pass `tools=` and
  `AgenticHarness.build_capabilities()` to get a multi-turn tool loop.

A class existing is **not** evidence that it is wired: the registry declares
the definition, while production call sites still choose which agent to build
and when. The inventory below is a map of the definitions, not a wiring claim.

---

## Agent inventory

`all_specs()` is the authoritative list; the tables below group the 37
definitions by pipeline area. `Output` is the spec's `result_type` and `Deps`
its `deps_type`.

### Compilation and references

| Spec | Output | Deps |
|------|--------|------|
| `extraction` | `ExtractionOutput` | `PipelineAgentDeps` |
| `extraction_reduce` | `ExtractionOutput` | `PipelineAgentDeps` |
| `agentic_generation` | `GenerationOutput` | `ToolGenerationDeps` |
| `reference_extraction` | `ReferenceExtractionOutput` | `PipelineAgentDeps` |
| `reference_resolver` | `ReferenceResolutionOutput` | `PipelineAgentDeps` |
| `memory_consolidation` | `str` | `MemoryConsolidationDeps` |
| `remediation` | `str` | `PipelineAgentDeps` |
| `agentic_remediation` | `str` | `RemediationDeps` |
| `verification` | `dict[str, Any]` | `PipelineAgentDeps` |
| `agentic_verification` | `str` | `VerificationDeps` |
| `wicer_probe` | `WicerProbeOutput` | `WicerProbeDeps` |
| `wicer_evaluate` | `WicerEvaluateOutput` | `WicerEvaluateDeps` |
| `wicer_diagnose` | `WicerDiagnoseOutput` | `WicerDiagnoseDeps` |

### Capture and organize

| Spec | Output | Deps |
|------|--------|------|
| `capture` | `CaptureOutput` | `PipelineAgentDeps` |
| `distill` | `DistillOutput` | `DistillDeps` |
| `agentic_distill` | `str` | `DistillAgentDeps` |
| `write` | `str` | `PipelineAgentDeps` |
| `agentic_write` | `str` | `WriteDeps` |
| `categorize` | `CategoryTree` | `CategorizeDeps` |

### Search and query

| Spec | Output | Deps |
|------|--------|------|
| `query` | `str` | `QueryDeps` |
| `agentic_query` | `str` | `QueryAgentDeps` |
| `query_expansion` | `QueryExpansionOutput` | `PipelineAgentDeps` |
| `crag_grade` | `CragGradeOutput` | `PipelineAgentDeps` |
| `hyde` | `str` | `PipelineAgentDeps` |
| `query_intent` | `QueryIntentDecision` | `QueryIntentDeps` |

### Decision sidecar

| Spec | Output | Deps |
|------|--------|------|
| `decision` | `ErrorDecision` | `DecisionDeps` |
| `adaptation` | `ErrorDecision` | `DecisionDeps` |
| `figure_relevance` | `FigureRelevanceDecision` | `FigureRelevanceDeps` |

`query_intent` is listed with the search specs above; all four run only when
`[decision] enabled = true`. See
[Decision model](../guides/llm-config.md#decision-model-optional-sidecar).

### Chunking

| Spec | Output | Deps |
|------|--------|------|
| `context_chunk` | `str` | `PipelineAgentDeps` |
| `agentic_context_chunk` | `str` | `ContextChunkDeps` |
| `summarizer` | `str` | `PipelineAgentDeps` |
| `agentic_summarizer` | `str` | `SummarizerDeps` |

### Output, diagnostics and vision

| Spec | Output | Deps |
|------|--------|------|
| `wiki_index` | `dict[str, Any]` | `PipelineAgentDeps` |
| `agentic_overview` | `str` | `OverviewDeps` |
| `doctor` | `str` | `DoctorDeps` |
| `vlm` | `str` | `VlmDeps` |
| `lint_rule` | `BaseModel \| str` | `PipelineAgentDeps` |

### KompendModel

`KompendModel` is a thin delegate that wraps any native pydantic-ai `Model`
and adds kompend-specific concerns:

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
| Change the output schema | **Output type** (`KompendAgentBase` generic parameter) |
| Change which model is used | **`default_model`** (one model per run; per-role overrides were removed) |

Skills are additive — they augment the system prompt without replacing
it. Templates define the core instruction; skills provide additional
context.

---

## Tools and capabilities

Agentic agents run a multi-turn loop over pydantic-ai tools. Two mechanisms:

- **Tools** — plain Python functions passed as `tools=` when the agent is
  built (`AgenticRemediationAgent` → `fetch_gaps`, `save_page`,
  `mark_unfixable`; `AgenticVerificationAgent` → `get_claim`,
  `search_sources`, `read_page`, `mark_verified`, …). The tool list lives in
  the matching `*_tools.py` module and must match the system prompt.
- **Capabilities** — shared cross-cutting behaviour supplied by
  `AgenticHarness.build_capabilities()` (sliding-window context compaction,
  oversized-message clamping, memory). Every `make_agentic_*` factory passes
  these in addition to `tools=`.

Attaching external **MCP servers** to an agent is still future work:
`KompendAgentBase._build_agent()` is where pydantic-ai tool/capability servers
would be registered, and pipeline callers would need no change.

---

## Writing a new agent

1. **Declare the spec** in `src/kompend/agents/<name>.py`:

```python
from typing import Final

from kompend.agents.base import AgentSpec
from kompend.agents.deps import PipelineAgentDeps
from kompend.compilation.schema import MyOutput  # or define a Pydantic model

SPEC: Final[AgentSpec] = AgentSpec(
    name="my_agent",
    result_type=MyOutput,
    deps_type=PipelineAgentDeps,
    template_name="my_prompt.md.j2",
    system_template_name="my_system.md.j2",
    template_prefix="compilation/",
)
```

2. **Add the typed facade** — the class callers may instantiate directly:

```python
from typing import ClassVar

from kompend.agents.base import AgentSpec, KompendAgentBase


class MyAgent(KompendAgentBase[MyOutput, PipelineAgentDeps]):
    """One-line description of what this agent does."""

    spec: ClassVar[AgentSpec | None] = SPEC
```

3. **Register the name** in `src/kompend/agents/specs.py`: add a member to
   `AgentName` and map it to the module in `_MODULES`. `get_spec("my_agent")`
   then resolves `SPEC`, and `all_specs()` includes it.

4. **Create the Jinja2 templates** in `src/kompend/templates/<prefix>/`:

   - `<name>_system.md.j2` — system prompt, with a `role:` in its frontmatter
   - `<name>_prompt.md.j2` — user prompt template (receives `deps`
     fields as Jinja2 variables)

   `template_prefix` must match the template's directory exactly — a mismatch
   fails at render time.

5. **Computed prompts go on the spec.** If a template cannot express the
   prompt, set `user_prompt` or `system_prompt` to a callable taking
   `(agent, deps)`; `install_hooks()` installs it on the materialised class.
   Output checks that must run after the model returns go in `validate_output`
   with the signature `(ctx, output)`.

6. **Wire it** with `build_agent("my_agent", model_string=...)` at the stage
   that needs it, or instantiate `MyAgent(...)` directly when the call site
   needs typed access. Add a `make_agentic_<name>_agent` factory only when the
   agent needs the harness's tools and capabilities.

7. **Export** the agent from `src/kompend/agents/__init__.py` if callers import
   it from the package. `tests/unit/test_agent_specs.py` checks that every
   registered spec materialises into a typed class with its hooks installed.

See the [LLM Configuration guide](../guides/llm-config.md) for model
strings, fallback chains, and reasoning-effort controls.
