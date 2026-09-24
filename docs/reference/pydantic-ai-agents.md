# PydanticAI Agents & Skills

Every LLM call in KNOX routes through a **PydanticAI agent** in
`knox.agents.*`. Agents come in two shapes:

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

### Base class

All agents inherit from `KnoxAgentBase[ResultT, DepsT]` — a KNOX wrapper around
`pydantic_ai.Agent` that adds:

- **Model bridge** — builds a [`KnoxModel`](#knoxmodel) from a
  `provider/model` string.
- **Typed result and deps** — the two generic parameters supply the output
  schema (`ResultT`) and the run context dataclass (`DepsT`). They are resolved
  at construction and handed to pydantic-ai as the agent's `output_type`; there
  is no separate `result_type` class attribute to set.
- **Template-driven prompts** — system and user prompts are Jinja2
  templates loaded by `TemplateLoader`. No prompt text lives in Python.
- **Skill injection** — user-defined skills are appended to the system
  prompt at render time.
- **Output validation + retry** — PydanticAI validates the LLM response
  against the output type; on mismatch, it sends actionable feedback and
  retries up to `output_retries` times.
- **Usage tracking** — token usage is recorded per call for
  observability.
- **Reasoning effort** — maps `reasoning_effort` config to the
  provider's thinking controls (e.g. `think=False` for Ollama).
- **Tools and capabilities** — agentic subclasses pass `tools=` and
  `AgenticHarness.build_capabilities()` to get a multi-turn tool loop.

### How agents are wired into the pipeline

Agents are constructed at the point of use, not via a global registry. Some are
built directly by the stage that needs them; the agentic ones are built by a
`make_agentic_*` factory that also attaches their tools and harness, and that
factory is called by the CLI/MCP layer. Example (single-shot extraction):

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

A class file existing is **not** evidence that it is wired: several agents are
implemented but have no production construction site. The inventory below
marks this explicitly, and wiring status is tracked in the epic `knox-0iv2`.

---

## Agent inventory

> `Output` / `Deps` are the `KnoxAgentBase` generic parameters. `Template role`
> is the `role:` value declared in the agent's prompt frontmatter. `Status` is
> from a call-site scan: **wired** = a production construction site exists,
> **not wired** = class/factory exists but production never constructs it,
> **dead** = unreachable.

### Agentic agents (multi-turn, tool-using)

| Agent | Stage | Output | Deps | Template role | Status |
|-------|-------|--------|------|---------------|--------|
| `AgenticGenerationAgent` | Compilation T4 (page synthesis) | `GenerationOutput` | `ToolGenerationDeps` | `compile.generate` | wired |
| `AgenticRemediationAgent` | Lint (remediation) | `str` | `RemediationDeps` | `lint.remediate-agentic` | wired |
| `AgenticVerificationAgent` | Trust (claim verification) | `str` | `VerificationDeps` | `lint.verify-agentic` | wired |
| `AgenticOverviewAgent` | Output (overview synthesis) | `str` | `OverviewDeps` | `output.synthesize-agentic` | wired |
| `AgenticWriteAgent` | Capture (`write`) | `str` | `WriteDeps` | `capture.write-agentic` | wired |
| `AgenticDistillAgent` | Capture (`distill`) | `str` | `DistillAgentDeps` | `capture.distill.agentic` | wired |
| `AgenticQueryAgent` | Query synthesis | `str` | `QueryAgentDeps` | `search.query.agentic` | wired |
| `AgenticSummarizerAgent` | Chunking (internal-node summaries) | `str` | `SummarizerDeps` | — | not wired |
| `AgenticContextChunkAgent` | Chunking (contextual retrieval) | `str` | `ContextChunkDeps` | — | not wired |

### Single-shot agents

| Agent | Stage | Output | Deps | Template role | Status |
|-------|-------|--------|------|---------------|--------|
| `ExtractionAgent` | Compilation T3 (extraction) | `ExtractionOutput` | `PipelineAgentDeps` | `compile.extract` | wired |
| `ExtractionReduceAgent` | Compilation T3 (map-reduce) | `ExtractionOutput` | `PipelineAgentDeps` | `compile.extract.reduce` | wired |
| `ReferenceExtractionAgent` | Compilation (references) | `ReferenceExtractionOutput` | `PipelineAgentDeps` | `compile.references` | wired |
| `ReferenceResolverAgent` | References (resolve) | `ReferenceResolutionOutput` | `PipelineAgentDeps` | `compile.references` | wired |
| `WicerProbeAgent` | WiCER probe | `WicerProbeOutput` | `WicerProbeDeps` | `wicer.probe` | wired |
| `WicerEvaluateAgent` | WiCER judge | `WicerEvaluateOutput` | `WicerEvaluateDeps` | `wicer.judge` | wired |
| `WicerDiagnoseAgent` | WiCER refine | `WicerDiagnoseOutput` | `WicerDiagnoseDeps` | `wicer.refine` | wired |
| `QueryAgent` | Query synthesis (non-agentic path) | `str` | `QueryDeps` | — | wired |
| `QueryExpansionAgent` | Search (query expansion) | `QueryExpansionOutput` | `PipelineAgentDeps` | `search.expand` | wired |
| `CragGradeAgent` | Search (CRAG grading) | `CragGradeOutput` | `PipelineAgentDeps` | — | wired |
| `HydeAgent` | Search (HyDE) | `str` | `PipelineAgentDeps` | — | wired |
| `VlmAgent` | Resolution (figure captioning) | `str` | `VlmDeps` | — | wired |
| `CaptureAgent` | Capture | `CaptureOutput` | `PipelineAgentDeps` | `capture` | wired |
| `DistillAgent` | Capture (distill, single-shot) | `DistillOutput` | `DistillDeps` | `capture.distill` | wired |
| `CategorizeAgent` | Organize | `CategoryTree` | `CategorizeDeps` | `organize.categorize` | wired |
| `LintRuleAgent` | Lint rules | `BaseModel \| str` | `PipelineAgentDeps` | `lint.*` | wired |
| `DoctorAgent` | Doctor diagnostics | `str` | `DoctorDeps` | — | wired |
| `WriteAgent` | Capture (`write`, single-shot) | `str` | `PipelineAgentDeps` | — | wired |

### Implemented but not wired

These classes exist and are accepted by their consuming component, but no
production path constructs them. Do not describe them as live pipeline stages.

| Agent | Stage | Output | Deps | Notes |
|-------|-------|--------|------|-------|
| `RemediationAgent` | Lint (remediation) | `str` | `PipelineAgentDeps` | Superseded by `AgenticRemediationAgent`; only the agentic factory has callers. |
| `VerificationAgent` | Trust (verification) | `dict[str, Any]` | `PipelineAgentDeps` | Superseded by `AgenticVerificationAgent`. |
| `SummarizerAgent` | Chunking (summaries) | `str` | `PipelineAgentDeps` | `NodeSummarizer` accepts it, but `build_pipeline` constructs `NodeSummarizer` without an agent. |
| `ContextChunkAgent` | Chunking (contextual retrieval) | `str` | `PipelineAgentDeps` | `ContextualEnricher` accepts it, but is constructed only in tests. |
| `AgenticSummarizerAgent` | Chunking (summaries) | `str` | `SummarizerDeps` | Factory `make_agentic_summarizer_agent()` has no caller. |
| `AgenticContextChunkAgent` | Chunking (contextual retrieval) | `str` | `ContextChunkDeps` | Factory `make_agentic_context_chunk_agent()` has no caller. |

### Dead (superseded, unreachable)

| Agent | Output | Deps | Notes |
|-------|--------|------|-------|
| `WikiIndexAgent` | `dict[str, Any]` | `PipelineAgentDeps` | No construction site; superseded by `AgenticOverviewAgent`. |

The legacy template `GenerationAgent` (and its `generate_all_pages()` loop) were
removed in `knox-0iv2.12`.

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
| Change the output schema | **Output type** (`KnoxAgentBase` generic parameter) |
| Change which model is used | **`default_model`** (per-role overrides are not yet applied) |

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
`KnoxAgentBase._build_agent()` is where pydantic-ai tool/capability servers
would be registered, and pipeline callers would need no change.

---

## Writing a new agent

1. **Create the agent class** in `src/knox/agents/<name>.py`. The result and
   deps types are the generic parameters — do **not** declare a `result_type`
   attribute:

```python
from typing import ClassVar

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import PipelineAgentDeps
from knox.compilation.schema import MyOutput  # or define a Pydantic model


class MyAgent(KnoxAgentBase[MyOutput, PipelineAgentDeps]):
    """One-line description of what this agent does."""

    template_prefix: ClassVar[str] = "compilation/"
    template_name: ClassVar[str] = "my_prompt.md.j2"
    system_template_name: ClassVar[str] = "my_system.md.j2"
```

2. **Create the Jinja2 templates** in `src/knox/templates/<prefix>/`:

   - `<name>_system.md.j2` — system prompt, with a `role:` in its frontmatter
   - `<name>_prompt.md.j2` — user prompt template (receives `deps`
     fields as Jinja2 variables)

   `template_prefix` must match the template's directory exactly — a mismatch
   fails at render time (this has bitten agentic agents whose templates moved
   between directories).

3. **Wire it** at the point of use (the stage that needs it), or add a
   `make_agentic_<name>_agent` factory next to the class and call it from the
   CLI/MCP layer. There is no central registry.

4. **Export** the agent from `src/knox/agents/__init__.py` if callers import it
   from the package.

See the [LLM Configuration guide](../guides/llm-config.md) for model
strings, fallback chains, and reasoning-effort controls.
