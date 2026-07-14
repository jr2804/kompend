# PydanticAI agent migration: all remaining LLMCall paths

**Status: COMPLETED** — `src/knox/llm/client.py` has been deleted.

**Net change:** ~800 lines of new agent code, ~600 lines of old LLMClient code removed.
The `_build_agent()` base class refactoring eliminated 9 duplicate methods (−100 lines).

**Goal:** Remove `src/knox/llm/client.py` by migrating every consumer to a PydanticAI agent.

## Current state

14 files still import from `LLMClient`. The compilation pipeline agents are already migrated; everything else still routes through the old client.

| Status | Path | Agent |
|---|---|---|
| ✅ | Compilation (extract, generate, wicer probe/eval/diagnose) | `ExtractionAgent`, `GenerationAgent`, `WicerProbe/Eval/DiagnoseAgent` |
| ✅ | Query synthesis | `QueryAgent` |
| ✅ | Doctor | `DoctorAgent` |
| ❌ | Query expansion / CRAG / HyDE | `search/retriever.py`, `search/pipeline.py` |
| ❌ | Capture | `capture/capture.py` |
| ❌ | Distill | `capture/distill.py` |
| ❌ | Organize categorize | `organize/categorize.py`, `organize/engine.py` |
| ❌ | Lint LLM rules | `lint/llm_rules.py` |
| ❌ | Contextual chunking | `chunking/contextual.py` |
| ❌ | Summarizer | `chunking/summarizer.py` |
| ❌ | VLM enricher | `resolution/vlm_enricher.py` |
| ❌ | Verification | `verification.py` |
| ❌ | Wiki index | `output/wiki_index.py` |
| ❌ | Conversational agent tools | `agent/conversational.py` |
| ❌ | ACP agent LLM calls | `agent/knox_agent.py` |
| ❌ | LLMClient constructors | `core/pipeline/orchestrator.py`, `mcp/server.py`, `cli/cmds/*.py` |

## Phase 0 — Prerequisites

Three `LLMClient` services must be moved into the agent layer before any consumer can drop the old class.

### 0a. `_discover_context_window` → standalone module

**Current:** `LLMClient.__post_init__` calls `_discover_context_window()` to detect model context limits. `doctor.py` and `two_phase.py` import it independently already.

**Target:** Move to `src/knox/llm/context_window.py`. The function stays unchanged; only its import path changes. The cap, fallback limits, and Ollama/Ollama/models.dev discovery logic remain identical.

**Files:** Move function (lines 223-377 of `client.py`), update imports in `doctor.py`, `two_phase.py`. Keep the re-export from `client.py` until deletion.

**Risk:** None — function is already used independently. Pure move.

### 0b. `UsageTracker` → integrate into `LitellmModel._fn`

**Current:** `LLMClient.usage` is a `UsageTracker` instance. After every `litellm.completion()`, `LLMClient.complete()` calls `self.usage.record(...)`. Pipeline code reads `client.usage.to_dict()` after compilation.

**Target:**

- `LitellmModel._fn` extracts `response.usage` (prompt_tokens, completion_tokens, model) and stores it via a shared `UsageTracker`
- `KnoxAgentBase` exposes a `usage_tracker` property that aggregates across calls
- Pipeline reads `agent.usage_tracker.to_dict()` instead of `client.usage.to_dict()`

**The tricky bit:** `LitellmModel._fn` is a bare function, not a bound method, so it can't access `self`. The tracker must be passed as part of the model's closure at construction time — done via `functools.partial` or a closure in `LitellmModel.__init__`.

**Risk:** Medium — if the tracker isn't wired correctly, token counts silently drop to zero. Add a unit test that verifies usage is populated after `agent.run_sync()`.

### 0c. `FailoverChain` → integrate into `LitellmModel._fn`

**Current:** `LLMClient.complete()` wraps litellm in a `FailoverChain` — try model A, catch rate-limit / timeout / service-unavailable, retry with model B, etc. `LLMClient.__init__` builds the chain from `default_model + fallback_chain + role_models`.

**Target:**

- `LitellmModel.__init__` accepts `fallback_models: list[str]`
- `LitellmModel._fn` catches `litellm.APIError` / `RateLimitError` / `ServiceUnavailableError` and retries with the next model in the chain
- `KnoxAgentBase.__init__` gains a `fallback_models` parameter, passes it to `LitellmModel`
- The fallback chain wraps the same `_fn` — same prompt, same schema, different model

**Design sketch:**

```python
class LitellmModel(FunctionModel):
    def __init__(self, model: str, *, fallback_models: list[str] | None = None, ...):
        self._models = [model] + (fallback_models or [])
        self._usage_tracker = UsageTracker()
        super().__init__(self._fn, ...)

    def _fn(self, messages, info):
        for model in self._models:
            try:
                response = litellm.completion(model=model, messages=llm_messages, ...)
                self._usage_tracker.record(...)
                return response.choices[0].message.content
            except (litellm.APIError, litellm.RateLimitError, litellm.ServiceUnavailableError) as exc:
                if model is self._models[-1]:
                    raise
                logger.warning("Model %s failed: %s — trying %s", model, exc, ...)
```

**Risk:** Medium — failover is exercised only when the primary fails, so it's hard to test automatically without a mock failing model. Add a unit test that mocks `litellm.completion` to raise on the first call and succeed on the second.

---

**After Phase 0:** `KnoxAgentBase` + `LitellmModel` have feature parity with `LLMClient` for all consumer use cases. Consumer migration can begin without losing failover or token tracking.

**Dependency:** All Phase 1-4 migrations depend on Phase 0. Without failover, migrated consumers lose fallback capability. Without usage tracking, the compile summary shows zero tokens.

---

## Phase 1 — Simple structured-output migrations (P2)

Each follows the established pattern: define an agent class, wire a template, replace one `LLMClient.complete()` / `.try_complete()` call.

### 1a. Capture + Distill

| Call site | `capture/capture.py:78` | `capture/distill.py:67` |
|---|---|---|
| Current | `llm_client.complete(messages, JSON_RESPONSE_FORMAT)` → `_parse_llm_json()` | Same pattern |
| Output schema | `CapturedNote(title, body, tags, entities, attachments)` | `DistillNote(summary, key_insights, sources)` |
| Agent | `CaptureAgent` | `DistillAgent` |
| Deps type | `PipelineAgentDeps` (text=user_note) | `PipelineAgentDeps` (text=captured_notes) |
| Template | `capture/capture_system.md.j2` | `capture/distill_system.md.j2` |
| Template variables | `{text}` | `{text}` |
| Effort | Small — 1 call site each | Same |

**CLI construction:** `cli/cmds/capture.py` instantiates an `LLMClient` for the VLM model (image analysis) and passes it as a parameter. Agent construction moves into the capture command.

### 1b. Organize (categorize)

| Call site | `organize/categorize.py:142` |
|---|---|
| Current | `llm_client.complete(categorize_prompt, JSON_RESPONSE_FORMAT)` → `CategoryTree` |
| Output schema | `CategoryTree(name, children, files)` |
| Agent | `CategorizeAgent` |
| Deps type | `CategorizeDeps(text, files_json, source_id)` |
| Template | `organize/categorize_system.md.j2` + `organize/categorize_user.md.j2` |
| Template variables | `{files_json}`, `{max_depth}`, `{min_subcategories}` |
| Effort | Medium — `organize/engine.py` also constructs an `LLMClient` for the script prompt |

### 1c. Lint (LLM rules)

| Call site | `lint/llm_rules.py:45` |
|---|---|
| Current | `llm_client.complete(messages, JSON_RESPONSE_FORMAT)` → per-rule schema |
| Output schema | Rule-specific dict |
| Agent | `LintRuleAgent` |
| Deps type | `PipelineAgentDeps` (text=chunk) |
| Effort | Single LLM call covering multiple rules at once |

### 1d. Query pipeline (expansion, CRAG, HyDE)

Three LLM calls inside `AISearchPipeline` in `search/pipeline.py`, plus one in `search/retriever.py`.

| Call | File:line | Current | Output |
|---|---|---|---|
| Multi-query expansion | `pipeline.py:167` | `llm_client.complete(..., JSON_RESPONSE_FORMAT)` | `list[str]` |
| CRAG grading | `pipeline.py:195` | `llm_client.complete(..., JSON_RESPONSE_FORMAT)` | `list[float]` (relevance scores) |
| HyDE | `pipeline.py:287` | `llm_client.try_complete(...)` | `str` (hypothetical answer) |
| CRAG (retriever) | `retriever.py:107` | `llm_client.complete(..., JSON_RESPONSE_FORMAT)` | `list[float]` |

| Agents | `QueryExpansionAgent`, `CragGradeAgent`, `HydeAgent` |
|---|---|
| Deps types | `PipelineAgentDeps` (expansion, HyDE), `CragDeps(text, chunks_json)` (CRAG) |
| Effort | Medium — `AISearchPipeline` currently holds one `_llm_client`. Needs to hold three agents instead (or one agent with tool dispatch). |

### 1e. Wiki index

| Call site | `output/wiki_index.py:466` |
|---|---|
| Current | `LLMClient(default_model=...).complete(..., JSON_RESPONSE_FORMAT)` |
| Output schema | Index structure dict |
| Agent | `WikiIndexAgent` |
| Deps type | `PipelineAgentDeps` |
| Effort | Small — constructs its own temporary `LLMClient` inline |

### 1f. Verification

| Call site | `verification.py:155` |
|---|---|
| Current | `llm_client.complete(..., JSON_RESPONSE_FORMAT)` → `ClaimVerification` |
| Output schema | `list[ClaimVerification]` |
| Agent | `VerificationAgent` |
| Deps type | `VerificationDeps(wiki_text, claims_json)` |
| Effort | Small — 1 call site |

---

## Phase 2 — Special-case migrations (P3)

### 2a. Contextual chunking

| Call site | `chunking/contextual.py:120` |
|---|---|
| Current | `llm_client.complete(messages, JSON_RESPONSE_FORMAT)` |
| Output | Free-text context paragraph (~1 sentence) wrapped in `{"context": "..."}` JSON |
| Agent | `ContextChunkAgent` → `str` |
| Deps type | `PipelineAgentDeps` |
| Effort | Small — PydanticAI free-text output handles this cleanly |

### 2b. Summarizer

| Call site | `chunking/summarizer.py:90` |
|---|---|
| Current | `llm_client.complete(messages, JSON_RESPONSE_FORMAT)` |
| Output | Free-text summary wrapped in JSON |
| Agent | `SummarizerAgent` → `str` |
| Deps type | `PipelineAgentDeps` |
| Effort | Same pattern as contextual chunking |

### 2c. VLM enricher

| Call site | `resolution/vlm_enricher.py:85` |
|---|---|
| Current | `llm_client.complete(messages)` — messages contain `content=[{"type": "image_url", "image_url": {"url": ...}}]` |
| Output | Free-text caption |
| Agent | `VlmAgent(ImageDeps)` → `str` |
| Deps type | `ImageDeps(text: str, image_url: str, source_id: str, timeout: float)` |
| PydanticAI issue | PydanticAI's `agent.run_sync()` accepts only text prompts natively. To pass images: (A) inject the image URL into the message list before calling `litellm.completion()` inside `_fn`, or (B) serialize the image as base64 and include in the prompt text. Option A is cleaner — `LitellmModel._fn` has access to all messages, so the deps can carry the image URL and `run()` injects it as a second message alongside the text. |
| Risk | Medium — images are the one input modality PydanticAI doesn't abstract. Must be handled at the `_fn` level. |

### 2d. Conversational agent tool loop

| Call site | `agent/conversational.py:45` |
|---|---|
| Current | Manual loop: `LLMClient.complete_with_tools(messages, tools)` → parse tool call → execute → repeat until `max_turns` |
| Target | PydanticAI native `@agent.tool` decorators + `agent.run_sync()` |
| PydanticAI pattern | `agent.tool(tool_func)` registers callable tools. `run_sync()` returns when the model produces a final response. Iterations handled internally. |
| Tool wiring | The `ToolDispatcher` must be adapted to PydanticAI's `@agent.tool` protocol |
| Effort | **Complex.** The current conversational agent has a manual loop with tool dispatch, state accumulation, and an ACP protocol wrapper. |
| **Recommendation** | Migrate last. New code, not a regression risk, most architecturally different pattern. |

---

## Phase 3 — Remove LLMClient from construction sites

After all LLM calls are agent-based, `LLMClient` only survives in construction code:

- `core/pipeline/orchestrator.py:959` — `build_llm_client()` returns an `LLMClient` for the pipeline
- `core/pipeline/orchestrator.py:1090` — `vlm_client = LLMClient(...)` construction
- `mcp/server.py:39` — imports `LLMClient` for MCP tool invocations
- `cli/cmds/lint.py`, `cli/cmds/organize.py`, `cli/cmds/query.py`, `cli/cmds/capture.py` — construct `LLMClient` from config

Each must be replaced with the corresponding agent constructor or, for infrastructure code that just needs a quick LLM check, a bare `litellm.completion()` call inside the CLI module.

## Phase 4 — Final deletion

**Final act.** Delete `src/knox/llm/client.py`:

- Remove from `src/knox/llm/__init__.py` exports
- Remove `JSON_RESPONSE_FORMAT`, `_strip_reasoning` re-exports (already available from `format.py` and `litellm_model.py`)
- Remove `USAGE_REPORT_FILTER` and other constants that moved elsewhere
- Update all remaining imports across the 14 consumer files
- Verify all tests pass

## Implementation ordering (recommended)

```text
Phase 0 — Prerequisites (must land before anything else)
  ├─ 0a. Move _discover_context_window → standalone module
  ├─ 0b. Integrate UsageTracker into LitellmModel
  └─ 0c. Integrate FailoverChain into LitellmModel

Phase 1 — Simple structured-output migrations (P2, can be parallelized)
  ├─ 1a. CaptureAgent + DistillAgent
  ├─ 1b. CategorizeAgent
  ├─ 1c. LintRuleAgent
  ├─ 1d. QueryExpansion/CragGrade/HydeAgent
  ├─ 1e. WikiIndexAgent
  └─ 1f. VerificationAgent

Phase 2 — Special cases (P3, can be parallelized)
  ├─ 2a. ContextChunkAgent
  ├─ 2b. SummarizerAgent
  ├─ 2c. VLM enricher agent
  └─ 2d. Conversational agent tools

Phase 3 — Construction cleanup
  └─ 3a-3d. Replace LLMClient(...) construction sites

Phase 4 — Final deletion
  └─ 4. Delete src/knox/llm/client.py
```

## Migration patterns (one-line reference)

> **Note:** `core/` was split into nested submodules (pipeline/, cache/, models/, templates/, etc.)
> after this plan was drafted. All consumer files updated imports but retained their logic.
> File paths in this plan reflect the new layout.

| Before | After |
|---|---|
| `LLMClient.complete(msgs, JSON_RESPONSE_FORMAT)` | `Agent(PipelineAgentDeps(text=...)).run(deps)` |
| `llm_client.try_complete(msgs, response_model=X)` | `Agent(deps).run()` — PydanticAI handles validation + retry |
| `LLMClient.complete_with_tools(...)` | PydanticAI `@agent.tool` decorators |
| `LLMClient(..., context_window=...).complete(...)` (VLM) | `_fn` injects image URL as second message alongside text |
| `LLMClient.from_config(...)` | Agent factory or bare `litellm.completion()` |
| `llm_client.usage.to_dict()` | `agent.usage_tracker.to_dict()` |
| `llm_client.context_window` | `_discover_context_window(model, cap)` standalone |
| `_discover_context_window` import from `client` | Import from `knox.llm.context_window` |
| `JSON_RESPONSE_FORMAT` import from `client` | Import from `knox.llm.format` (already available) |
