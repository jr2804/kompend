# Migration Plan: `summarization-pydantic-ai` Integration

## Summary

Integrate `summarization-pydantic-ai` (v0.1.10) into KNOX's LLM call layer
to replace hand-rolled workarounds. `instructor` (v1.15.4) has **no
pydantic-ai adapter** — it patches raw OpenAI/Anthropic clients, not
`pydantic_ai.Agent`. JSON repair stays as a lightweight fallback.

## Pre-requisite: Format Consolidation (Completed)

Before integrating summarization capabilities, the LLM data format system was
consolidated to eliminate duplicated, scattered configuration:

- **`LlmDataFormat`** (StrEnum) is now the single source of truth for all
  structured data serialization: `JSON` (default), `TOON`, `ISON`, `MARKDOWN`
  (= `"text"`, human-readable pass-through), and legacy `TEXT` alias.
- **`QueryFormat`** is a true type alias (`type[LlmDataFormat] = LlmDataFormat`),
  not a separate class. Backward-compatible — `from knox.core.config import
  QueryFormat` still works.
- **`encode_for_llm()`** handles all four values: JSON→`json.dumps`, TOON→
  `_toon.encode`, ISON→`_ison.from_dict`, MARKDOWN/TEXT→pass-through.
- **`QueryConfig.format`** default changed from `TEXT` to `MARKDOWN`.
- **Documentation note**: These are fundamentally **LLM input formats** —
  compact encodings for sending data to the model. The `query.format` setting
  reuses them for structured CLI output because that output is often consumed
  by downstream agents (not humans), making token efficiency equally important.
- This enum is **orthogonal to** the forced JSON output format (`response_format:
  json_object`) that pydantic-ai enforces on LLM responses. The data formats
  here concern *how KNOX serializes input data for the LLM*, not *how the LLM
  returns output*.

| # | Current Workaround | Replaced By | Phase |
|---|-------------------|-------------|-------|
| 1 | `_compact_messages()` in `agents/knox_model.py` — drops all ModelResponse on retry | `ContextManagerCapability` (auto-compress on threshold) | 1 |
| 2 | No token budget awareness | `ContextManagerCapability` (token tracking + threshold) | 1 |
| 3 | `_repair_tool_call_args()` — `json_repair` before validator | **kept** (instructor no-op for pydantic-ai) | — |
| 4 | `_repair_text_part()` — `json_repair` on structured output text | **kept** (ditto) | — |
| 5 | No context-limit warnings | `ContextManagerCapability` (callback hooks) | 2 |

## Architecture

```text
Agent.run()
  └─ pydantic_ai.Agent
       ├── capabilities: [ContextManagerCapability]              ← NEW
       │    ├── Token tracking (UsageTracker integration)
       │    ├── Auto-compression at compress_threshold (90%)
       │    └── before_model_request: compress history if needed
       └── model: KnoxModel (fallback chain)
            ├── _repair_tool_call_args()   ← KEPT
            ├── _repair_text_part()        ← KEPT
            ├── [_compact_messages()]      ← REMOVED (Phase 2)
            ├── usage tracking, debug, thinking
            └── native pydantic-ai Model (ollama, openai, …)
```

## Capability Catalog

`summarization-pydantic-ai` v0.1.10 provides four capability classes:

| Capability | What It Does | Cost | Use Case |
|-----------|-------------|------|---------|
| `ContextManagerCapability` | Token tracking + auto-compress at threshold + tool output truncation | 1 LLM call per compression | **Primary pick** — unified, observable, comes with callbacks |
| `SummarizationCapability` | Summarize history at trigger count | 1 LLM call per summary | Lighter alternative when token tracking not needed |
| `SlidingWindowCapability` | Trim oldest messages at trigger count | Zero | Local models where summarization is too slow |
| `LimitWarnerCapability` | Inject warning as trailing user message when limits near | Zero | Agents with known iteration caps (WiCER) |

### `ContextManagerCapability` — Primary Integration

```python
ContextManagerCapability(
    max_tokens: int | None = None,          # auto-detect via genai-prices (200K fallback)
    compress_threshold: float = 0.9,        # compress when 90% of budget used
    keep: ContextSize = ("messages", 0),    # messages to keep after summary
    summarization_model: ModelType = "openai:gpt-4.1-mini",
    token_counter: TokenCounter,            # approximate counter (default)
    summary_prompt: str,                    # customizable
    max_tool_output_tokens: int | None = None,  # KNOX: irrelevant (no tools)
    tool_output_head_lines: int = 5,            # KNOX: irrelevant
    tool_output_tail_lines: int = 5,            # KNOX: irrelevant
    on_usage_update: UsageCallback | None = None,      # integrate with UsageTracker
    on_before_compress: BeforeCompressCallback | None, # observability
    on_after_compress: AfterCompressCallback | None,   # observability
    include_compact_tool: bool = False,    # KNOX: irrelevant (not chat agent)
)
```

Key methods (on `SummarizationProcessor`):

- `plan_compression(messages) → CompressionPlan | None` — decide if/when to compress (no LLM)
- `execute_plan(plan) → SummarizationResult` — run the summary LLM
- `process(messages) → SummarizationResult` — plan + execute in one call

### `ContextSize` Types

```python
ContextFraction = tuple[Literal["fraction"], float]   # ("fraction", 0.5)  → 50% of budget
ContextTokens   = tuple[Literal["tokens"], int]       # ("tokens", 4000)   → 4000 tokens
ContextMessages = tuple[Literal["messages"], int]     # ("messages", 50)   → 50 messages
```

### `CompressionPlan` & `SummarizationResult`

```python
@dataclass
class CompressionPlan:
    cutoff_index: int
    messages_to_summarize: list[ModelMessage]
    preserved_messages: list[ModelMessage]
    system_parts: list[SystemPromptPart]

@dataclass
class SummarizationResult:
    messages: list[ModelMessage]
    summarized: bool
    cutoff_index: int = 0
    summary: str | None = None
    skip_reason: SkipReason | None = None  # "not_triggered" | "cutoff_zero" | "failed"
```

## Phase 1 — `ContextManagerCapability` (replaces `_compact_messages` + adds token awareness)

### 1.1 Add capability config to `KnoxAgentBase`

```python
# src/knox/agents/base.py

from pydantic_ai_summarization import ContextManagerCapability, SlidingWindowCapability

class KnoxAgentBase[ResultT, DepsT](ABC):
    _context_capability: ContextManagerCapability | SlidingWindowCapability | None

    def __init__(self, ...,
                 context_manager: str | bool = "default",
                 ...):
        ...
        self._context_capability = _resolve_context_manager(
            context_manager,
            self._config,
            self._build_top_level_model(),  # use same failover chain for summarization
        )
```

Where `_resolve_context_manager` maps:

- `"default"` → `ContextManagerCapability(max_tokens=ctx_window, compress_threshold=0.9, summarization_model=primary_model)`
- `"aggressive"` → `ContextManagerCapability(max_tokens=ctx_window, compress_threshold=0.7, keep=("messages", 5), summarization_model=primary_model)`
- `"sliding"` → `SlidingWindowCapability(trigger=("messages", 100), keep=("messages", 50))`
- `False` / `None` → no capability
- `True` → default preset

### 1.2 Attach capability to pydantic-ai Agent

In `_build_agent()`:

```python
def _build_agent(self) -> Agent:
    ...
    capabilities = []
    if self._context_capability:
        capabilities.append(self._context_capability)

    agent = Agent(
        self._build_top_level_model(),
        deps_type=self.deps_type,
        output_type=output_type,
        model_settings=model_settings,
        retries=self._retries,
        capabilities=capabilities or None,  # NEW
    )
```

### 1.3 UsageTracker Integration

`ContextManagerCapability.on_usage_update` fires on each token usage update.
Wire it to KNOX's `UsageTracker`:

```python
def _on_usage_update(pct: float, current: int, max_tokens: int) -> None:
    """Called by ContextManagerCapability when token usage changes."""
    if hasattr(thread_local, 'usage_tracker'):
        thread_local.usage_tracker.record_context_usage(current, max_tokens)

capability = ContextManagerCapability(
    ...,
    on_usage_update=_on_usage_update,
    on_before_compress=_on_before_compress,  # log compression events
    on_after_compress=_on_after_compress,    # log summary quality
)
```

### 1.4 Configuration wiring

Add to `compile.llm` config section:

```yaml
# default_config.yml
compile:
  llm:
    context_manager: "default"   # "default" | "aggressive" | "sliding" | false
    context_window_cap: 32768    # token budget (0 = auto-detect, reuses existing key)
```

### 1.5 Test surface

- **Unit**: `KnoxAgentBase` builds Agent with `ContextManagerCapability`
- **Integration**: Run WiCER 8-iteration loop with tight budget, verify compression fires
- **Regression**: All existing agent tests pass with `context_manager="default"`
- **Edge cases**: Compression fails → `SummarizationResult(summarized=False, skip_reason="failed")` → history preserved unchanged

### Risk assessment

| Risk | Mitigation |
|------|-----------|
| Compression LLM call costs extra tokens | Only fires at 90% budget threshold — rare in typical KNOX runs (2-3 turns). Configurable to "sliding" (zero cost) or off. |
| Summarization model mismatch | Pass `self._build_top_level_model()` as `summarization_model` — same failover chain. |
| Compression fails (model error) | `SummarizationResult.summarized=False` → original history preserved, no data loss. |
| Auto-detected token budget wrong | Configurable via `context_window_cap` config key. |
| `genai-prices` dependency | Only used for auto-detection when `max_tokens=None`. Can hardcode budget per model if dependency unwanted. |
| pydantic-ai capability API changes | Pin `summarization-pydantic-ai>=0.1.10,<0.2` |

## Phase 2 — Remove `_compact_messages`

Once Phase 1 is stable:

1. Remove `_compact_messages()` from `agents/knox_model.py`
2. Remove the call at line 99: `compacted = _compact_messages(messages)`
3. `ContextManagerCapability.before_model_request` handles history compression at
   the pydantic-ai level, before KnoxModel sees the messages.

The summarization approach is **strictly better**: instead of dropping all
ModelResponse messages (losing context), it synthesizes a summary that
preserves key information. When the compression model fails, history is
preserved unchanged — unlike `_compact_messages` which always truncates.

## Phase 3 — Observability Callbacks

Wire compression events into KNOX's logging and runlog:

```python
def _on_before_compress(messages: list, cutoff_index: int) -> None:
    logger.info("Compressing %d messages at cutoff %d", len(messages), cutoff_index)

def _on_after_compress(
    messages: list, summarized: bool, summary: str | None
) -> str | None:
    if summarized:
        logger.info("Compression produced %d-char summary", len(summary or ""))
    else:
        logger.warning("Compression skipped (history preserved)")
    return None  # don't modify the summary
```

## Phase 4 — Per-Agent Tuning

Specific agents get tailored context management:

### WiCER (8 iterations with multiple LLM calls each)

```python
class WicerProbeAgent(KnoxAgentBase[...]):
    def __init__(self, ...):
        super().__init__(
            ...,
            context_manager="aggressive",  # 70% threshold, keeps 5 messages
        )
```

### Extraction (large documents via message_history threading)

Message_history threading between batches means context grows across
batch boundaries. `ContextManagerCapability` processes the full history
before each model request, so this is handled automatically — the
compression is orthogonal to the batching loop.

## What `instructor` Does NOT Do

`instructor` v1.15.4 has these providers:

```text
from_openai, from_anthropic, from_genai, from_gemini, from_cohere,
from_perplexity, from_xai, from_groq, from_mistral, from_fireworks,
from_cerebras, from_writer, from_bedrock, from_vertexai, from_litellm,
from_openrouter, from_databricks, from_deepseek, from_together,
from_anyscale
```

No `from_pydantic_ai`. The library works by patching
`client.chat.completions.create` on raw provider clients — it doesn't
integrate with `pydantic_ai.Agent`.

### JSON repair stays

The `salvage_json` / `json_repair` path in
`KnoxModel._repair_tool_call_args()` and `_repair_text_part()` remains —
it's the only defense against malformed JSON from small local models.
Lightweight (stdlib `json.loads` fast path first, `json_repair` only on
failure) and eliminates ~67% of generation retries.

If a future `instructor` release adds pydantic-ai support, we can revisit.

## KNOX-Specific Notes

- KNOX agents are **tool-free**: they're JSON-in/JSON-out extractors.
  `max_tool_output_tokens`, `tool_output_head_lines`, `tool_output_tail_lines`,
  and `include_compact_tool` on `ContextManagerCapability` are irrelevant.
- KNOX uses fallback chains (`KnoxModel` wrapping multiple pydantic-ai models).
  `summarization_model` accepts `str | Model | KnownModelName`, so we pass
  the same fallback chain.
- `CompressionPlan.plan_compression()` is a pure decision step (no network).
  We can call it from `on_before_compress` to inspect whether compression
  will fire and log the cutoff index.

## Appendix: `pydantic_ai.MemoryTool` (Not Applicable)

`pydantic_ai` v2.13 ships `MemoryTool` — a native tool that exposes the
model's built-in memory system (Anthropic's `computer_use` memory).

**Why it doesn't help KNOX:**

1. **Anthropic-only.** `MemoryTool` is a thin wrapper registering `kind="memory"`
   on `AbstractNativeTool`. It passes through to the model's native memory
   — which only Anthropic supports. KNOX supports Ollama, OpenAI, Google,
   DeepSeek, Mistral, Groq, Cohere, Bedrock, and custom OpenAI-compatible API
   servers. One-provider-only feature is not useful.

2. **Agent-loop memory, not compile memory.** `MemoryTool` gives the agent
   a way to read/write persistent key-value pairs across conversation turns.
   KNOX's agents are single-shot extractors (one call per batch), not
   multi-turn conversational agents. There's no conversation loop to persist
   across.

3. **KNOX already has its own memory system.** `knox.memory` provides:
   - `MemoryStore`: query/answer storage with semantic retrieval (zvec-backed)
   - Memory lifecycle: confidence scoring, tier promotion, decay
   - MCP tools: `memory_add`, `memory_search`, `memory_synthesize`
   - Compile memory: tracks superseded/stale wiki nodes

   `MemoryTool` is orthogonal to all of this — it's about per-agent-turn
   recall across user interactions, which KNOX agents don't have.

**Verdict: Skip.** Zero benefit for KNOX. Not even optional.

## Implementation Order

1. **Add `ContextManagerCapability` to `KnoxAgentBase`** — capability active, `_compact_messages` still runs
2. **Add config wiring** — `compile.llm.context_manager` and `context_window_cap`
3. **Add callback wiring** — `on_usage_update` → UsageTracker
4. **Integration test** — WiCER 8-iteration with tight budget, verify compression
5. **Remove `_compact_messages`** — capability handles it
6. **Per-agent tuning** — WiCER "aggressive", others "default"
7. **Observability** — log compression events via callbacks

## Estimates

| Phase | Files touched | Net new LOC | Risk |
|-------|--------------|-------------|------|
| 1 — ContextManagerCapability | 3 (base.py, deps.py, config) | ~80 | Low |
| 2 — Remove _compact_messages | 1 (knox_model.py) | -15 | Low |
| 3 — Observability callbacks | 1 (base.py) | ~25 | Low |
| 4 — Per-agent tuning | 2 (wicer.py, extraction.py) | ~10 | Low |
| **Total** | **5** | **~100** | **Low** |
