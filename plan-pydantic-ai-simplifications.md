# Pydantic-AI Documentation Review: Simplification Opportunities

Crawled the full pydantic-ai docs (overview, retries, ACP, message-history,
tools, toolsets, models, multi-agent) and compared against the KNOX codebase.
Seven concrete simplification targets found, ranked by impact.

---

## T1 — Replace KnoxLiteLLMModel failover with `FallbackModel` (HIGH)

**Current:** `litellm_model.py` `request()` implements a hand-rolled failover
loop over `[primary] + fallback_models`, catching `_TRANSIENT_ERRORS`
(Timeout, APIConnectionError, RateLimitError, ServiceUnavailableError) and
retrying each model in turn. ~40 lines of orchestration.

**Pydantic-AI built-in:** `pydantic_ai.models.fallback.FallbackModel` does
exactly this — tries models in sequence, falls back on exceptions or
response-based handlers. Default fallback is `ModelAPIError` (4xx/5xx).
Configurable via `fallback_on=[ExceptionType, exc_handler, response_handler]`.
Per-model settings supported. Raises `FallbackExceptionGroup` (an
`ExceptionGroup` subclass) if all fail.

**Why it's better:**

- Battle-tested, documented, maintained upstream
- Response-based fallback (e.g. truncate/length finish reason) — KNOX currently
  has `_log_response_anomalies` detecting exactly this but only logging it
- Proper exception grouping via `FallbackExceptionGroup` instead of our
  last-error re-raise
- Composes with `ConcurrencyLimitedModel`, HTTP retry transports, etc.

**Risk:** The `_TRANSIENT_ERRORS` tuple maps cleanly to the `ModelAPIError`
hierarchy. The one custom aspect — catching `litellm.Timeout` specifically —
needs verification that pydantic-ai's litellm bridge surfaces these as
`ModelAPIError` subclasses (it should, since the bridge wraps litellm exceptions
in `ModelHTTPError`).

**Net diff:** ~40 lines deleted from `litellm_model.py`. The `request()`
override goes away entirely — `FallbackModel(KnoxLiteLLMModel(primary),
*fallback_models)` is constructed in `_build_agent()` instead.

---

## T2 — Replace manual HTTP retries with `AsyncTenacityTransport` (HIGH)

**Current:** No HTTP-layer retry. The failover chain (T1) retries whole models,
not individual HTTP requests. Rate limits (429) and transient 5xx are not
retried at the transport level.

**Pydantic-AI built-in:** `pydantic_ai.retries.AsyncTenacityTransport` wraps
an httpx `AsyncClient` with tenacity-based retries. `wait_retry_after`
automatically respects `Retry-After` headers from 429 responses and falls back
to exponential backoff. Configure via `RetryConfig(retry=...,
wait=wait_retry_after(fallback_strategy=wait_exponential(...), max_wait=300),
stop=stop_after_attempt(5))`.

**Why it's better:**

- Handles transient failures *before* escalating to model-level failover
- Respects `Retry-After` headers (KNOX doesn't)
- Exponential backoff with configurable caps
- Logs every retry automatically (instrumented via httpx)
- Separates concerns: HTTP retries (transport) vs model failover (FallbackModel)

**But:** KNOX uses `KnoxLiteLLMModel` (a `LiteLLMModel` subclass), not a native
pydantic-ai model. litellm manages its own httpx clients internally. To use
`AsyncTenacityTransport` we'd need to either (a) inject a custom httpx client
into litellm's provider, or (b) adopt native pydantic-ai models
(`OpenAIChatModel` with `LiteLLMProvider`) which accept `http_client=`.

This pairs with T7 (native models). Do T7 first, then T2 becomes trivial.

---

## T3 — Replace `_enrich_retry_messages` with `output_validator` + `ModelRetry` (HIGH)

**Current:** `litellm_model.py` `_enrich_retry_messages()` post-processes the
litellm message list to find assistant→user "Validation errors" pairs and
inject the valid partial JSON extracted via `salvage_json`. ~50 lines of
message-list mutation. This is exactly the "custom RetryPromptPart injection"
pattern you proposed.

**Pydantic-AI built-in:** The `@agent.output_validator` decorator lets you
validate output and raise `ModelRetry(message)` to send feedback to the model.
PydanticAI auto-generates the `RetryPromptPart` from the exception. The
retry budget is controlled by `Agent(retries={'output': N})`.

**Why it's better:**

- Uses the documented, stable extension point instead of intercepting the
  message list inside the model subclass
- The validator runs at the *right* layer (output processing), not at the
  message-conversion layer (where `_enrich_retry_messages` hacks in)
- The validator has full access to the parsed output via Pydantic, so it can
  produce richer retry messages than our current "here's the valid JSON,
  here's what's broken" heuristic
- PydanticAI already extracts valid partial output via `allow_partial` — we
  can reuse that instead of calling `salvage_json` ourselves

**The pattern:**

```python
@agent.output_validator
def salvage_and_repair(ctx, output):
    # ctx has the raw response; we can inspect partial state
    # raise ModelRetry with actionable feedback if output is incomplete
    return output
```

**Risk:** Need to confirm the output_validator fires *before* the model sees
the retry — it does (PydanticAI runs validators then emits RetryPromptPart on
ModelRetry). Our current `_enrich_retry_messages` runs *after* PydanticAI's
own validation, which is the wrong place.

**Net diff:** ~50 lines deleted from `litellm_model.py`, ~15 lines added as
`output_validator` in `KnoxAgentBase` or per-subclass.

---

## T4 — Replace conversational tool loop with native PydanticAI tools (HIGH)

**Current:** `acp/conversational.py` `ConversationalAgent.run()` implements a
hand-rolled tool-calling loop: system+user → `litellm.completion(tools=...)` →
parse tool_calls → execute via `ToolDispatcher` → append results → repeat,
bounded by `max_turns`. ~150 lines including message formatting, tool-call
dict construction, and `_notify` progress chunks.

**Pydantic-AI built-in:** `@agent.tool` / `@agent.tool_plain` decorators
register functions as tools. PydanticAI handles the entire loop, tool-call
parsing, result feeding, and retry budgeting internally. Tool docstrings
auto-extract parameter descriptions. `tool_choice`, `sequential`,
`requires_approval`, `prepare`, and `defer_loading` all built-in.

**Why it's better:**

- Removes ~150 lines of bespoke tool-loop orchestration
- Proper `ModelRetry` handling for tool failures
- Auto-extracted tool schemas from signatures + docstrings (griffe)
- Native support for `ToolReturn`, multimodal content, deferred approval
- The agent loop is testable with `TestModel` and `FunctionModel`

**The pattern:**

```python
class KnoxConversationalAgent:
    def __init__(self, dispatcher: ToolDispatcher):
        self._agent = Agent(
            model=KnoxLiteLLMModel(model_name),
            tools=[self._wrap_tool(name) for name in dispatcher.enabled_tools],
            deps_type=SessionDeps,
            retries={'tools': max_turns},
        )

    def _wrap_tool(self, name):
        async def tool(ctx, **kwargs):
            return await self._dispatcher.execute(name, kwargs)
        tool.__doc__ = self._dispatcher.docstring(name)
        return tool
```

**Risk:** The `ToolDispatcher` currently wraps MCP server functions. Native
tools need to map cleanly. The `_notify` progress chunks (ACP streaming) need
to hook into `agent.iter()` events rather than being called inline. This is
doable — pydantic-ai's `agent.iter()` yields `AgentStreamEvent` including
`PartStartEvent` / `PartDeltaEvent` for tool calls.

**Net diff:** ~150 lines deleted from `conversational.py`, replaced by ~40
lines of tool-wrapping. Total `acp/` shrink ~30%.

---

## T5 — Replace hand-rolled ACP server with `run_acp_stdio_sync` (HIGH, experimental)

**Current:** `acp/knox_agent.py` (463 lines) implements the full `acp.Agent`
protocol by hand: `initialize`, `new_session`, `prompt`, provider RPC handlers
(`providers/list`, `providers/set`, `providers/disable`), slash commands,
session management, progress chunking.

**Pydantic-AI Harness built-in:** `pydantic_ai_harness.experimental.acp.run_acp_stdio_sync`
serves any PydanticAI `Agent` as an ACP agent over stdio. It handles:

- Streamed assistant text/reasoning, chunked under wire limit
- Rich tool-call rendering (file edits as diffs, shell commands)
- Human-in-the-loop tool approval via `requires_approval`
- Per-workspace sessions via `session_config` factory
- Session persistence via `session_store`
- MCP server plumbing via `mcp_capabilities`
- Model selection via `models=['...']`
- Cancellation and multi-turn history

**Why it's better:**

- Removes ~400 lines of ACP boilerplate
- Gets diff rendering, approval workflows, session persistence for free
- Maintained upstream as the protocol evolves
- Pydantic-ai team owns the wire-format edge cases

**The pattern:**

```python
from pydantic_ai_harness.experimental.acp import run_acp_stdio_sync, AcpSession, AcpSessionConfig

agent = build_knox_agent()  # T4 simplified agent

def session_config(session: AcpSession) -> AcpSessionConfig:
    return AcpSessionConfig(
        deps=SessionDeps(cwd=session.cwd, vault=resolve_vault(session.cwd)),
        toolsets=[knox_toolset_for(session)],
    )

if __name__ == '__main__':
    run_acp_stdio_sync(agent, session_config=session_config)
```

**Caveats:**

- **Experimental API:** `pydantic_ai_harness.experimental` may change without
  deprecation. Acceptable for KNOX's ACP feature (also new and
  non-load-bearing).
- **Slash commands:** the adapter "does not yet advertise any commands" — our
  `/knox compile`, `/knox query`, etc. would need to become regular tools or
  instructions rather than slash commands. This is arguably *better* (the
  conversational agent just calls the `compile_wiki` tool when asked to compile).
- **Provider injection (providers/set):** the harness has its own model
  selection via `models=` and `model_resolver`. Our current `ProviderCredentials`
  store (per-session model switching) needs to map onto `model_resolver`.

**Net diff:** ~400 lines deleted, ~50 lines added. `acp/` directory shrinks
from 1528 lines to ~600. The hand-rolled `KnoxAgent(Agent)` class is deleted
entirely; only `provider_store.py`, `definitions.py`, `tools.py` survive.

**Recommendation:** Do T4 first (conversational agent → native tools), then T5.
T5 is the biggest win but depends on T4.

---

## T6 — Add `NativeOutput` for Ollama schema-constrained decoding (MEDIUM)

**Current:** `litellm_model.py` `_build_completion_kwargs` sets
`response_format={"type": "json_object"}` for structured-output agents. This
is "JSON mode" — valid JSON but not schema-constrained. The model can produce
any valid JSON, which is why we need `salvage_json` + retry enrichment.

**Pydantic-AI built-in:** `NativeOutput(MyModel)` wraps the output type and
tells pydantic-ai to use the model's native structured-output feature. For
Ollama (v0.5.0+), this is `response_format` with `json_schema` and
llama.cpp's grammar-constrained decoder — the model literally *cannot*
generate output that violates the schema.

**Why it's better:**

- Schema-constrained generation eliminates most JSON repair needs
- `salvage_json` becomes unnecessary for Ollama models
- Retry enrichment becomes unnecessary for Ollama models
- Works with cloud providers that support native structured output (OpenAI,
  Gemini)

**But:** KNOX uses `KnoxLiteLLMModel` which sets `response_format` manually.
To use `NativeOutput` we'd pass `output_type=NativeOutput(MyModel)` to the
`Agent` constructor and let pydantic-ai manage the response_format. This
conflicts with our manual `response_format={"type": "json_object"}` in
`_build_completion_kwargs` — one or the other, not both.

**Recommendation:** Make `output_type` configurable per agent. For Ollama
backends, use `NativeOutput`. For backends without native structured output,
fall back to `ToolOutput` (default) or `PromptedOutput`.

Pairs with T7 (native models) — `NativeOutput` is a pydantic-ai-native concept;
our `LiteLLMModel` subclass would need to respect it rather than force
`json_object`.

---

## T7 — Evaluate native pydantic-ai models over `KnoxLiteLLMModel` (MEDIUM, strategic)

**Current:** `KnoxLiteLLMModel(LiteLLMModel)` exists to preserve litellm's
custom-provider dispatch (`opencode-zen/*`, `ollama_cloud/*`, `claudinio/*`).
The `pydantic-ai-litellm` `LiteLLMModel` is "a model that uses LiteLLM to call
various LLM providers" — but pydantic-ai's *native* models (`OpenAIChatModel`,
`AnthropicModel`, `GoogleModel`, `OllamaModel`) use vendor SDKs directly.

**Pydantic-AI now supports:**

- `OllamaModel` — dedicated Ollama support with `NativeOutput` for
  schema-constrained decoding (T6)
- `LiteLLMProvider` — for LiteLLM proxy servers (not our use case)
- All OpenAI-compatible providers (`OpenAIChatModel` + provider classes) for
  Ollama Cloud, DeepSeek, etc.

**The question:** Do KNOX's custom litellm providers (`opencode-zen/*`) still
need litellm's dispatch? Or can they be expressed as OpenAI-compatible
endpoints via `OpenAIChatModel` + `OpenAIProvider(base_url=..., api_key=...)`?

**If yes (custom providers can become OpenAI endpoints):**

- `KnoxLiteLLMModel` collapses to `OpenAIChatModel` with per-provider config
- T2 (HTTP retries via `AsyncTenacityTransport`) becomes available
- T6 (`NativeOutput`) works natively
- Custom model profiles via `OpenAIModelProfile` (e.g.
  `openai_chat_supports_multiple_system_messages=False` for Ollama)

**If no (litellm's provider dispatch is load-bearing):**

- Keep `KnoxLiteLLMModel` but audit every override against the parent
- The current overrides are: `_completion_create` (adds response_format,
  reasoning_effort), `_process_response` (salvage_json),
  `_enrich_retry_messages` (retry enrichment), `request` (failover).
- T1, T3, T6 each eliminate one override; after T1+T3 only `_process_response`
  (salvage) and `_completion_create` (response_format) remain, both of which
  `NativeOutput` would replace.

**Recommendation:** Audit the custom providers first. If they're all
OpenAI-compatible HTTP endpoints (likely — they're wrappers around
OpenAI-compatible inference servers), migrate to native models. This is the
single highest-leverage simplification — it makes T1, T2, T3, T6 all trivial
and unlocks the rest of pydantic-ai's native features.

---

## Summary table

| ID | Target | Lines saved | Risk | Dependency |
|----|--------|-------------|------|------------|
| T1 | FallbackModel for failover | ~40 | Low | None |
| T2 | AsyncTenacityTransport for HTTP retries | ~20 | Low | T7 |
| T3 | output_validator + ModelRetry | ~50 | Low | None |
| T4 | Native tools for conversational loop | ~110 | Medium | None |
| T5 | run_acp_stdio_sync for ACP | ~350 | Medium | T4 |
| T6 | NativeOutput for Ollama schema | ~30 | Low | T7 |
| T7 | Native models over KnoxLiteLLMModel | ~100 | High | Audit first |

**Total potential:** ~700 lines removed from `agents/` and `acp/`,
replaced with documented, maintained pydantic-ai built-ins.

## Recommended order

1. **T7 audit first** — check whether custom providers are OpenAI-compatible.
   This determines whether the whole stack moves to native models.
2. **T1 (FallbackModel)** — immediate, low-risk, removes hand-rolled loop.
3. **T3 (output_validator)** — removes the `_enrich_retry_messages` hack.
4. **T4 (native tools)** — simplifies conversational agent, enables T5.
5. **T5 (run_acp_stdio_sync)** — biggest single win, depends on T4.
6. **T6 (NativeOutput)** — depends on T7 outcome.
7. **T2 (AsyncTenacityTransport)** — last, only fully available after T7.
