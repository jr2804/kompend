# DOX — kompend.llm

<!-- on entry: loaded when agent enters src/kompend/llm -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

LLM layer: model construction, context-window discovery, format
utilities (JSON/TOON/ISON), and usage tracking. Business modules build
models via `build_model()` — never construct pydantic-ai models or
import LLM SDKs directly.

## Model construction — `kompend.llm.models`

`build_model(model_string)` is the single entry point from
`provider/model` strings to native pydantic-ai `Model` instances.

Resolution order:

1. Look up the provider in the **models.dev catalogue** (`kompend.llm.catalog`,
   backed by `llmdb` and a local cache).
   - If the provider has a known `api` base URL (e.g. `claudinio`,
     `opencode-zen`, `deepseek`), construct an OpenAI-compatible
     `OpenAIChatModel` with the discovered endpoint and API key env var.
   - Otherwise fall through to pydantic-ai native resolution.
2. If the catalogue has no entry for the provider either, try
   pydantic-ai's `infer_model()` (supports ollama, openai, anthropic,
   google-gla, groq, mistral, cohere, bedrock, etc.).
3. If all resolution paths fail, raise `ValueError`.

| Prefix | Native model | Source |
|--------|--------------|-------|
| `claudinio/` | `OpenAIChatModel` at `api.claudin.io/v1` | catalogue |
| `ollama-cloud/` | `OpenAIChatModel` at `ollama.com/v1` | catalogue |
| `opencode-zen/`, `opencode-go/` | `OpenAIChatModel` at the catalogue's `api` URL | catalogue |
| `deepseek/` | `OpenAIChatModel` at `api.deepseek.com/v1` | catalogue |
| `ollama/` | `OllamaModel` | pydantic-ai `infer_model` |
| `openai/` | `OpenAIResponsesModel` | pydantic-ai `infer_model` |
| `anthropic/` | `AnthropicModel` | pydantic-ai `infer_model` |
| `google/` | `GoogleModel` | pydantic-ai `infer_model` |

`_parse_model_string(model_string)` splits on the **first** `/` or `:`
— whichever comes first — into `(provider, model_name)`. The remainder
is passed verbatim and may contain further slashes or colons (e.g.
`ollama/hf.co/qwen3.6:latest` → provider `ollama`, model
`hf.co/qwen3.6:latest`).

Provider ID aliases (`opencode-zen` → `opencode`) are maintained in
`_PROVIDER_ALIASES` — the only hardcoded mapping in the module.

`resolve_model_limits(model_id)` looks up `(max_input, max_output)`
token limits in the catalogue (`catalog.model(...)`). It raises `KeyError`
when the model is not listed and `LookupError` when it carries no limit
data; the caller (`discover_context_window`) decides how to handle the miss.

## Context-window discovery

`discover_context_window()` (in `context_window.py`) resolves a model's
token limits through a first-hit chain: `[providers.<name>]` override
(from kompend.toml; per-model → provider-level → legacy `context_window_cap`)
→ Ollama runtime `/api/show` → `models.dev` catalog → `[compile.llm]`
fallback → global cap. Explicit user config always beats auto-discovery,
mirroring `build_model()`'s `[providers.<name>]` precedence. The result
records its `ContextSource` (`user_provider`, `ollama`, `models_dev`,
`fallback`) for doctor diagnostics.

An override-only `[providers.<name>]` block (no `base_url`) is valid for any
provider resolvable natively or via the catalog — `is_known_provider()`
(exported from `models.py`) is the predicate `kompend doctor` uses to
distinguish such blocks from typo'd provider ids.

### Answer room — `ContextWindow`

A response is bounded by the window, not by a setting: kompend sends **no
`max_tokens`** anywhere. `ContextWindow.answer_reserve`
(`answer_reserve_tokens(window)`) is the *planning* assumption a caller uses
when it decides how much input to pack: a quarter of the window, clamped to
512…8192 tokens. `input_budget(reserved_tokens=0)` is the rest, and
`history_budget` (twice the reserve) is where threaded message history gets
compacted. These values never travel to a provider — they only split work into
requests that fit. `kompend doctor` reports them in its `window-fit` check.

**One exception, and it is not a cap.** The Anthropic protocol rejects a
request without `max_tokens` (pydantic-ai then substitutes 4096, capping every
reply, thinking included). For that protocol only, the agent passes a callable
`model_settings` that computes the value per request - window minus estimated
prompt minus a 5% margin, clamped by the model's own catalog output limit.
`requires_output_ceiling(model)` marks such models (it unwraps wrappers and
fallback chains); OpenAI-compatible endpoints and Ollama get no ceiling at all.
`kompend doctor` prints which rule applies in its `output-ceiling` check.

**Catalog freshness.** `llmdb` downloads the models.dev payload instead of
shipping it, and keeps it in memory only - so kompend caches it at
`~/.cache/kompend/models.json` (see `kompend.llm.catalog`). Reads never touch
the network; `kompend doctor` refreshes a cache older than
`catalog.CACHE_TTL_HOURS` (168 h), and a cache miss downloads once. A brand-new
model therefore shows up after a doctor run, not after a package upgrade. A
model still missing from the payload makes kompend send the computed ceiling
without a clamp and take the window from `[providers.<name>]` config.

## Request-fit planner — `kompend.llm.request_budget`

`RequestBudget` is the one place that decides whether a request fits the
window. A stage declares the window and the **named** payloads it reserves
(system prompt, known titles, pinned facts, tool framing) and reads the
content budget from it:

```python
budget = RequestBudget(window=cw, reserved={"system": sys_tokens})
chars = budget.content_chars(chars_per_token)
```

| Member | Meaning |
|--------|---------|
| `window` | The `ContextWindow` the request is measured against |
| `reserved` | Named fixed payloads; only the sum is subtracted |
| `answer_tokens` | Defaults to `window.answer_reserve` |
| `safety_tokens` | Defaults to `context_safety_tokens(window.max_input)` |
| `utilization` | Scales the remainder (a stage needing a second leg takes a fraction) |
| `available_tokens` | Signed: window − answer − safety − reserved |
| `content_tokens` / `content_chars()` | What is left for content (≥ 1) |
| `fits()` / `require()` | Check, or raise `LLMOutputError` naming the components |
| `report()` | One-line log description of the budget |

Contracts:

- **Content is never cropped to fit.** A too-small budget makes the work split
  across more batches or segments; it never drops text.
- The answer reserve and the safety margin are paid by default, so a stage
  cannot forget either by accident.
- `context_safety_tokens` is defined **once** here (8% of the window, floor
  512 tokens). `kompend.compilation._two_phase_core` re-exports it for the
  historical import path — a second definition is how stages drift apart.
- `chars_to_tokens` / `tokens_to_chars` are the single chars-per-token
  conversion; pass the configured `compile.llm.default_chars_per_token`.

Stages that size a request must use this planner rather than their own
arithmetic: extraction batching, agentic generation segments, reference
extraction, the agentic section digest, WiCER probes.

## Provider overflow — `kompend.llm.overflow`

Discovery reports what a model *can* do; a server may enforce less. The
server says so in the rejection, and it usually names its own limit: `request
(33145 tokens) exceeds the available context size (32768 tokens), try
increasing it`.

| Member | Meaning |
|--------|---------|
| `OVERFLOW_MARKERS` | The provider wordings of "this does not fit" |
| `is_context_overflow(exc)` | Whether *exc* is that failure (walks the `raise … from …` chain) |
| `stated_context_size(exc)` | The limit the provider named, or `None` |

Contracts:

- **Only the limit lowers the window, never the request size.** The pattern
  set matches the *server's* number; a message naming only the requested size
  yields `None`.
- A provider that names no limit gives kompend nothing to plan against — the
  caller must not guess, so the failure propagates.
- The markers live here and nowhere else. A stage that needs to recognise an
  overflow imports `is_context_overflow`; it does not carry its own list.

`EffectiveWindow` (`kompend.llm.request_budget`) holds the window a run
actually plans against and lowers it once, for the rest of the run:

| Member | Meaning |
|--------|---------|
| `current` | The window stages size against |
| `clamp(max_input, reason=…)` | Lower it; returns whether it changed |
| `clamped` / `discovered` / `reason` / `report()` | What changed, and why |

Lowering is **one-way**: a provider that rejected a request of size *N* has
rejected the evidence for a larger window. A limit at or above the current
window is ignored — the provider either repeated the limit kompend already
planned against, or named a number larger than the real one.

`LLMContextOverflowError` (`kompend.core.utils.exceptions`) carries the
provider's number in `stated_limit`. `KompendAgentBase` raises it in place of
the raw provider error, so a stage can lower the window and re-plan instead of
retrying the same rejected request.

## Format module

`kompend.llm.format` is the single source of truth for LLM communication
formats. It owns `salvage_json`, `encode_for_llm`, `decode_from_llm`,
`format_note` and `encode_payload`.

Supported formats (`LlmDataFormat` enum, config key
`compile.llm.data_format`):

| Value | Token reduction | Package | Scope |
|-------|----------------|---------|-------|
| `JSON` | baseline | stdlib | input + output |
| `TOON` | 30-60% | `toon_format` (core dep) | input only |
| `ISON` | 30-70% | `ison-py` → `ison_parser` (core dep) | input only |

**Input only.** TOON and ISON apply to context data serialized into
prompt templates. Model *output* stays JSON: providers enforce
`response_format: json_object`, and the structured-output modes of
pydantic-ai are JSON-schema based.

**Teach the format, or the model cannot read it.** `encode_payload(data, fmt)`
appends the format specification (`format_note`, rendered from
`templates/llm/format_<format>.md.j2`) to the encoded data. Use it for every
structured blob injected into a prompt; use `encode_for_llm` only when the
caller supplies the specification itself (e.g. console output in
`cli/_query_helpers.py`, which is not a prompt).

**The format travels with the agent.** `KompendAgentBase.data_format` holds the
resolved `compile.llm.data_format`; prompt-building sites read it off the agent
instead of threading a new parameter (`CategorizeAgent` → `_build_files_payload`).
Config holders without an agent read `config.llm.data_format` directly.

**`decode_from_llm(text, fmt)`** is the inverse and never raises: a model that
cannot produce the format is a fact to report (`kompend doctor`), not an error.

`kompend doctor` probes each format on the configured model
(`cli/cmds/quality/_capabilities.py`) and prints the input-side size saving, so
the format choice is evidence-based.

## Failure classification and the decision sidecar

`kompend.llm.errors` answers one question about a failed call: **transient or
semantic?** `is_transient_error(exc)` walks the `raise ... from ...` chain and
matches exception *types* and HTTP *status codes* only (`TRANSIENT_STATUS_CODES`
= 408/409/425/429/5xx/529). There is deliberately no message matching — a
gateway that words a 400 as "timed out" is still a client error, and a 504 is
still a server-side timeout. `KompendAgentBase` raises `LLMTransientError`
(a subclass of `LLMOutputError`) for a transient cause, so callers can retry
that class and treat the rest as unretryable.

`kompend.llm.decision` holds the vocabulary for the optional **decision model**
(TypeSafe SystemOne / Jev): `ErrorAction = Literal["retry", "skip", "abort"]`
and `ErrorDecision` (only `action` — a decision model can fill choices and
booleans, never free text). The model itself is an ordinary agent
(`kompend.agents.decision.DecisionAgent`) reached through pydantic-ai's
`typesafe` provider, configured as `decision.enabled/model/api_key/base_url/
timeout`. Every helper returns `None` when the sidecar is off or unreachable,
so the deterministic classification stays the floor.

## Verification

- `uv run pytest tests/unit/test_llm_models.py -x` — model construction
- `uv run pytest tests/unit/test_llm_context_window.py -x` — window discovery
- `uv run pytest tests/unit/test_request_budget.py -x` — request-fit planner
- `uv run pytest tests/unit/test_llm_overflow.py -x` — provider overflow
