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
   backed by `llmdb` and a local cache). A provider with a known `api` base URL
   (e.g. `claudinio`, `opencode-zen`, `deepseek`) becomes an OpenAI-compatible
   `OpenAIChatModel` with the discovered endpoint and API key env var;
   otherwise fall through to pydantic-ai native resolution.
2. If the catalogue has no entry for the provider either, try pydantic-ai's
   `infer_model()` (ollama, openai, anthropic, google-gla, groq, mistral,
   cohere, bedrock, etc.).
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

`_split(model_string)` applies the `kompend.llm.model_id` rule (first `/` or `:` wins). Provider
aliases live in `_PROVIDER_ALIASES`, the module's only hardcoded mapping. `resolve_model_limits`
raises `KeyError` for an unlisted model and `LookupError` without limit data; discovery handles both.

## Model string parsing — `kompend.llm.model_id`

`provider_id_of(model_string)` → provider, `None` when bare; `model_slug(model_string)` → model part.
One rule for every consumer (`user_providers`, `debug`, `diagnostics`, `rag.vector_store`): first `/`
or `:` wins (never position 0); the model part may hold further separators (`ollama/hf.co/qwen3.6:latest`
→ `ollama`, `hf.co/qwen3.6:latest`). No call site keeps its own split.

## Context-window discovery

`discover_context_window()` (in `context_window.py`) resolves a model's
token limits through a first-hit chain: `[providers.<name>]` override
(from kompend.toml; per-model → provider-level → legacy `context_window_cap`)
→ Ollama runtime `/api/show` → `models.dev` catalog → `[compile.llm]`
fallback → global cap. Explicit user config always beats auto-discovery,
mirroring `build_model()`'s `[providers.<name>]` precedence. The result
records its `ContextSource` (`user_provider`, `ollama`, `models_dev`,
`fallback`) for doctor diagnostics. An override-only `[providers.<name>]` block
(no `base_url`) is valid for any provider resolvable natively or via the
catalog — `is_known_provider()` (exported from `models.py`) is the predicate
`kompend doctor` uses to distinguish such blocks from typo'd provider ids.

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
fallback chains); OpenAI-compatible endpoints and Ollama get no ceiling, and
`kompend doctor` prints which rule applies in its `output-ceiling` check.

**Catalog freshness.** The models.dev payload is downloaded, not shipped, and
cached at `~/.cache/kompend/models.json`; reads never touch the network and
`kompend doctor` refreshes a cache older than `catalog.CACHE_TTL_HOURS`
(168 h). A model missing from the payload makes kompend send the computed
ceiling without a clamp and take the window from `[providers.<name>]`.

## Request-fit planner — `kompend.llm.request_budget`

`RequestBudget` is the one place that decides whether a request fits the
window. A stage declares the window and the **named** payloads it reserves
(system prompt, known titles, pinned facts) and reads the content budget:

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
| `utilization` | Content share of usable (default `compile.llm.request_utilization` = 0.9); the rest is headroom for tokenizer and framing drift |
| `available_tokens` | Signed: window − answer − safety − reserved |
| `content_tokens` / `content_chars()` | What is left for content (≥ 1), sized by `compile.llm.content_window_tokens` |
| `fits()` / `require()` | Check, or raise `LLMOutputError` naming the components |
| `report()` | One-line log description of the budget |

Contracts:

- **Content is never cropped to fit.** A too-small budget makes the work split
  across more batches or segments; it never drops text.
- The answer reserve and the safety margin are paid by default, so a stage cannot forget either.
- `context_safety_tokens` is defined **once** here (8% of the window, floor
  512 tokens) and `kompend.compilation.two_phase` re-exports it for the root
  path; `usable_window_tokens` derives from it the share a trigger is a fraction of.
- `chars_to_tokens` / `tokens_to_chars` are the single chars-per-token
  conversion; stages size with `calibrated_density_for` — the floor tightened
  by the run's measured density, never loosened past it. Content keeps
  `compile.llm.request_utilization` (0.9) of usable as headroom.

Stages that size a request must use this planner, not their own arithmetic:
extraction batching, agentic generation segments, reference extraction, the
agentic section digest, WiCER probes.

**Time is the second bound** (`kompend.llm.throughput`). `fit()` solves prefill
and decode rates from recorded calls, `timeout_for()` prices one request and
clamps it to `compile.llm.request_ceiling_s`, and `time_bounded_chars()` turns
that into a payload size; below `MIN_SAMPLES` the ceiling stands.

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
formats, owning `salvage_json`, `encode_for_llm`, `decode_from_llm`,
`format_note` and `encode_payload`. Supported formats
(`LlmDataFormat` enum, config key `compile.llm.data_format`):

| Value | Token reduction | Package | Scope |
|-------|----------------|---------|-------|
| `JSON` | baseline | stdlib | input + output |
| `TOON` | 30-60% | `toon_format` (core dep) | input only |
| `ISON` | 30-70% | `ison-py` → `ison_parser` (core dep) | input only |

**Input only.** TOON and ISON apply to context data serialized into prompt
templates. Model *output* stays JSON: providers enforce `response_format:
json_object`, and pydantic-ai's structured-output modes are JSON-schema based.

**Teach the format, or the model cannot read it.** `encode_payload(data, fmt)` appends the format
specification (`format_note`, rendered from `templates/llm/format_<format>.md.j2`) to the encoded
data — use it for every structured blob injected into a prompt. `encode_for_llm` is only for a
caller that supplies the specification itself (e.g. console output in `cli/_query_helpers.py`).

**The format travels with the agent.** `KompendAgentBase.data_format` holds the resolved
`compile.llm.data_format`; prompt-building sites read it off the agent
(`CategorizeAgent` → `_build_files_payload`), and config holders without an agent read
`config.llm.data_format` directly.

**`decode_from_llm(text, fmt)`** is the inverse and never raises: a model that
cannot produce the format is a fact to report (`kompend doctor`), not an error.

## One sentence for a missing model

`kompend.llm.messages.NO_MODEL_CONFIGURED_MESSAGE` is the single wording every surface raises
when `compile.llm.default_model` is unset; it names the key and `kompend config set`, a surface
appends only its own next step, and `tests/unit/test_llm_messages.py` fails on a second copy of
the literal. `kompend doctor` probes each format on the configured model
(`cli/cmds/quality/_capabilities.py`) and prints the input-side size saving.

## Failure classification and the decision sidecar

`kompend.llm.errors` answers **transient or semantic?** `is_transient_error`
matches exception *types* and HTTP *status codes* only (`TRANSIENT_STATUS_CODES`
= 408/409/425/429/5xx/529, `httpx`/`httpx2`); `KompendAgentBase` raises
`LLMTransientError` (an `LLMOutputError`) for a transient cause, the rest is
unretryable. Sanctioned exception: `is_quota_refusal` — quota death and throttle
share 429, so the body's quota vocabulary (`QUOTA_BODY_MARKERS`, narrower than
"rate limit") tells "do not send again this run" from "retry in a minute".

`kompend.llm.breaker` is the per-model circuit breaker: one `ModelBreaker` per
run, consulted atop every `invoke_agent` attempt (quota trips unconditionally,
identical failures past one full retry cycle — `transient_retries`, at least a pair,
refusing with `BreakerOpenError`, never transient, one WARNING on the trip).

`kompend.llm.decision` holds the **decision model** vocabulary (TypeSafe SystemOne /
Jev): `ErrorAction`, `ErrorDecision` (action only — never free text),
`DECISION_RESULT_TYPES` (the doctor's offline schema check reads it). The model is an
ordinary agent on the `typesafe` provider; `decision.enabled/model/timeout` name it,
`[providers.*]` carries the credentials. `decide_query_intent` returns the verdict or
`None`; the generation helpers return a three-state outcome (`ANSWERED`, silent
`NOT_CONFIGURED`, `UNAVAILABLE` = degrade the run and say so). A quota-dead primary
fails over to `compile.llm.fallback_chain` first (dead-model errors only; config errors
stay `UNAVAILABLE`). **The endpoint follows the config, not the call order**: an
unconfigured provider raises rather than trusting the SDK default — only `typesafe/...` may use it.

`build_decision_model` sends decision payloads uncompressed
(`compression=False`): the SDK zstd-compresses bodies of 1 KiB and more, and
third-party systemone endpoints (ollama.com) do not strip `Content-Encoding`
— the frame reaches them as JSON starting with `0x28` (`'('`). api.typesafe.ai
decompresses natively; nothing else is assumed to.

## Verification

- `uv run pytest tests/unit/test_llm_models.py tests/unit/test_llm_context_window.py tests/unit/test_request_budget.py tests/unit/test_throughput.py tests/unit/test_llm_overflow.py tests/unit/test_doctor_decision.py tests/unit/test_llm_messages.py -q` — model construction, window discovery, request-fit planner, provider overflow, the decision result-type registry against the real grammar, the missing-model sentence
