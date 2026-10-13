# LLM Configuration

KOMPEND routes every LLM call through **PydanticAI agents** (`kompend.agents.*`)
so every pipeline stage — extraction, generation, query, WiCER, doctor,
capture — gets the same retry, failover, and observability semantics.

Under the hood, each agent wraps a native pydantic-ai Model built by
[`kompend.llm.models.build_model`](https://github.com/jr2804/kompend/src/branch/main/src/kompend/llm/models.py)
from a `provider/model` string. This guide covers:

- Model strings and provider resolution
- API key discovery
- The fallback chain
- Retry and reasoning-effort controls
- Token usage observability

## Model strings

KOMPEND uses the `provider/model` convention everywhere (e.g.
`ollama/qwen3`, `openai/gpt-4o`, `claudinio/claude-4`). `build_model()`
resolves a string to a native pydantic-ai `Model` in three steps:

1. **pydantic-ai native providers** — pydantic-ai's `infer_model()` handles
   standard providers (`ollama`, `openai`, `anthropic`, `google`, `groq`,
   `mistral`, `cohere`, …).
2. **models-dev catalog** — if the provider has a known API base URL in the
   [models.dev](https://models.dev) catalog (e.g. `claudinio`, `opencode`,
   `deepseek`), an OpenAI-compatible `OpenAIChatModel` is constructed with
   the catalog-advertised endpoint.
3. **User-defined providers** — any OpenAI-compatible endpoint declared under
   `[providers.<name>]` in `kompend.toml`. The name becomes the `provider/`
   prefix, e.g. `default_model = "my-llama/mistral-nemo"`. See below.

If no path resolves, `build_model()` raises `ValueError`.

## User-defined providers

Use this when your model host is not a native pydantic-ai provider and not
in the models-dev catalog — typical cases are a self-hosted `llama.cpp`
server, vLLM, text-generation-inference, an OpenAI-compatible mirror, or
an API-proxy that exposes multiple protocols.

**Recommended location:** define providers **globally** in
`~/.config/kompend/config.toml` — they carry credentials and are usually
machine-wide, not per-vault. Per-vault definitions in `.kompend/config.toml`
work too.

```toml
[providers.my-llama]
base_url = "http://localhost:8081/v1"
api_key = "${MY_API_KEY}"            # optional; resolved from env var
api_type = "openai-chat"             # openai-chat (default) | openai-responses | anthropic
context_window = 32768               # default context window for all its models

# Per-model context window — finest granularity, wins over provider-level:
[providers.my-llama.models."mistral-nemo"]
context_window = 131072

[compile.llm]
default_model = "my-llama/mistral-nemo"
```

The provider `id` (`my-llama` above) is just the TOML table key. It must
not collide with a native or catalog provider id. The endpoint is
referenced as `<id>/<model_name>` everywhere `build_model()` is used,
including `default_model`, `fallback_chain`, and CLI overrides.

| Field | Required | Notes |
| ----- | -------- | ---------------------------------------------------- |
| `base_url` | no¹  | Provider base URL (e.g. `http://host:8081/v1`). Omit for an override-only block on an integrated provider (see below) |
| `api_key` | no | Literal value or `${ENV_VAR}` reference |
| `api_type` | no | Wire protocol: `openai-chat` (default), `openai-responses`, or `anthropic` |
| `context_window` | no | Default context window length for all models on this provider |
| `max_connections` | no | Client-side limit on parallel LLM connections to this provider; clips `compile.workers`. `None` (default) = no limit |
| `models."<name>".context_window` | no | Per-model context window (wins over provider-level) |

¹ `base_url` is required to _define_ a custom endpoint; omit it to write an
**override-only block** that only tunes settings for an existing provider
(integrated like `ollama` or catalog-based), e.g.:

```toml
[providers.ollama]        # no base_url — override-only
max_connections = 2       # clip compile.workers to 2 for ollama models
```

`max_connections` applies wherever KOMPEND creates parallel workers for a model
on that provider (parse today, and future parallel LLM stages) — a quiet way
to rate-limit online APIs so tokens aren't burned faster than intended, or to
stop two local GPU servers from starving each other under `compile.workers > 1`.

`api_key` supports the same `${ENV_VAR}` interpolation KOMPEND uses elsewhere.
A plain literal works too. Omit `api_key` entirely for keyless endpoints;
KOMPEND sends a placeholder token because pydantic-ai/OpenAI SDK require one.

**`api_type`** selects the wire protocol KOMPEND speaks to the endpoint:

- `openai-chat` — OpenAI Chat Completions (`/v1/chat/completions`); the
  default and what virtually every OpenAI-compatible server implements.
- `openai-responses` — OpenAI Responses API (`/v1/responses`).
- `anthropic` — Anthropic Messages API (`/v1/messages`), for proxies that
  expose Anthropic's protocol.

**Context window resolution order** (finest wins):

1. `[providers.X.models."<model>"] context_window`
2. `[providers.X] context_window`
3. models.dev catalog
4. `[compile.llm] fallback_max_input` (last resort)

Set `context_window` so KOMPEND knows the model's token budget — without it,
KOMPEND falls back to `fallback_max_input` from `[compile.llm]` (since
self-hosted endpoints are not in the models.dev catalog).

> **Deprecated:** `context_window_cap` (provider level and
> `compile.llm.context_window_cap`) was renamed to `context_window`. A config
> still setting it keeps working — the value is migrated and a deprecation
> warning is logged — but please update it.

## API key discovery

There is no `api_base` / `api_key` config field. API keys are read from
the environment variable(s) the models-dev catalog advertises for each
provider. For example:

| Provider     | Env var                                                 |
| ------------ | ------------------------------------------------------- |
| `claudinio/` | `CLAUDINIO_API_KEY`                                     |
| `opencode/`  | `OPENCODE_API_KEY`                                      |
| `deepseek/`  | `DEEPSEEK_API_KEY`                                      |
| `openai/`    | `OPENAI_API_KEY` (pydantic-ai native)                   |
| `anthropic/` | `ANTHROPIC_API_KEY` (pydantic-ai native)                |
| `ollama/`    | none (local); requires `OLLAMA_BASE_URL` if not default |

```bash
export OPENCODE_API_KEY="oc-..."
```

## Configuring the primary model

Set the primary model in `[compile.llm]`:

```toml
[compile.llm]
default_model = "ollama/qwen3"
vlm_model = "ollama/qwen3"          # separate VLM if desired
timeout = 120.0                      # per-call timeout in seconds
```

## Fallback chain

A chain is an ordered list of models. The primary is tried first; each
subsequent model is tried only when the previous one fails.

Fallback is implemented by wrapping the primary and fallback `KompendModel`
instances in `pydantic_ai.models.fallback.FallbackModel` at agent
construction time.

### CLI

```bash
kompend compile \
  --llm-fallback-chain openai/gpt-4o \
  --llm-fallback-chain anthropic/claude-3-haiku-20240307 \
  --llm-fallback-chain ollama/llama3.2
```

`--llm-fallback-chain` is repeatable. Without it, the pipeline runs in
single-model mode.

### Environment

```bash
export KOMP_LLM_FALLBACK_CHAIN="openai/gpt-4o,anthropic/claude-3-haiku-20240307,ollama/llama3.2"
export KOMP_LLM_FALLBACK_MAX_RETRIES=2
```

### TOML

```toml
[compile.llm]
default_model = "ollama/qwen3"
fallback_chain = ["openai/gpt-4o", "anthropic/claude-3-haiku-20240307"]
fallback_max_retries = 2
```

## Context window requirements

KOMPEND auto-detects each model's context window (Ollama `num_ctx` →
[models.dev](https://models.dev) catalog → `fallback_max_input`) and sizes
every LLM call to fit. So there is **no hard minimum** that the pipeline
enforces — but a model that's too small produces tiny, low-quality chunks.

### How input is budgeted

Each compile stage carves out its output reserve _before_ sizing input:

```text
stage_input_tokens = context_window − system_prompt − stage_output_max
```

System prompts are small (~200 tokens). The per-stage output reserves
(`BudgetConfig`):

| Stage                    | Output reserve (tokens)              |
| ------------------------ | ------------------------------------ |
| **Generation** (binding) | `gen_output_max` = **16 384**        |
| Extraction               | `extract_output_max` = 8 192         |
| Overview                 | `overview_output_max` = 4 096        |
| Lint                     | `lint_output_max` = 8 192            |
| Verify / WiCER           | 4 096                                |
| Summary                  | bounded at 8 000 chars (~2 K tokens) |

### Recommended minimums

| Context window | Verdict                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| **< 16 K**     | ❌ Not enough headroom after the generation reserve; chunks become near-empty. Avoid for compilation.        |
| **16 K–20 K**  | ⚠️ Absolute functional floor. Generation gets ~2 K tokens of input — usable but cramped for dense documents. |
| **32 K**       | ✅ Comfortable minimum for typical PDFs and web pages. The default `fallback_max_input`.                     |
| **128 K+**     | ✅ Recommended for large/long documents. The chunker ceiling is 131 072 tokens.                              |

The **binding stage is generation** (largest reserve at 16 384 tokens). The
practical floor so a model can both receive a meaningful chunk _and_ emit a
full page is:

```text
chunk_size_min (2 048) + gen_output_max (16 384) + overhead (~512) ≈ 19 000
```

For best results use a model with **≥ 32 K** context. Set
`fallback_max_input` if your provider isn't in the catalog:

```toml
[compile.llm]
fallback_max_input = 32768    # used when auto-detection fails
```

Cap an over-large window (e.g. to bound memory) with `context_window`:

```toml
[compile.llm]
context_window = 65536    # hard ceiling on effective window
```

## Retry and structured output

PydanticAI validates structured output against the result type's schema.
When validation fails, the agent retries before surfacing the error.

### Retry controls

```toml
[compile.llm]
output_retries = 2                  # PydanticAI retry count (N = N+1 total attempts)
output_retries_enabled = true       # Master switch — set false to disable all retries
delta_retry_enabled = true          # Delta-retry: fix partial output instead of regenerating
```

When `output_retries_enabled = false`, the agent makes a single attempt with
no retries regardless of `output_retries`.

`delta_retry_enabled` controls the delta-retry hook: when the model returns
partially valid JSON, the hook sends a **correction message** identifying
exactly which fields are broken, rather than asking for a full regeneration.
**For smaller models (< 3 B parameters), disabling delta-retry is
recommended** — the correction message often confuses small models further,
increasing latency without improving results:

```toml
[compile.llm]
delta_retry_enabled = false  # recommended for small models
```

### Batch failure detection

The two-phase compiler tracks per-batch success/failure via a circuit
breaker (`BatchFailureTracker`). When too many batches fail, it stops early
and logs that the model is unsuitable for structured JSON output:

```toml
[compile.llm]
batch_failure_threshold_ratio = 0.5  # stop when >50 % of batches fail
batch_failure_threshold_min   = 5    # minimum batches before the ratio is trusted
batch_retry_attempts          = 3    # extra per-batch retries before counting as failed
```

For small corpora (fewer than 5 sources), lower `batch_failure_threshold_min`
so the circuit breaker can trigger:

```toml
[compile.llm]
batch_failure_threshold_min = 2
```

### Pre-compilation check

Run `kompend doctor` before compiling to verify the configured model can
produce structured JSON. The doctor command probes the model with a trivial
request and reports pass/fail — catching the "empty string" failure mode
before it wastes a full pipeline run.

See [Architecture Decisions → Extraction Retry](../architecture/decisions.md)
for the full failure-cascade diagram.

## Reasoning effort

Controls how much a model "thinks" before responding — useful for local
models that tend to over-reason:

```toml
[compile.llm]
# Supported: "none", "minimal", "low", "medium", "high", "xhigh", "default"
# For Ollama, "none"/"minimal"/"low" also set think=False.
# Default (unset / None) uses the provider's own default.
reasoning_effort = "none"
```

Set via `KOMP_LLM_REASONING_EFFORT`. Best-effort: providers that don't
recognize the key ignore it.

## RLM (code-exploration mode)

RLM lets the agent explore the document context **with code** instead of
having the whole context packed into the prompt. When enabled, the context
attribute is replaced by a placeholder in the rendered prompt, and an
`execute_code` REPL is attached with the full context pre-loaded as a
`context` variable. The model runs search/aggregation code against it, so
conversation size stays bounded even for corpora larger than the model's
context window.

```toml
[compile.llm.rlm]
enable = true               # on/off switch
max_tool_calls = 150        # tool-call budget for the whole run (UsageLimits)
code_timeout = 120.0        # seconds per code execution
sub_model = "ollama/qwen3:8b"  # llm_query sub-model; defaults to the primary
max_depth = 2               # rlm_query recursion depth (0 = non-recursive)
node_budget = 32            # total recursive rlm_query calls per root run
```

Requires a local model that calls tools — a weak or non-agentic model may
ignore the `execute_code` tool and answer from the placeholder, which
contains no content.

Verify against a live local model with
`uv run python scripts/verify/verify_rlm.py` (runs the query agent with RLM on
vs. off over a synthetic corpus larger than the model's context window;
skips cleanly when Ollama is unavailable).

## Token usage observability

Every successful call feeds a shared
[`UsageTracker`](https://github.com/jr2804/kompend/src/branch/main/src/kompend/llm/usage.py)
which accumulates token counts across the whole pipeline run:

| Field               | Meaning                                            |
| ------------------- | -------------------------------------------------- |
| `prompt_tokens`     | Total input tokens across all calls                |
| `completion_tokens` | Total output tokens across all calls               |
| `call_count`        | Number of successful calls                         |
| `per_model`         | Per-model breakdown: `{prompt, completion, calls}` |
| `avg_ttft`          | Average time-to-first-token                        |

After a compile run, the tracker's data is attached to the wiki's
metadata for the [`kompend log`](cli.md) command to render.

## Debug logging

Set `KOMP_DEBUG_LLM_DIR` to a directory path to dump every LLM call
(message list + response) to disk for manual inspection:

```bash
export KOMP_DEBUG_LLM_DIR=".kompend/cache/debug"
```

## OpenTelemetry tracing

kompend can emit [OpenTelemetry](https://opentelemetry.io) traces for its
pydantic-ai agent runs — including the agent run, every model request,
and every tool call — so you can replay and inspect exactly what the
compiler's agents did, including where a local/quantized model hits its
retry ceiling or where the WiCER evaluate/probe steps diverge from the
graph compiler.

The SDKs are already shipped with kompend (`opentelemetry-sdk` is an
optional `otel` extra). Enable tracing with two environment variables:

```bash
export KOMP_OTEL_ENABLED=true
export KOMP_OTEL_ENDPOINT="http://localhost:4318"   # your OTLP backend
```

Options:

- `KOMP_OTEL_ENABLED` — `true`/`1`/`yes` turns tracing on. Unset or any
  other value keeps kompend fully uninstrumented (zero overhead). Default off.
- `KOMP_OTEL_ENDPOINT` — OTLP HTTP endpoint. Default
  `http://localhost:4318`. For local observability, use
  Arize Phoenix: ``http://localhost:6006``.
- `KOMP_OTEL_SERVICE_NAME` — service label in the OTel UI (default `kompend`).
- `KOMP_OTEL_HEADERS` — optional `key=value` HTTP headers for the endpoint,
  e.g. `Authorization=Bearer <token>`.

Install the extra if it isn't present already:

```bash
uv add kompend[otel]
# or, in this repo's editable install:
uv sync --extra otel
```

## Errors

When an agent's LLM call fails (network error, empty content after
retries, non-salvageable output), it raises
[`LLMOutputError`](https://github.com/jr2804/kompend/src/branch/main/src/kompend/core/utils/exceptions.py)
from `kompend.core.utils.exceptions`. Callers (pipeline, WiCER, lint)
surface this as a hard stop rather than silently skipping — a model that
cannot produce usable output should abort cleanly so you can pick a
different one.

## FAQ

### Why did the LLM call fail?

Set `KOMP_DEBUG_LLM_DIR` and re-run. The dumped request/response files
show exactly what was sent and returned. Common causes: wrong provider
prefix (check the [models.dev catalog](https://models.dev) for the
exact name), missing API key env var, or a model too small to produce
valid structured output.

### Does failover apply to single-model setups?

A single model still gets `output_retries` attempts on schema-validation
failures, but there is no secondary model to fall back to.

### How do I use a custom OpenAI-compatible provider?

If the provider is in the [models.dev catalog](https://models.dev),
just use its `provider/model` string and set the advertised API key env
var. If it isn't in the catalog, KOMPEND cannot route to it — file an issue
to get the provider added to models.dev upstream.
