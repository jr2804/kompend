# LLM Configuration

kompend routes every LLM call through **PydanticAI agents** (`kompend.agents.*`)
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

kompend uses the `provider/model` convention everywhere (e.g.
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

`max_connections` applies wherever kompend creates parallel workers for a model
on that provider (parse today, and future parallel LLM stages) — a quiet way
to rate-limit online APIs so tokens aren't burned faster than intended, or to
stop two local GPU servers from starving each other under `compile.workers > 1`.

The count it clips is a **thread** count. Threads only run Python code in
parallel on a free-threaded interpreter (CPython 3.14t+); on a standard GIL
build, not how much runs concurrently. See
[Parallel workers and the GIL](compilation.md#parallel-workers-and-the-gil).

`api_key` supports the same `${ENV_VAR}` interpolation kompend uses elsewhere.
A plain literal works too. Omit `api_key` entirely for keyless endpoints;
kompend sends a placeholder token because pydantic-ai/OpenAI SDK require one.

**`api_type`** selects the wire protocol kompend speaks to the endpoint:

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

Set `context_window` so kompend knows the model's token budget — without it,
kompend falls back to `fallback_max_input` from `[compile.llm]` (since
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
```

### TOML

```toml
[compile.llm]
default_model = "ollama/qwen3"
fallback_chain = ["openai/gpt-4o", "anthropic/claude-3-haiku-20240307"]
```

## Context window requirements

kompend auto-detects each model's context window (Ollama `num_ctx` →
[models.dev](https://models.dev) catalog → `fallback_max_input`) and sizes
every LLM call to fit. So there is **no hard minimum** that the pipeline
enforces — but a model that's too small produces tiny, low-quality chunks.

### How input is budgeted

Each compile stage carves out room for the answer _before_ sizing input:

```text
answer_reserve    = clamp(context_window / 4, 512, 8192)
stage_input_tokens = context_window - system_prompt - answer_reserve - safety_margin
```

The reserve is derived from the model's window (`ContextWindow.answer_reserve`)

- there is no per-stage output ceiling and kompend never sends `max_tokens`.
A stage that writes a longer answer simply gets it: the window stops it, not a
configured number. What the reserve buys is a _planned_ prompt that leaves the
answer somewhere to go.

**The Anthropic protocol is the exception.** Anthropic-compatible endpoints
(e.g. a `[providers.minimax]` block with `api_type = "anthropic"`) reject a
request that carries no `max_tokens`; left unset, the SDK fills in 4096 and
silently caps every reply. For those models only, kompend sends a value
computed per request — window − estimated prompt − a 5% margin, clamped by the
model's own output limit from the models.dev catalog when it knows the model.
`kompend doctor` reports which rule applies (`output-ceiling-llm`).

The catalog comes from [models.dev](https://models.dev) through `llmdb`.
kompend caches the payload (`~/.cache/kompend/models.json`, TTL 168 h) so a
run never waits for a download, and `kompend doctor` refreshes it when it is
stale - a model released this week therefore appears after a doctor run, not
after a package upgrade. While the model is still missing, kompend sends the
computed value without the clamp and takes the window from
`[providers.<name>].context_window`.

Practical effect by window (generation, the binding stage):

| Context window | Answer reserve | Input for a generation call |
| -------------- | -------------- | --------------------------- |
| 8 K            | 2 048          | ~5.9 K tokens               |
| 32 K           | 8 192          | ~24.5 K tokens              |
| 128 K          | 8 192          | ~119 K tokens               |

### Recommended minimums

| Context window | Verdict                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| **< 8 K**      | Not enough room for a chunk plus an answer - chunks become near-empty. Avoid. |
| **8 K-16 K**   | Functional floor: generation gets ~6 K tokens of input - usable, cramped when dense. |
| **32 K**       | Comfortable minimum for typical PDFs and web pages. The default `fallback_max_input`. |
| **128 K+**     | Recommended for large/long documents. The chunker ceiling is 131 072 tokens. |

The binding constraint is the smallest window that still fits a minimum chunk plus
the answer reserve:

```text
chunk_size_min (2 048) + answer_reserve (window / 4, at most 8 192)
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

## Extraction strategies

`compile.llm.extraction_strategy` only affects **multi-chunk sources** — a
source that fits in one chunk is extracted the same way under all three.

| Strategy       | Mechanics                                                                                                                                                                                                                            | Cache                    |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------ |
| `incremental` _(default)_ | One agent conversation per source: batches are fed sequentially via `message_history`, so each batch sees what earlier batches contributed. History is compacted when it exceeds 2× the answer reserve (oldest batches first), so the window never overflows. | Per batch                |
| `reduce`       | Map-reduce: every batch is extracted independently, then one LLM pass merges the partial results into a document-level extraction. Every request is bounded by construction.                                                           | Per batch (+1 merge call) |
| `agentic`      | One agent walks the source's sections in order, accumulating a structured memory digest (concepts, entities, tags, recent section summaries). The digest is injected as a compact system-prompt part and model-consolidated when it outgrows its budget — never cropped. | Per source               |

### Choosing a strategy

- **`incremental` is the right default** for mixed corpora and typical PDFs and
  web pages: cross-batch continuity, per-batch resume, and no model skills
  beyond ordinary structured output.
- **`reduce` for small windows and small models** (roughly ≤ 16–32 K, or any
  model whose tool-calling is shaky). Every request fits by construction; the
  price is one extra merge call per source and no cross-batch context during
  extraction.
- **`agentic` for long, coherent, high-value sources** — manuals, standards,
  books — with a comfortable window (rule of thumb **≥ 48–64 K**) and a model
  that keeps multi-turn tool loops on track. It is the most demanding strategy:
  a rerun re-extracts the whole source (source-level cache only), and the
  digest must fit beside each section. When it cannot, the run aborts with
  `increase compile.llm.context_window or lower compile.budget.chunk_size_max`
  instead of silently cropping — nothing is ever dropped from the digest.
- **Dense, table-heavy references are the agentic edge case.** An API manual
  that chunks into 100+ sections can drive the consolidated digest past
  ~20 K tokens; at a 32 K window that burns. Use `incremental` or `reduce`
  there, or raise the window (a 64 K window fits the same document with room
  to spare).

### Local models: match the window to the server, not the metadata

kompend sizes every request from the _discovered_ window, but an Ollama server
serves whatever context it was launched with (`OLLAMA_CONTEXT_LENGTH`). When
the two disagree, oversized sections fail per request with
`request exceeds the available context size`. Point `compile.llm.context_window`
at the server's real value — and restart Ollama after raising it.

## Retry and structured output

PydanticAI validates structured output against the result type's schema.
When validation fails, the agent retries before surfacing the error.

### Retry controls

```toml
[compile.llm]
tool_retries = 3                   # retries when a tool raises ModelRetry (missing file, bad argument)
transient_retries = 2                # whole-request retries after a timeout / 429 / 5xx
transient_backoff = 5.0              # first backoff in seconds, doubled per attempt
output_retries = 2                  # PydanticAI retry count for output validation (N = N+1 total attempts)
output_retries_enabled = true       # Master switch — set false to disable all retries
delta_retry_enabled = true          # Delta-retry: fix partial output instead of regenerating
```

Three failure modes get separate budgets:

- **`tool_retries`** — a tool that raises `ModelRetry` (a file that does not
  exist, an invalid argument). The tool answers with feedback and the model
  gets another attempt. The default is 3: one hallucinated tool call must not
  abort a run.
- **`output_retries`** — the model produced output that does not match the
  expected schema. A validation error is a different problem with a different
  remedy, so it has its own budget.
- **`transient_retries`** — the request never produced an answer (a timeout, a
  dropped connection, a 429 or 5xx). Nothing about the request needs to change,
  so the same request is repeated with exponential backoff.

The transient policy lives on `KompendAgentBase` (`_invoke`), which every stage
inherits — generation, extraction, lint, verification, decision. A stage must
not grow its own backoff loop: when a call needs more tries, raise
`transient_retries`, not a copy of the loop.

`agentic_output_retries` (default 4) replaces `output_retries` for agentic
agents: they run multi-turn loops where every turn can invalidate the result.

When `output_retries_enabled = false`, the agent makes a single attempt with
no retries regardless of `output_retries` and `tool_retries`.

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

### Lint call timeout

```toml
[lint]
llm_timeout = 300.0    # seconds, per LLM lint call
```

`kompend lint --llm` and the post-compile lint run four LLM rules. They are
long-context JSON requests - `coverage-gap` sends a batch of page summaries,
`clarity-score` and `factual-consistency` send whole pages - so they need more
headroom than a compile call of the same model. `lint.llm_timeout` is separate
from `compile.llm.timeout` for that reason. A transient failure is retried by
the agent base (`compile.llm.transient_retries`) before the rule reports
nothing. A semantic failure is not retried.

### Interaction format (JSON / TOON / ISON)

```toml
[compile.llm]
data_format = "json"   # json | toon | ison | text
```

Structured context that kompend injects into a prompt (lint gap lists, the
organizer's file inventory, partial extractions, WiCER probes, distilled notes)
can be encoded in a compact format:

| Format | Input tokens | Notes |
|--------|--------------|-------|
| `json` | baseline | universal, the default |
| `toon` | ~30–60 % fewer | indentation-based, tabular |
| `ison` | ~30–70 % fewer | table rows, `null`/reference syntax |

The saving is on the **input** side — the model reads fewer tokens, so more of
the context window is left for the answer. Model _output_ stays JSON: providers
enforce `response_format: json_object`, and the structured-output validation is
JSON-schema based.

Every compact payload carries a short syntax specification (rendered from
`templates/llm/format_<format>.md.j2`), because a small model has never seen
TOON or ISON. That costs a few hundred tokens per call, so a compact format only
pays off when the payload is larger than the specification.

Check the model first:

```bash
kompend doctor
```

The doctor probes, per model, native tool-call output, schema-in-prompt output,
raw JSON, and the configured compact format, and prints the input-side size
saving per format (`input-format-size`). A model that fails the
`structured-output-toon` probe should keep `data_format = "json"`.

### Agent notebook (memory)

```toml
[compile.llm]
agent_memory = true    # default
memory_store = "vault" # default
```

The agentic harness can give an agent a scratch notebook (`write_memory`,
`read_memory`, `search_memory`). It only helps a stage where **one agent
instance handles many outputs**: summarisation (one session, many nodes),
verification (one pass, many claims), remediation (one pass, many findings),
distillation (one session, many tag groups).

It does not help stages that build a fresh agent per unit — generation per
segment, page writing per page, query per question, indexing per run: the
notebook dies with the agent. Those stages deliberately keep it off.

`memory_store` picks where the notebook lives:

- `vault` (default) — `.kompend/agent-memory/`: a `MEMORY.md` index per agent
  (injected into prompts, capped at ~2000 tokens) plus topic files the model
  reads on demand. The agent writes that notebook itself, so it is not part of
  the summary cache key — edit it and cached summaries are reused as they are.
  The vault backend backs every notebook stage: summarisation, verification,
  remediation, distillation.
- `session` — in-process only, fresh per stage run.

Turn the notebook off for small models. A model that invents a memory file name
gets `ModelRetry` from `read_memory`; once the tool budget is spent the run dies
with the node left unsummarized.

### Decision model (optional sidecar)

```toml
[decision]
enabled = true
model = "ollama/clef"          # any provider that serves a decision model
timeout = 10.0                 # credentials + base URL come from [providers.*]
```

A decision model answers questions the pipeline cannot answer from types:
"is this failure worth repeating, should this unit of work be skipped, or is
the run doomed?". It is named exactly like every other model
(`provider/model`), so its credentials and base URL come from the same
`[providers.*]` block you already use - TypeSafe's Jev
(`typesafe/jev-latest`) and Ollama's Clef (`ollama/clef`) are both valid.

Where it is used today: **error triage** in the generation loop. A failed
model call is classified first by type (`kompend.llm.errors`): a timeout, a
dropped connection or a retryable status (408, 409, 425, 429, 5xx, 529) raises
`LLMTransientError` and is retried with backoff; a validation or schema failure
is not retried. With `decision.enabled = true`, the decision model gets the
last word - `retry`, `skip` or `abort` - and can therefore also abort a run
when retrying is pointless.

A decision model can only fill choices and booleans, never free text, so its
output type is a single `StrEnum` field (`ErrorAction`). When the sidecar is
disabled or unreachable, the deterministic classification applies and the
pipeline behaves exactly as it does without it.

### Prompt caching

```toml
[compile.llm]
prompt_cache = true   # default
```

Agent runs resend the whole conversation at every step, and the agentic loops
(generation, extraction, summarization) resend the same long prefix on every
pass. With `prompt_cache = true` the agent sets pydantic-ai's unified `cache`
model setting, which is translated per provider:

| Provider | What it becomes |
|---|---|
| Anthropic (API, Vertex, Foundry, Bedrock Messages) | `anthropic_cache` breakpoints on tools, instructions and conversation |
| OpenAI GPT-5.6+ | `openai_prompt_cache_options` (implicit, 30 min) plus an instruction breakpoint |
| OpenAI before GPT-5.6, OpenRouter | implicit caching, with explicit breakpoints passed through |
| Gemini, DeepSeek, xAI, Mistral | implicit caching, or a cache key — nothing extra to send |
| Ollama and other local backends | no cross-request cache exists |

Caching is not free: a cache write costs 1.25x (5-minute retention) or 2x
(1-hour) the input price, while a cache read costs 0.1x. A prefix therefore
breaks even after one read (5 min) or two (1 hour) — which a multi-pass loop
reaches immediately. A one-off request that is never repeated pays the write
premium, so the setting is worth turning off for purely single-shot pipelines.

A gateway that does not accept the field answers 400 with the field name in the
body. The agent detects that once, logs `Prompt caching rejected by …`, rebuilds
itself without caching and retries the same call; set
`prompt_cache = false` to skip that probe.

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
var. If it isn't in the catalog, kompend cannot route to it — file an issue
to get the provider added to models.dev upstream.
