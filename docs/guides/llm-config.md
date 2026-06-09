# LLM Configuration

KNOX routes every LLM call through a shared client
([`knox.llm.LLMClient`](https://codeberg.org/jr2804/knox/src/branch/main/src/knox/llm/client.py)) so every tier
gets the same retry, failover and observability semantics.

This guide covers the **Q12** features: typed errors and a
configurable failover chain.

## Typed errors

The shared client raises one of three classes from
[`knox.llm.errors`](https://codeberg.org/jr2804/knox/src/branch/main/src/knox/llm/errors.py) — never a raw
provider exception:

| Class | When it fires | Retry behaviour |
|---|---|---|
| `TransientError` | Rate limit, timeout, 5xx, connection error | Retried with exponential backoff, then the chain advances to the next model |
| `PermanentError` | Bad request, content policy, context window exceeded, malformed JSON | No retry — surfaced or advances to the next model |
| `ConfigurationError` | Missing API key, unknown model, permission denied | Fatal — fails fast without trying any other model |

Unknown exception types default to `TransientError` so a single
upstream library change cannot break the entire pipeline.

## Failover chain

A chain is an ordered list of models. The primary is tried first;
each subsequent model is tried only when the previous one exhausts
its retries with transient errors.

### CLI

```bash
knox compile sources/ \
  --llm-fallback-chain openai/gpt-4o \
  --llm-fallback-chain anthropic/claude-3-haiku-20240307 \
  --llm-fallback-chain ollama/llama3.2
```

`--llm-fallback-chain` is repeatable. Without it the pipeline runs in
single-model mode — the pre-Q12 behaviour.

### Environment

```bash
export KNOX_LLM_FALLBACK_CHAIN="openai/gpt-4o,anthropic/claude-3-haiku-20240307,ollama/llama3.2"
export KNOX_LLM_FALLBACK_MAX_RETRIES=2
```

### Programmatic

```python
from knox.core.pipeline import PipelineConfig, build_llm_client

config = PipelineConfig(
    source_dir=Path("sources"),
    output_dir=Path("wiki"),
    llm_model="gpt-4o",
    llm_provider="openai",
    llm_fallback_chain=[
        "anthropic/claude-3-haiku-20240307",
        "ollama/llama3.2",
    ],
    llm_fallback_max_retries_per_model=2,
)
client = build_llm_client(config)
```

## Retry semantics

For each model in the chain:

1. First attempt
2. On `TransientError`: sleep `initial_backoff_s × 2^retry` (default
   1s → 2s → 4s, capped at 8s)
3. Retry up to `max_retries_per_model` times
4. On `PermanentError`: skip retries, advance to the next model
5. On `ConfigurationError`: raise immediately

`max_retries_per_model = 2` (the default) means each model gets up to
three attempts (1 initial + 2 retries) before the chain advances.

## Observability

Every successful call produces a `CallDiagnostics` record:

```text
LLM ok role=resolve model=openai/gpt-4o attempts=2 latency_s=1.42
```

| Field | Meaning |
|---|---|
| `role` | The pipeline role that issued the call (e.g. `resolve`, `wicer.judge`) |
| `model_used` | The model in the chain that produced the result |
| `attempts` | Total provider invocations across all models tried |
| `total_latency_s` | Wall-clock time including backoff sleeps |

Failed calls produce a warning with the same fields plus the
classified error type.

## FAQ

### Why did the LLM call fail?

Look for `LLM call failed role=… model=… attempts=… <error>` in the
logs. The error message is the classified type — `TransientError`
means every model in the chain exhausted retries; `PermanentError`
means the prompt itself is rejected and no model could help;
`ConfigurationError` means an API key or model name needs fixing.

### How do I see which model was used?

`LLM ok role=… model=…` log lines emit `model_used` for every
successful call. When using `LLMClient.complete()` directly,
`result.diagnostics.model_used` returns the same value.

### Does failover apply to single-model setups?

Yes — even with no fallback chain, `max_retries_per_model` still
applies to the single model so transient errors are retried.
