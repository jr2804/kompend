# Observability with OpenTelemetry

knox instruments its pydantic-ai agents with OpenTelemetry (OTel) out of the box.
When enabled, every agent run, model request, tool call, and validation retry
emits structured spans you can inspect in any OTel-compatible backend.

> **Naming:** this is the OpenTelemetry **tracing** feature (agent runs).
> For the page-to-source provenance command, see [Provenance](provenance.md).

This guide shows the fastest path to local visibility using **Arize Phoenix** — a
pure-Python OTel collector + UI that runs as a single process (no Docker,
no external services).

---

## 1. Install the OTel extra

```bash
# In the knox repo (editable install)
uv sync --extra otel

# Or in a downstream project
uv add "knox[otel]"
```

This pulls in `opentelemetry-sdk` with the OTLP HTTP exporter.

---

## 2. Start Arize Phoenix (local process, no venv pollution)

```bash
uvx arize-phoenix serve
```

Phoenix starts two endpoints:

- **OTLP receiver**: `http://localhost:6006` (receives traces from knox)
- **Web UI**: `http://localhost:6006` (same port — open in your browser)

No Docker, no external services, no auth setup, no venv pollution. The process
stays running in your terminal (or background it with `&`). To stop, press Ctrl+C.

---

## 3. Enable knox OTel export

```bash
export KNOX_OTEL_ENABLED=true
export KNOX_OTEL_ENDPOINT="http://localhost:6006"
export KNOX_OTEL_SERVICE_NAME="knox"
# optional, if you later put Phoenix behind a proxy:
# export KNOX_OTEL_HEADERS="Authorization=Bearer <token>"
```

Then run any knox command that invokes agents:

```bash
knox -w wiki compile --tier wicer
# or
knox -w wiki lint all
# or
knox -w wiki query "your question"
```

---

## 4. Inspect traces in the Phoenix UI

Open `http://localhost:6006` in your browser. You'll see:

| View | What it shows |
|------|---------------|
| **Traces** | One trace per agent invocation (extraction, generation, wicer-evaluate, wicer-probe, lint-rule, etc.) |
| **Spans** | Hierarchical: Agent Run → Model Request → Tool Call → Retry Loop |
| **Attributes** | Each span carries `model`, `prompt`, `response`, `tool_name`, `tool_args`, `tool_result`, `validation_error`, `retry_count` |
| **Filter/Query** | Filter by `service.name = "knox"`, `span.name = "model.request"`, `retry_count > 0`, etc. |

### What to look for

When you see unusual agent behavior — retries that shouldn't fire, long
latencies, validation failures, or budget blow-up — the trace view is the
cheapest place to start:

- **Retry chains**: A failing model run shows up as
  `model.request` → `validation.error` → `model.retry` → … → success/fail.
  Counting the retry spans answers “how often is this happening?” without
  parsing logs.
- **Tool call details**: `tool_call` spans record the exact arguments sent
  to the model and the tool's return value — useful when a tool call is
  what triggers a validation failure downstream.
- **Token usage**: `usage` attributes on model spans carry prompt and
  completion tokens; correlate with `retry_count` to see whether a budget
  cap is the actual failure cause.

---

## 5. Environment variables reference

| Variable | Default | Description |
|----------|---------|-------------|
| `KNOX_OTEL_ENABLED` | (unset/off) | `true`/`1`/`yes` to enable OTel export |
| `KNOX_OTEL_ENDPOINT` | `http://localhost:4318` | OTLP HTTP endpoint (Phoenix: `http://localhost:6006`) |
| `KNOX_OTEL_SERVICE_NAME` | `knox` | Service label in the OTel UI |
| `KNOX_OTEL_HEADERS` | (unset) | Optional `key=value` HTTP headers for auth |

---

## 6. Other OTel-compatible backends

The instrumentation is backend-agnostic — spans conform to standard OTel
semantic conventions, so you can point `KNOX_OTEL_ENDPOINT` at any OTLP HTTP
receiver (typically `:4318`):

- **Jaeger** — `http://localhost:4318`
- **SigNoz** — `http://localhost:4318`
- **OpenObserve** — `http://localhost:5081/api/default`
- **Grafana Tempo** — `http://localhost:4318`
- **collector + downstream** — any OTLP HTTP endpoint

### High-volume runs: enable sampling

A long compile (`--tier wicer`) can emit thousands of spans. `BatchSpanProcessor`
exports everything by default. For high-volume runs, enable ratio-based sampling:

```bash
export OTEL_TRACES_SAMPLER=traceidratio
# keep 10% of traces
# (this is an OpenTelemetry env var, read by the SDK — no knox prefix)
export OTEL_TRACES_SAMPLER_ARG=0.1
```

To validate later, set `OTEL_TRACES_SAMPLER=always_on` to capture everything.

---

## 7. Managed SaaS: Logfire

The instrumentation is powered by the **pydantic-ai** framework, which has
built-in support for [Logfire](https://pydantic.dev/logfire) (the observability
platform from the pydantic team). Logfire is the recommended observability
backend for knox — it captures agent runs, tool calls, and LLM request/response
spans out of the box with no extra configuration.

Enable it by setting `KNOX_LOGFIRE_ENABLED=true` (or installing the `logfire`
package and running `logfire auth`). knox auto-configures Logfire at startup:

```bash
export KNOX_LOGFIRE_ENABLED=true
```

To disable Logfire for a specific invocation, set `KNOX_LOGFIRE_DISABLED=true`
or pass `--no-logfire` to the CLI.

For existing OTel users, Logfire also accepts OTLP endpoints:

```bash
export KNOX_OTEL_ENABLED=true
export KNOX_OTEL_ENDPOINT="https://logfire-us.pydantic.dev"
export KNOX_OTEL_HEADERS="Authorization=Bearer <your-write-token>"
```

---

## 8. What gets traced

| Agent type | Span prefix | Notes |
|------------|-------------|-------|
| `ExtractionAgent` | `extraction.*` | Concept/entity extraction from source chunks |
| `GenerationAgent` | `generation.*` | Wiki page generation with structured output |
| `WiCER` | `wicer.evaluate` / `wicer.probe` | Two-phase refinement; retries visible as child spans |
| `LintRuleAgent` | `lint.rule.*` | Rule evaluation + remediation spans |
| `QueryAgent` | `query.*` | Search + synthesis pipeline |

Each agent run produces one root trace. The trace carries `knox.agent_type`
and `knox.source_id` attributes for filtering.

---

## 9. Disabling

Unset `KNOX_OTEL_ENABLED` or set it to anything other than `true`/`1`/`yes`.
Zero overhead when disabled — the tracer provider is never constructed.
