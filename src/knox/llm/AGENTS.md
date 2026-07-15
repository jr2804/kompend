# DOX — knox.llm

## Purpose

LLM layer: model construction, context-window discovery, error
classification, failover chains, format utilities (JSON/TOON/ISON),
and usage tracking. Business modules build models via
`build_model()` — never construct pydantic-ai models or import litellm
directly.

## Model construction — `knox.llm.models`

`build_model(model_string)` is the single entry point from
`provider/model` strings to native pydantic-ai `Model` instances.

| Prefix | Native model | Notes |
|--------|--------------|-------|
| `claudinio/` | `OpenAIChatModel` at `api.claudin.io/v1` | OpenAI-compatible |
| `ollama-cloud/` | `OpenAIChatModel` at `ollama.com/v1` | OpenAI-compatible |
| `opencode-zen/`, `opencode-go/` | `_OpencodeModel` | Lazy endpoint-format discovery (see below) |
| `ollama/`, `openai/`, `anthropic/`, `google/`, etc. | pydantic-ai `infer_model()` | Translated `provider/model` → `provider:model` |

Model strings split at the **first** slash; the remainder is the model
name verbatim and may itself contain slashes (e.g.
`ollama/hf.co/qwen3.6` → provider `ollama`, model `hf.co/qwen3.6`).

### OpenCode endpoint discovery

OpenCode Zen/Go serve each model at exactly one API format (OpenAI Chat
Completions, Anthropic Messages, or OpenAI Responses). The format is not
disclosed by `/v1/models`, so `_OpencodeModel` probes each endpoint with a
minimal request on first use, constructs the matching native model, and
caches the result per `(tier, model_name)`. **No hardcoded model-name
prefixes** — discovery is entirely runtime. Auth errors (401/403)
propagate so the user gets a clear message rather than a silent
fallthrough.

## Format module

`knox.llm.format` is the single source of truth for LLM communication
formats. It owns `salvage_json` and `encode_for_llm`.

Supported formats (`LlmDataFormat` enum, config key
`compile.llm.data_format`):

| Value | Token reduction | Package | Scope |
|-------|----------------|---------|-------|
| `JSON` | baseline | stdlib | input + output |
| `TOON` | 30–60% | `toon_format` (core dep) | input only |
| `ISON` | 30–70% | `ison-py` → `ison_parser` (core dep) | input only |

**Input only.** TOON and ISON apply to context data serialized into
prompt templates (`encode_for_llm`).

## Context-window discovery

`discover_context_window()` (in `context_window.py`) resolves a model's
token limits through a first-hit chain: litellm static tables → Ollama
runtime `/api/show` → `models.dev` catalog → fallback (clamped, one
WARNING) → user cap.

`models_dev.py` resolves models absent from litellm's static DB. It
fetches `models.dev/catalog.json`, caches in-memory and on disk
(`~/.config/knox/cache/models-dev-catalog.json`, 7-day TTL). Matching:
exact key → provider-local stem → normalized base + size fuzzy match.

## Error handling

The broad `except Exception` in `failover.py` `FailoverChain.run` is
**deliberate** — it catches everything around the arbitrary
`call(model)` and routes it through `classify_litellm_error`
(guaranteed fallthrough → `TransientError`). Error-hierarchy trust
boundary. Do not narrow it unless litellm ships a `LiteLLMException`
base class.

## Verification

- `uv run pytest tests/unit/test_llm_models.py -x` — model construction
- `uv run pytest tests/unit/test_litellm_model.py -x` — KnoxModel wrapper
