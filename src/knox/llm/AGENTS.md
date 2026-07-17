# DOX — knox.llm

## Purpose

LLM layer: model construction, context-window discovery, format
utilities (JSON/TOON/ISON), and usage tracking. Business modules build
models via `build_model()` — never construct pydantic-ai models or
import litellm directly.

## Model construction — `knox.llm.models`

`build_model(model_string)` is the single entry point from
`provider/model` strings to native pydantic-ai `Model` instances.

Resolution order:

1. Look up the provider in the **models-dev** bundled catalog.
   - If the provider has a known `api` base URL (e.g. `claudinio`,
     `opencode-zen`, `deepseek`), construct an OpenAI-compatible
     `OpenAIChatModel` with the discovered endpoint and API key env var.
   - Otherwise fall through to pydantic-ai native resolution.
2. If models-dev has no entry for the provider either, try
   pydantic-ai's `infer_model()` (supports ollama, openai, anthropic,
   google-gla, groq, mistral, cohere, bedrock, etc.).
3. If all resolution paths fail, raise `ValueError`.

| Prefix | Native model | Source |
|--------|--------------|-------|
| `claudinio/` | `OpenAIChatModel` at `api.claudin.io/v1` | models-dev |
| `ollama-cloud/` | `OpenAIChatModel` at `ollama.com/v1` | models-dev |
| `opencode-zen/`, `opencode-go/` | `OpenAIChatModel` at `opencode.ai/zen/v1` or `opencode.ai/zen/go/v1` | models-dev |
| `deepseek/` | `OpenAIChatModel` at `api.deepseek.com/v1` | models-dev |
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
token limits from the models-dev catalog. Returns `None` when the
model is not found. The caller (`discover_context_window`) decides how
to handle the miss.

## Context-window discovery

`discover_context_window()` (in `context_window.py`) resolves a model's
token limits through a first-hit chain: litellm static tables → Ollama
runtime `/api/show` → `models.dev` catalog → fallback (clamped, one
WARNING) → user cap.

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

## Verification

- `uv run pytest tests/unit/test_llm_models.py -x` — model construction
- `uv run pytest tests/unit/test_litellm_model.py -x` — FunctionModel wrapper
- `uv run pytest tests/unit/test_context_window.py -x` — window discovery
