# T7 Implementation Plan — Updated with Answers

## Answers to the three questions

### Q1: Opencode OpenAI-vs-Anthropic routing

**models.dev cannot help** — verified: 0 entries for opencode/zen/claudinio/ollama-cloud
in the 245-model catalog.

**`/v1/models` cannot help** — verified: returns only `{"id", "object", "created",
"owned_by"}`, no endpoint-routing field.

**But the routing is deterministic per model id** — the opencode docs tables are
the source of truth. The pattern is clean and prefix-based:

| Model prefix | Endpoint | Native pydantic-ai model |
|--------------|----------|--------------------------|
| `claude-*`, `qwen*`, `minimax-*` | `/v1/messages` (Anthropic) | `AnthropicModel` |
| `gpt-*` | `/v1/responses` (OpenAI Responses) | `OpenAIResponsesModel` |
| `gemini-*` | `/v1/models/<id>` (Google) | `GoogleModel` |
| everything else (deepseek, glm, kimi, grok, mimo, big-pickle, etc.) | `/v1/chat/completions` (OpenAI-compat) | `OpenAIChatModel` |

**Decision:** Hard-code the prefix → endpoint mapping as a static dict. No
`/v1/models` discovery needed. This deletes the entire 300-line discovery +
streaming-workaround machinery in `opencode.py`. The mapping is ~15 lines and
covers every model in both Zen and Go tables.

### Q2: No LiteLLM fallback — native only

**Confirmed.** `build_model()` only handles our known prefixes. Unknown
prefixes raise `ValueError` with a helpful message listing supported prefixes.

This means dropping litellm-prefixed models the user might have used
(openrouter, together, groq, fireworks, etc.). But:
- pydantic-ai's `infer_model()` already handles these via `:` separator
  (`openrouter:foo`, `together:foo`)
- Users can switch to the `:` convention for those

### Q3: Pydantic-ai auto-provider-resolve

**Yes, but with `:` separator, not `/`.**

`infer_model("ollama:qwen3")` → `OllamaModel`. `infer_model("openai:gpt-5.2")`
→ `OpenAIResponsesModel`. `infer_model("anthropic:claude")` → `AnthropicModel`.

Auto-resolved prefixes (verified): `ollama`, `openai`, `openai-chat`,
`openai-responses`, `azure`, `azure-responses`, `anthropic`, `google`,
`google-cloud`, `groq`, `cohere`, `mistral`, `bedrock`, `huggingface`, `xai`,
`deepseek`, `fireworks`, `github`, `heroku`, `moonshotai`, `nebius`,
`openrouter`, `ovhcloud`, `sambanova`, `together`, `vercel`, `zai`,
`cerebras`, `alibaba`, `litellm`.

**The problem:** KNOX uses litellm's `provider/model` convention throughout
(config files, CLI args, docs). Switching to `provider:model` would break
every config file and every doc example.

**Decision:** `build_model()` translates KNOX's `provider/model` strings to
pydantic-ai's `provider:model` where they differ, and calls `infer_model()`.
For our custom providers (claudinio, ollama-cloud, opencode-zen/-go), it
builds native models directly. This keeps the user-facing `provider/model`
convention while using pydantic-ai's auto-resolution under the hood.

---

## Revised implementation

### `src/knox/llm/models.py` (~100 lines, replaces providers/)

```python
"""Build pydantic-ai Model instances from KNOX model strings.

KNOX uses the 'provider/model' convention (litellm-style). This module
translates to native pydantic-ai models, using 'provider:model' auto-
resolution for standard providers and explicit construction for KNOX's
custom providers (claudinio, ollama-cloud, opencode-zen, opencode-go).
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider


def build_model(model_string: str) -> Model:
    """Build a pydantic-ai Model from a KNOX 'provider/model' string."""
    prefix, _, name = model_string.partition("/")

    # Custom providers with explicit construction
    if prefix == "claudinio":
        return OpenAIChatModel(
            name,
            provider=OpenAIProvider(
                base_url="https://api.claudin.io/v1",
                api_key=os.environ.get("CLAUDINIO_API_KEY", ""),
            ),
        )
    if prefix == "ollama-cloud":
        return OpenAIChatModel(
            name,
            provider=OpenAIProvider(
                base_url="https://ollama.com/v1",
                api_key=os.environ.get("OLLAMA_API_KEY", ""),
            ),
        )
    if prefix in ("opencode-zen", "opencode-go"):
        return _build_opencode_model(prefix, name)

    # Standard providers: use pydantic-ai auto-resolution
    # KNOX uses 'provider/model'; pydantic-ai uses 'provider:model'
    if ":" not in model_string and prefix:
        return infer_model(f"{prefix}:{name}")
    return infer_model(model_string)


# Opencode routing table (deterministic per model id, from opencode docs)
_OPENCODE_ANTHROPIC_PREFIXES = ("claude-", "qwen", "minimax-")
_OPENCODE_OPENAI_PREFIXES = ("gpt-",)
_OPENCODE_GOOGLE_PREFIXES = ("gemini-",)


def _build_opencode_model(tier: str, name: str) -> Model:
    base = f"https://opencode.ai/zen{'/go' if tier == 'opencode-go' else ''}"
    api_key = (
        os.environ.get("OPENCODE_GO_API_KEY")
        or os.environ.get("OPENCODE_API_KEY", "")
        if tier == "opencode-go"
        else os.environ.get("OPENCODE_ZEN_API_KEY") or os.environ.get("OPENCODE_API_KEY", "")
    )

    if name.startswith(_OPENCODE_ANTHROPIC_PREFIXES):
        return AnthropicModel(name, provider=AnthropicProvider(base_url=base, api_key=api_key))
    if name.startswith(_OPENCODE_OPENAI_PREFIXES):
        return OpenAIResponsesModel(name, provider=OpenAIProvider(base_url=f"{base}/v1", api_key=api_key))
    if name.startswith(_OPENCODE_GOOGLE_PREFIXES):
        # Google via OpenAI-compat endpoint — opencode serves gemini at /v1/models/<id>
        return OpenAIChatModel(name, provider=OpenAIProvider(base_url=f"{base}/v1", api_key=api_key))
    # Default: OpenAI-compatible chat/completions
    return OpenAIChatModel(name, provider=OpenAIProvider(base_url=f"{base}/v1", api_key=api_key))
```

### What gets deleted

| Path | Lines |
|------|-------|
| `src/knox/llm/providers/__init__.py` | 169 |
| `src/knox/llm/providers/_base.py` | 160 |
| `src/knox/llm/providers/claudin_io.py` | 40 |
| `src/knox/llm/providers/ollama_cloud.py` | 38 |
| `src/knox/llm/providers/opencode.py` | 310 |
| `src/knox/llm/providers/AGENTS.md` | 70 |
| `tests/unit/test_providers.py` | (rewrite) |
| **Total deleted** | **~790** |
| `src/knox/llm/models.py` (new) | ~100 |
| `tests/unit/test_llm_models.py` (new) | ~120 |
| **Net** | **~−570 lines** |

### Knock-on simplifications (enabling T1, T3, T6)

Once `build_model()` returns native models:

- **T6 (NativeOutput for Ollama):** Local `ollama:<model>` now returns
  `OllamaModel`. Wire `output_type=NativeOutput(MyModel)` for structured-output
  agents when the model is an `OllamaModel`. Eliminates most `salvage_json`
  for local Ollama.

- **T1 (FallbackModel):** `KnoxAgentBase._build_agent()` changes from
  `KnoxLiteLLMModel(primary, fallback_models=[...])` to
  `FallbackModel(build_model(primary), *[build_model(m) for m in fallbacks])`.
  Removes the hand-rolled failover loop in `litellm_model.py`.

- **T3 (output_validator):** `_enrich_retry_messages` in `litellm_model.py`
  moves to an `@output_validator` on `KnoxAgentBase`. Since native models
  don't need the litellm message-list hack, this becomes a clean validator.

- **`KnoxLiteLLMModel` itself shrinks or deletes.** After T1+T3, the only
  remaining reason to keep it is the `_process_response` salvage_json override.
  With T6 (NativeOutput), that's also unnecessary for Ollama. For cloud
  models without native structured output, salvage stays — but it can move
  into an `output_validator` too.

### Migration path for `KnoxLiteLLMModel`

After `build_model()` exists:
1. `KnoxAgentBase._build_agent()` calls `build_model(self._litellm_model)`
   instead of constructing `KnoxLiteLLMModel`.
2. `KnoxLiteLLMModel` is kept only if salvage_json / reasoning_effort /
   response_format customizations are still needed for non-native paths.
   They're not, once native models + NativeOutput + output_validator are in
   place.
3. `KnoxLiteLLMModel` deletes, `litellm_model.py` deletes, the
   `pydantic-ai-litellm` dependency can be removed.

---

## Verification checklist

Before deleting `providers/`:
- [ ] Confirm `opencode-zen/gpt-5.4-nano` routes to `OpenAIResponsesModel`
- [ ] Confirm `opencode-zen/claude-sonnet-4-6` routes to `AnthropicModel`
- [ ] Confirm `opencode-go/qwen3.7-plus` routes to `AnthropicModel` (Go)
- [ ] Confirm `claudinio/claudinio` hits the claudin.io API
- [ ] Confirm `ollama-cloud/deepseek-v4-flash` hits ollama.com
- [ ] Confirm local `ollama/qwen3` routes to `OllamaModel` via `:`
- [ ] Confirm all config files (`config-cloud.toml`, `knox.toml.example`) still work
- [ ] Update `scripts/config-cloud.toml` examples if model strings change
