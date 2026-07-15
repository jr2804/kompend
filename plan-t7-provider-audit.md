# T7 Audit: Custom Provider Migration to Native Pydantic-AI Models

## Summary

**Verdict: Migrate to native pydantic-ai models.** All four custom providers
map cleanly to `OpenAIChatModel` / `AnthropicModel` / `OllamaModel`. The
custom litellm `CustomLLM` infrastructure (~700 lines) can be deleted.

**Bonus finding: the custom providers are currently broken at runtime** — a
regression from the LLMClient deletion. This must be fixed regardless of T7.

---

## Critical bug discovered: custom providers never register at runtime

`knox.llm.providers` is **never imported by the application**. It was
previously imported transitively by `src/knox/llm/client.py:25`
(`import knox.llm.providers  # noqa: F401`), but `client.py` was deleted in
commit `f4ca2c0` (LLMClient migration) and nothing replaced that import.

**Effect:** model strings like `ollama-cloud/deepseek-v4-flash` and
`opencode-zen/...` fail at runtime with:
```
litellm.BadRequestError: LLM Provider NOT provided.
```

The cloud config (`scripts/config-cloud.toml`) references these providers in
commented examples. The tests pass only because `tests/unit/test_providers.py`
explicitly imports the module.

**Fix:** Either (a) add `import knox.llm.providers` somewhere in the app
startup path (immediate fix, ~1 line), or (b) complete T7 and delete the
custom providers entirely (strategic fix).

---

## Provider-by-provider audit

Each custom provider was inspected and tested against native pydantic-ai
model construction.

### 1. `claudinio/` — OpenAI-compatible ✅

**Current:** `_ClaudinIOProvider` (claudin_io.py) delegates to litellm's
`openai/` handler at `https://api.claudin.io/v1`. ~30 lines.

**Native equivalent:**
```python
OpenAIChatModel(
    '<model>',
    provider=OpenAIProvider(
        base_url='https://api.claudin.io/v1',
        api_key=os.environ['CLAUDINIO_API_KEY'],
    ),
)
```

**Mapping:** Direct. No special handling needed. The provider is a plain
OpenAI-compatible endpoint.

### 2. `ollama-cloud/` — OpenAI-compatible ✅

**Current:** `_OllamaCloudProvider` (ollama_cloud.py) delegates to litellm's
`ollama_chat/` handler at `https://ollama.com`. ~30 lines.

**Native equivalent:**
```python
OpenAIChatModel(
    '<model>',
    provider=OpenAIProvider(
        base_url='https://ollama.com/v1',
        api_key=os.environ['OLLAMA_API_KEY'],
    ),
)
```

**Mapping:** Direct. Ollama Cloud exposes an OpenAI-compatible `/v1` endpoint.
Could also use `OllamaModel(provider=OllamaProvider(base_url='https://ollama.com/v1'))`,
but `OpenAIChatModel` is simpler and identical on the wire.

**Note:** For *local* Ollama, `OllamaModel` is the better choice because it
enables `NativeOutput` (schema-constrained decoding via llama.cpp grammars).
For cloud, OpenAI-compat is fine.

### 3. `opencode-zen/` and `opencode-go/` — mixed OpenAI + Anthropic ⚠️

**Current:** `_OpenCodeProvider` (opencode.py, ~300 lines) does:
- Model discovery via `GET {base}/v1/models`
- Per-model endpoint routing: `openai/` handler for some models,
  `anthropic/` handler for others
- Stream workaround (opencode sends non-spec SSE chunks)
- Error wrapping with available-models list

**Native equivalent — two options:**

**Option A (simpler, recommended):** Treat all opencode models as
OpenAI-compatible. The `/v1/models` discovery and per-model routing was a
workaround for opencode serving some models via Anthropic-format endpoints.
But opencode also exposes OpenAI-compat for *all* models at `/v1`. Verify
this, then:
```python
OpenAIChatModel(
    '<model>',
    provider=OpenAIProvider(
        base_url='https://opencode.ai/zen/v1',  # or /zen/go/v1
        api_key=os.environ['OPENCODE_ZEN_API_KEY'],
    ),
)
```

**Option B (preserves Anthropic routing):** Build a factory that queries
`/v1/models`, determines the endpoint per model, and returns either
`OpenAIChatModel` or `AnthropicModel`. More code but preserves the current
behavior for models that only work via Anthropic endpoints.

**Recommendation:** Try Option A first. If specific models fail, fall back to
Option B only for those. The model discovery + streaming workaround is the
bulk of the complexity; if Option A works, ~300 lines delete.

### 4. Local Ollama — `OllamaModel` with `NativeOutput` ✅

Not a custom provider, but relevant: local Ollama should move from
`KnoxLiteLLMModel("ollama/...")` to native `OllamaModel`, which enables
`NativeOutput` for schema-constrained decoding. This is T6.

---

## What gets deleted

After T7:

| File/module | Lines | Status |
|-------------|-------|--------|
| `src/knox/llm/providers/__init__.py` | 169 | Delete |
| `src/knox/llm/providers/_base.py` | 160 | Delete |
| `src/knox/llm/providers/claudin_io.py` | 40 | Delete |
| `src/knox/llm/providers/ollama_cloud.py` | 38 | Delete |
| `src/knox/llm/providers/opencode.py` | 310 | Delete (or shrink to factory) |
| `src/knox/llm/providers/AGENTS.md` | — | Delete |
| `tests/unit/test_providers.py` | — | Rewrite as native-model tests |
| **Total** | **~720 lines** | |

Plus `KnoxLiteLLMModel` itself shrinks dramatically (T1, T3, T6 each remove
an override), or is replaced entirely by native model construction.

---

## What replaces it: a provider registry

Instead of litellm custom providers, build a small `knox.llm.models` registry
that maps KNOX model strings to native pydantic-ai `Model` instances:

```python
# src/knox/llm/models.py (proposed, ~80 lines)
def build_model(model_string: str, *, api_key: str | None = None) -> Model:
    """Build a pydantic-ai Model from a KNOX model string.

    Supported formats:
      ollama/<model>            -> OllamaModel (local)
      ollama-cloud/<model>      -> OpenAIChatModel at ollama.com/v1
      claudinio/<model>         -> OpenAIChatModel at api.claudin.io/v1
      opencode-zen/<model>      -> OpenAIChatModel at opencode.ai/zen/v1
      opencode-go/<model>       -> OpenAIChatModel at opencode.ai/zen/go/v1
      openai/<model>            -> OpenAIChatModel (native)
      anthropic/<model>         -> AnthropicModel (native)
      <other litellm prefixes>  -> LiteLLMModel (fallback for anything else)
    """
    prefix, _, name = model_string.partition('/')
    if prefix == 'ollama':
        return OllamaModel(name, provider=OllamaProvider(...))
    if prefix == 'ollama-cloud':
        return OpenAIChatModel(name, provider=OpenAIProvider(
            base_url='https://ollama.com/v1',
            api_key=api_key or os.environ.get('OLLAMA_API_KEY', ''),
        ))
    # ... etc
    # Fallback: any litellm-supported prefix
    return LiteLLMModel(model_string, api_key=api_key)
```

This keeps the `provider/model` string convention users know, while routing
to native models where possible and falling back to litellm for everything
else (OpenRouter, Together, Fireworks, etc. — all the providers litellm
supports natively that we never wrapped).

---

## Recommended sequence

1. **Immediate fix:** Restore provider registration so cloud configs work
   again. One-line: add `import knox.llm.providers` to `src/knox/llm/__init__.py`.
   This unblocks anyone using cloud configs today.

2. **Verify Option A for opencode:** Test whether `OpenAIChatModel` at
   `https://opencode.ai/zen/v1` works for all opencode models, or whether
   the Anthropic routing is load-bearing. If Option A works, the whole
   `opencode.py` (300 lines) deletes.

3. **Build `knox.llm.models.build_model()`** as the new entry point.

4. **Wire `KnoxAgentBase` to use `build_model()`** instead of
   `KnoxLiteLLMModel(model_string)`.

5. **Delete `knox.llm.providers/`** and the litellm custom-provider machinery.

6. **Simplify `KnoxLiteLLMModel`** — now only used as the litellm fallback.
   Most overrides (response_format, salvage, retry enrichment) either
   delete (because native models handle them) or move to `output_validator`.

---

## Open questions for the user

1. **Opencode Anthropic routing:** Do you know whether opencode-zen/go models
   that currently route to the Anthropic endpoint also work via the
   OpenAI-compat endpoint? If yes, Option A (delete the discovery logic).
   If no, keep a small factory.

2. **LiteLLM fallback:** Should `build_model()` keep `LiteLLMModel` as a
   fallback for prefixes we don't explicitly map (openrouter, together,
   fireworks, etc.)? Or should we add explicit mappings for every provider
   users might want?

3. **Local Ollama:** Switch local Ollama from litellm's `ollama/` prefix to
   native `OllamaModel`? This enables `NativeOutput` (T6) but changes the
   wire format slightly (pydantic-ai's Ollama handling vs litellm's).
