# DOX — knox.llm.providers

## Purpose

Custom litellm provider implementations for KNOX. Each provider routes a custom
prefix (e.g., `claudinio/`, `ollama-cloud/`, `opencode-zen/`) to a third-party
API through litellm's `CustomLLM` interface.

## Rules

### No hard-coded model names

Never hard-code a real model identifier (e.g. `claude-3.5-sonnet`, `llama3.2`,
`deepseek-v4-flash`) in provider code, docstrings, or examples. Use a generic
`<model>` placeholder when documentation needs an example.

Same rule applies to tests — use a constant like `_TEST_MODEL = "test-model"`.

### Model source priority

1. **Provider supports model list** (e.g. `GET /v1/models`): implement
   `_discover_models()` and use the returned list to route per-model to the
   correct endpoint format. See `opencode.py` for the pattern.
2. **Provider does NOT support model list**: accept whatever model name the
   user passes. Do NOT guess, hard-code, or maintain a static model list.

### Provider prefix must match models.dev (when listed)

KNOX discovers context windows from `models.dev/catalog.json` (see
`src/knox/llm/models_dev.py`). When a provider is listed on models.dev,
the registration prefix must match their provider name exactly — e.g.
`claudinio/`, not `claudin-io/` — so the catalog matcher can resolve it.

The catalog covers ~150+ providers and ~235 models at time of writing.
If a provider is listed on models.dev, it is almost always in the catalog.
When a model is NOT in the catalog (unlisted or very new), context window
falls through to the configured `fallback_max_input`. Set
`context_window_cap` or `fallback_max_input` in config when the fallback
doesn't match your model's actual context window.

### Architecture patterns

Two patterns are established. Pick the one that fits:

| Pattern | When to use | Examples |
|---------|-------------|---------|
| **Thin wrapper** | API uses a known format (Anthropic-compat, Ollama-compat). Just delegate with the right prefix + `api_base` + `api_key`. | `claudin_io.py`, `ollama_cloud.py` |
| **Discovery-based** | API supports `/v1/models` and may have per-model endpoint format requirements. | `opencode.py` + `_OpenCodeProvider` |

### Registration

1. Create a new Python module in `providers/` (e.g., `my_provider.py`)
2. Define a class inheriting from `litellm.CustomLLM` implementing all four
   methods: `completion`, `streaming`, `acompletion`, `astreaming`
3. Import the class in `providers/__init__.py` and add it to the
   `_register_custom_providers()` list

Providers auto-register on `import knox.llm.providers`.

### Tests

Every provider must have tests in `tests/unit/test_providers.py` covering:

- Registration: provider is auto-registered on import
- `_delegate_kwargs` format: correct model prefix and `api_base`
- API key env var: reads from env, returns empty when unset
- All four delegation methods: mock `litellm.completion` / `litellm.acompletion`
  and verify correct kwargs are passed

## File inventory

| File | Contents |
|------|----------|
| `__init__.py` | Re-exports, `_register_custom_providers()`, auto-registration |
| `claudin_io.py` | `_ClaudinIOProvider` — OpenAI-compat API at <https://api.claudin.io/v1> (prefix: `claudinio/`, alias: `claudin-io/`) |
| `ollama_cloud.py` | `_OllamaCloudProvider` — Ollama cloud API |
| `opencode.py` | `_OpenCodeProvider` base + `*ZenProvider`, `*GoProvider`, `*DummyProvider` |
