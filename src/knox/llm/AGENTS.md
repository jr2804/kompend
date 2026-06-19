# DOX — knox.llm

## Purpose

LLM client layer. All LLM calls go through `LLMClient` — never import litellm directly in business modules.

Custom litellm providers live in `knox.llm.providers/`.

## Provider submodule

`src/knox/llm/providers/` registers providers that litellm does not ship natively. To avoid having to set `OPENAI_BASE_URL` or `OPENAI_API_KEY` (which conflicts with real OpenAI usage), this submodule lets you implement your own custom providers.

Even though litellm supports a vast number of providers out of the box, some very obvious ones are still missing:

- **Ollama Cloud** — `ollama-cloud/<model>` routes to `https://ollama.com`
- **OpenCode Go** — `opencode-go/<model>` routes to `https://opencode.ai/zen/go`
- **OpenCode Zen** — `opencode-zen/<model>` routes to `https://opencode.ai/zen`
- **Claudin.io** — `claudin-io/<model>` routes to `https://api.claudin.io`

### Adding a new provider

1. Create a new Python module in `providers/` (e.g., `my_provider.py`)
2. Define a class inheriting from `litellm.CustomLLM` that implements the four methods: `completion`, `streaming`, `acompletion`, `astreaming`
3. Import the class in `providers/__init__.py` and add it to the `_register_custom_providers()` list

Providers auto-register on `import knox.llm.providers` (which happens at the top of `knox.llm.client`).

### Module structure

| File | Contents |
|------|----------|
| `__init__.py` | Re-exports, `_register_custom_providers()`, auto-registration |
| `claudin_io.py` | `_ClaudinIOProvider` — Anthropic-compat via <https://api.claudin.io> |
| `ollama_cloud.py` | `_OllamaCloudProvider` — Ollama cloud API |
| `opencode.py` | `_OpenCodeProvider` base + `*ZenProvider`, `*GoProvider`, `*DummyProvider` |

### Test hint

Provider registration is a side-effect of importing `knox.llm.providers`. Tests should mock the `httpx.get` call in `_discover_models` and set the relevant API key env var (or mock `_api_key`) before calling discovery methods.

## Verification

- `uv run pytest tests/unit/test_providers.py -x` — provider tests pass
- `uv run python -c "import knox.llm.providers"` — auto-registration fires without error

---

## Child DOX Index

*This module has no child AGENTS.md files.*
