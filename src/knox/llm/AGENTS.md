# DOX — knox.llm

## Purpose

LLM client layer. All LLM calls go through `LLMClient` — never import litellm directly in business modules.

Custom litellm providers live in `knox.llm.providers/`.

## Provider submodule

`src/knox/llm/providers/` registers providers that litellm does not ship
natively. See `providers/AGENTS.md` for the full rules, patterns, and
test requirements. Brief summary:

- No hard-coded model names — use generic `<model>` in docs and `test-model` in tests.
- If the provider supports model discovery, use it; otherwise pass the user's model through as-is.
- Two patterns: **thin wrapper** (no discovery, known API format) or **discovery-based** (opencode.py style).
- Register in `_register_custom_providers()`, test all 4 methods, add tests to `tests/unit/test_providers.py`.

### Test hint

Provider registration is a side-effect of importing `knox.llm.providers`. Tests should mock the `httpx.get` call in `_discover_models` and set the relevant API key env var (or mock `_api_key`) before calling discovery methods.

### Error handling

Two broad `except Exception` sites in this package are **deliberate**, not
sloppy — see the "Error handling" rule in `src/knox/AGENTS.md`:

- `failover.py` `FailoverChain.run`: catches everything around the arbitrary
  `call(model)` and routes it through `classify_litellm_error` (guaranteed
  fallthrough → `TransientError`). Error-hierarchy trust boundary.
- `client.py` model-introspection: `litellm.get_model_info` raises bare
  `Exception` for unmapped models (its own API), so a narrow catch is
  impossible. Optional-feature probe, logged at WARNING.

Do not "clean up" these broad catches — they are required by the upstream
APIs. If a narrow base class becomes available (e.g. litellm ships
`LiteLLMException`), narrow then.

## Verification

- `uv run pytest tests/unit/test_providers.py -x` — provider tests pass
- `uv run python -c "import knox.llm.providers"` — auto-registration fires without error

---

## Child DOX Index

| Path | Scope |
|------|-------|
| `providers/AGENTS.md` | Custom litellm provider rules, patterns, registration, testing |
