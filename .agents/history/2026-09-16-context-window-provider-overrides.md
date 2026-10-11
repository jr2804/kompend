# Plan — `[providers.<name>]` context_window overrides must win over auto-discovery

## 1. Problem

`knox doctor` reports the wrong context window for `ollama/*` models when the
user defines a provider override in `~/.config/knox/config.toml`:

```toml
[providers.ollama]
context_window = 65536        # intended global override
```

Observed:

```text
[OK] context-window-llm: llm: 131072 in / 131072 out tokens (source: ollama)  ← ignored user override
[!] Custom providers: ollama: missing base_url                                  ← misclassified override-only block
```

## 2. Root cause

Two related issues in the context-window and provider-check layers:

### 2a. Override precedence wrong in `discover_context_window`

`src/knox/llm/context_window.py` runs the Ollama live probe first (line 90):

```python
if model.startswith("ollama/"):
    ollama_ctx = _query_ollama_num_ctx(model.removeprefix("ollama/"))
    if ollama_ctx is not None:
        max_input = ollama_ctx
        max_output = ollama_ctx
        source = ContextSource.OLLAMA
```

The user-provider check (step 1b) only fires `if max_input is None`, so any
`[providers.ollama].context_window` is silently shadowed.

### 2b. Override-only blocks misclassified in `knox doctor`

`src/knox/cli/cmds/quality/doctor.py::_check_custom_providers` requires
`base_url` for every `[providers.<name>]` entry it iterates over. Provider
override blocks are documented to omit `base_url` intentionally — the check
errors out anyway:

```python
base_url = prov_cfg.get("base_url", "") if isinstance(prov_cfg, dict) else ""
if not base_url:
    failures.append(f"{prov_name}: missing base_url")
    continue
```

## 3. Architecture review (user's question)

The user asked whether the four discovery paths (Ollama probe, models-dev catalog,
user-provider overrides, fallback) are duplicating logic. **No** — each path
covers a distinct case:

| Path | Covers | Why separate |
|------|--------|--------------|
| User-provider override | explicit user-config (finest granularity) | wins over auto-discovery; per-model + per-provider keys |
| Ollama live probe | Ollama-specific runtime reality (server-loaded slot, env var) | Ollama has no models-dev entry; GGUF metadata is aspirational vs. KV memory |
| models-dev catalog | cloud / OpenAI-compatible mirrors | static metadata; only source for non-Ollama cloud providers |
| `[compile.llm]` fallback | last-resort when nothing else resolves | off by default; user must opt in |

The duplication is **not** in the discovery layer — it's in `build_model`
(`_build_ollama` calls `_query_ollama_num_ctx` to set per-request `num_ctx`).
That's an intentional second use: discovery tells the agent how big to size
prompts, per-request setting tells the server how big a slot to allocate.
Different concerns, same query, both legitimate.

**Conclusion:** keep all four paths. Fix the precedence and the doctor
classifier — that's all.

## 4. Solution

### 4a. Reorder discovery so user-provider overrides win

Reorder `discover_context_window` so the user-provider block runs **before**
the Ollama and catalog probes. Rationale: the user explicitly wrote a
`context_window` value, that beats a heuristic auto-probe every time.

Order becomes:

1. User-provider override (per-model → provider-level → legacy `context_window_cap`)
2. Constitutional source (Ollama live / models-dev catalog) — what the
   model really supports
3. Explicit `[compile.llm]` fallback (`fallback_max_input` / `fallback_max_output`)
4. Raise `LookupError` when nothing resolved (already implemented)

Add a `ContextSource.USER_PROVIDER = "user_provider"` enum value so the
doctor can show *which* layer supplied the value.

### 4b. Teach `_check_custom_providers` about override-only blocks

A `[providers.<name>]` section is an **override block** (not a standalone
endpoint definition) when:

- `base_url` is missing or empty, AND
- the provider id is one of the natively supported ids (pydantic-ai +
  models-dev)

In that case the check should report success (override registered) and
**not** ping an endpoint. Standalone provider entries without `base_url`
remain an error — that's a misconfiguration.

Source of truth for "native provider id" already exists:
`_discover_provider_ids()` in `src/knox/llm/models.py` (line ~227). It's
module-private; expose it (or its semantic check) so the doctor can use it.

## 5. Open questions

None — the design decisions are constrained by existing memory rules:

- **#44** (StrEnum / Final for cross-file keys): `ContextSource.USER_PROVIDER`
  already lives in an enum — no new string literal.
- **#15** (`config set` errors when no vault): n/a here.
- **#7** (`--wiki-dir` / `--config` precedence): n/a — this is a pure logic fix.

## 6. Implementation plan

### Files to change

| File | Change |
|------|--------|
| `src/knox/llm/context_window.py` | Reorder `discover_context_window`: user-provider first; add `ContextSource.USER_PROVIDER` |
| `src/knox/cli/cmds/quality/doctor.py` | `_check_custom_providers` accepts override-only blocks for native providers |
| `src/knox/llm/models.py` | Expose `_discover_provider_ids()` as public `discover_provider_ids()` |
| `src/knox/llm/AGENTS.md` | Update resolution order section |
| `tests/unit/test_llm_context_window.py` | New: override beats Ollama probe + beats catalog |
| `tests/unit/test_cli_doctor_status.py` | New: override-only `[providers.ollama]` passes check |

### Verification

1. `uv run pytest tests/unit/test_llm_context_window.py tests/unit/test_cli_doctor_status.py -v`
2. `uv run pytest tests/unit/test_usage_tracking.py -v` (override-only block regression)
3. Manual: edit `~/.config/knox/config.toml`, add `[providers.ollama].context_window = 65536`,
   run `knox doctor`, confirm `context-window-llm` reports `65536` and `Custom providers`
   reports success without `missing base_url`.
4. Update `tools.field_consistency` if any new cross-file keys introduced (none expected).

## 7. Out of scope

- Changing `_query_ollama_num_ctx` precedence (env var > ollama.ps > Modelfile > GGUF) — correct as-is.
- Changing `_build_ollama`'s use of `_query_ollama_num_ctx` for per-request `num_ctx` — distinct concern.
- Renaming `context_window_cap` (deprecated key) — already migrated to `context_window`.
- Adding a `[providers.<name>.models.<model>.num_ctx]` override — punt until requested.
