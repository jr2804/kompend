# AGENTS.md — Code

kompend code policies: antipatterns, boundaries, error handling. Load when writing
or reviewing code — production or tests.

## Python style

Follow the `python-ultimate` skill when in doubt about Python style, structure,
or conventions — it takes precedence over a conflicting habit or default.

## Antipatterns

| Pattern | Verdict | Fix |
|---|---|---|
| Hard-coded model name (`model="ollama/llama3.2"`) | ❌ | Config field default, never a function signature |
| Domain value as function default (`prompt_budget: int = 200_000`) | ❌ | Config dataclass field |
| `x: T | None = None` with hidden default in body | ❌ | Config field; docstring if `None` is truly meaningful |
| `content[:200]` / `text[:4096]` | ❌ | Full content, or a batch size derived from config/model |
| Silent exclusion by a config value (`candidates[:max_pages]`, `if len(x) > budget: drop`) | ❌ | Split the work into windows and warn — a budget shapes the work, it never excludes it (Boundaries) |
| Config fallback duplicating a runtime provider value (`dim = 384`) | ❌ | Read it from the provider at runtime (`provider.dim`) |
| Bare schema/field-name literals (`hit.get("source_file")`) | ❌ | `StrEnum` / `Final[str]` symbol imported at every site |
| `Literal["a", "b"]` as a value vocabulary | ❌ | a `StrEnum` with one member per value |
| Hard-coded `"KOMP_*"` env-var literal (`os.environ.get("KOMP_WIKI_DIR")`, `envvar=`) | ❌ | `EnvVar` member from `kompend.core.env` |
| Stable-shape dict without a schema (JSONL records, persisted entries — lint history, dedup store, graph edges, citations) | ❌ | `TypedDict` next to the owning model — types keys at zero runtime cost. Dataclasses only where the shape is also constructed/validated in code |
| Absolute path in a `.kompend/` artifact (`citing_source = str(path)`) | ❌ | Relativize at the persistence boundary (`rel_to_vault(path, base_dir)`) |
| `pip install` / `uv pip install` in code, docs or a comment | ❌ | `uv add` / `uv remove` / `uv lock` / `uv sync` — pip bypasses `uv.lock` (POLICIES, Boundaries) |
| `pip install …` / `uv pip install …` in docs, scripts or a fix hint | ❌ | `uv add` / `uv lock` / `uv sync` — a pip install bypasses `uv.lock` (`POLICIES.md`, Boundaries) |

The first five are instances of the universal magic-number rule
(see `POLICIES.md`); rows 6–9 are kompend-specific. Three of them are enforced:
`scripts/checks/check_field_consistency.py` (bare contract keys) and
`scripts/checks/check_antipatterns.py` (env-var literals, magic crops, pip
installs). Run both before committing — see `CHECKS.md`.

## Boundaries

- **Every behaviour-affecting value is a config field** in a `_KompendBaseModel`
  (`LLMConfig`, `BudgetConfig`, …) with a `Field(validation_alias=...)`. This
  holds even when the default equals the current literal: the field is the
  single source of truth, and `DEFAULTS` is generated from the schema.
- **No absolute paths in `.kompend/` artifacts.** Paths in every persisted record
  (provenance, references, assets, dedup, memory, search index, compile state,
  wiki cache) are vault-relative and POSIX-normalized. In-memory `Path` objects
  may be absolute; relativization happens at the persistence boundary.
- **No `from typing import TYPE_CHECKING`** and no `if TYPE_CHECKING:` blocks —
  they diverge runtime from the type checker. Break import cycles structurally
  (neutral module, Protocol, or a runtime-only import at its call site) and use
  `from __future__ import annotations` for forward references.
- **Never suppress** `PLC0415` (inline imports), `ANN001`/`ANN202` (missing
  annotations) — hard errors, no `# noqa`. For non-ruff scanner false positives
  (vibecheck, sloppylint, aislop) use that tool's own directive, so each
  suppression names the scanner it targets.
- **Heavy optional dependencies stay lazy** (parsecraft, torch, transformers,
  datasketch, scipy, crawl4ai, pydantic-ai, MCP SDK, zvec) so CLI startup stays
  fast. Package level: `lazy.attach(__name__, submod_attrs={...})`. Function
  level: the registry factory (`get_backend()`), never a module-scope import.
  Canonical pattern: `src/kompend/preprocessing/parsers/__init__.py`.
- **A budget shapes the work, it never excludes it.** A `budget.*` value may
  size one window (a request or a content block is *split*, never cropped),
  or set a threshold that warns. It never bounds a model's answer: no
  `max_tokens` is sent, and a planner's answer room comes from the model
  window (`ContextWindow.answer_reserve`). The single exception is a protocol
  that rejects an unset value (Anthropic): there the ceiling is computed from
  the window and the actual prompt, never read from config.
  It must never drop content, skip a document, or silently shrink a result
  set. The only legitimate stops are a provider/model timeout and an endless
  agent loop — and both keep the work done so far. Fix the loop; do not cap
  the budget.
- **Temp files** go in `./tmp` (gitignored) or `tempfile.TemporaryDirectory()`.

## Error handling — prefer "let it burn"

Propagate. `None` may mean a thing is **absent** (lookup miss, optional value
unset), never that an operation **failed** — returning `None` for a corrupt file
is the antipattern, and it pushes `if x is None` chains onto every caller.

The only valid `try/except` sites:

1. **Trust boundary** — translate a low-level error into a user-facing message
   at the CLI/MCP top level; catch narrowly, report the real cause.
2. **Optional-feature probe** — narrow to the specific exception
   (`ImportError`, `FileNotFoundError`), never `Exception`.
3. **Best-effort enrichment** — per-item catch that *records* the failure
   (SKIP result, warning log) and continues the batch.
4. **Cache miss** — narrow to `(json.JSONDecodeError, OSError)` and log at
   `WARNING` so corruption stays visible.

Forbidden: `except Exception: pass`, `except Exception: return None`,
`# noqa: BLE001`/`S110` to silence a bare except, and catching a corrupt
internal-state file (wiki cache, tier state, compile state) to reset it —
corruption is a bug; raise.

Removing a `try/except` that surfaces a real error to someone who needs it is
the correct fix.
