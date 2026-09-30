# AGENTS.md — Onboarding

First 5 minutes to get oriented.

## General Onboarding

This file follows a reusable template. Every project should cover:

1. **What to read first** — the AGENTS.md chain: read root, then subtree
   files for directories you'll touch.
2. **Key entry points** — 5-10 files that give the fastest orientation.
3. **Project structure** — one-line purpose per top-level directory.
4. **Essential commands** — build, test, lint, typecheck with time estimates.
5. **Key conventions** — the 5-8 rules you'll break most often if you don't
   know them.
6. **Where to dig deeper** — documentation index, issue tracker, child DOX docs.

### Extension

| Who | When | What |
|-----|------|------|
| Project maintainer | After major code reorganization | Sections 2-3 |
| Project maintainer | When build/test tooling changes | Section 4 |
| Project maintainer | When key conventions change | Section 5 |

**Never duplicate content.** Reference the source file. This document is a map,
not a rulebook — rules live in `POLICIES.md`, path constants in `FILES.md`,
verification in `CHECKS.md`.

---

## Project-specific Onboarding — KNOX

### 1. What is KNOX?

KNOX = Knowledge Organizer for eXtended documents. A parser-agnostic,
structure-preserving, LLM-aware knowledge base compiler. 5-tier pipeline:
Preprocessing → Chunking → Resolution → Compilation → Output.

Every LLM call routes through PydanticAI agents in `src/knox/agents/`.

### 2. Key entry points (read these first)

| File | Why |
|------|-----|
| `AGENTS.md` | Root rail — index to all `.agents/` files |
| `.agents/POLICIES.md` | Priorities, boundaries, DOX authoring, response format |
| `.agents/CODE.md` | Code antipatterns, boundaries, error handling |
| `.agents/FILES.md` | Path constants, VaultPaths, naming |
| `src/knox/AGENTS.md` | Python-specific: text-in-code, antipatterns |
| `docs/index.md` | User-facing documentation index |
| `knox.toml.example` | All config keys with comments |
| `src/knox/core/paths.py` | Single source of truth for artifact paths |
| `src/knox/core/config/generate.py` | `DEFAULTS` dict — all config defaults |

### 3. Project structure

| Path | Purpose |
|------|---------|
| `src/knox/core/` | Foundation: config, models, paths, pipeline, cache, templates |
| `src/knox/agents/` | PydanticAI agents for every LLM call |
| `src/knox/llm/` | LLM client, provider registration, failover |
| `src/knox/cli/` | Typer-based CLI (thin shell over core) |
| `src/knox/preprocessing/` | Parsers (Docling, Pandoc, LiteParse, Web) |
| `src/knox/compilation/` | Tier 4: extraction, generation, WiCER, output |
| `src/knox/search/` | Hybrid search: FTS + vector + RRF |
| `src/knox/memory/` | Memory lifecycle: working → episodic → semantic → procedural |
| `tests/` | Unit, integration, benchmarks |
| `docs/` | MkDocs user-facing documentation |
| `.agents/skills/` | Installed agent skills |

### 4. Essential commands

```bash
uv run pytest tests/unit -x --timeout=30          # unit tests
uv run pytest tests/unit -q --no-cov --tb=line     # unit tests (compact)
uv run pytest tests/integration -x --timeout=120   # integration tests
uv run pytest tests/benchmarks/ -q --no-cov        # benchmarks
ruff check                                           # lint
ty check src/knox                                    # type check (not mypy)
uv run python scripts/checks/check_field_consistency.py     # bare-string contract keys
uv run python scripts/checks/check_antipatterns.py          # banned-noqa, magic crops, global
uv run mkdocs build --strict                        # doc build check
```

### 5. Key conventions

- **`uv run`** for everything — never `python`, `python3`, `pip`
- **`base_dir`** — canonical name for the project root directory
- **`paths.py`** — all artifact names live here; never hard-code `.knox`, `cache`, etc.
- **`VaultPaths`** — resolve concrete paths with `VaultPaths(base_dir)`
- **StrEnum** — roles, tasks, compilers, tiers must use enum members, not bare strings
- **No tree figures** in AGENTS.md — regenerate: `rg --files | tree-cli --fromfile`
- **No cross-level duplication** in DOX chain — reference, never copy

### 6. Where to dig deeper

- Each `src/knox/` subdirectory has its own `AGENTS.md` (DOX child doc)
- `docs/index.md` — full documentation index
- `.agents/HISTORY.md` — past design decisions, known issues
- `bd` (beads) — issue tracking: `bd list`, `bd show <id>`
