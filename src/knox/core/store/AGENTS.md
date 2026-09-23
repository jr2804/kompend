# DOX — knox.core.store

<!-- on entry: loaded when agent enters src/knox/core/store -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

The single non-vector store for a vault: one SQLite file (`.knox/knox.db`)
holding every flat `.knox/` artifact that was previously loose JSON/JSONL.
The embedding branch (`vectors/`, `memory.db`) stays separate and regenerable.

Design + table mapping live in `.agents/plans/knox-storage-migration.md` §2.

## Modules

| Module | Contents |
|--------|----------|
| `schema.py` | DDL, `SCHEMA_VERSION`, all table-name constants, `apply_schema`, `current_version` |
| `connection.py` | `connect`, `open_store`, `ensure_schema`, clean-cut recreate |
| `legacy.py` | `obsolete_artifacts` — detect superseded flat files (not migration) |
| `__init__.py` | Public API re-exports |

## Local Contracts

- **`schema.py` is the only place DDL exists.** Add or change tables there, bump
  `SCHEMA_VERSION`, and reference tables through the `*_TABLE` constants — never
  inline a table literal.
- **Clean-cut only (policy B3).** A version mismatch drops and recreates the
  store. There are **no** importers and **no** legacy readers — old
  `.knox/*.json*` files are never read. `legacy.py` only *reports* them for
  `knox doctor` (Phase 2).
- **One opener.** All access goes through `open_store` / `connect` so PRAGMAs
  (WAL, `foreign_keys`, `busy_timeout`) and the version check always apply.
- **Row per record.** Typed columns for queried fields; a `TEXT` JSON payload
  for nested/graph content. FTS5 only where the design doc says it earns its
  place (`memory_log`; not `wiki_snapshot`).
- `references` is a SQLite keyword — the table is `bib_references`.
- Paths come from `knox.core.paths` (`VaultPaths.knox_db_file`); pathlib only.

## Verification

- `uv run pytest tests/unit/test_store.py -q`
- `uv run ruff check src/knox/core/store/` and `uv run ty check src/knox/core/store/`

## Child DOX Index

None.
