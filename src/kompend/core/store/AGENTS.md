# DOX — kompend.core.store

<!-- on entry: loaded when agent enters src/kompend/core/store -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

The single non-vector store for a vault: one SQLite file (`.kompend/kompend.db`)
holding every flat `.kompend/` artifact that was previously loose JSON/JSONL —
including the vector branch's manifest row (`vector_manifest`), so `.kompend/`
holds one state file. The embedding branch itself (`vectors/`, `memory.db`)
stays separate and regenerable.

Design + table mapping live in `.agents/history/2026-09-24-kompend-storage-migration.md` §2.

## Modules

| Module | Contents |
|--------|----------|
| `tables.py` | Core `METADATA` — the schema's single source of truth; FTS5/trigger/index DDL via `AFTER_CREATE_DDL` |
| `schema.py` | Applies the metadata to a connection (`apply_schema`), `SCHEMA_VERSION`, table-name constants, `current_version`, `reconcile_columns` |
| `connection.py` | `connect`, `open_store`, `ensure_schema`, clean-cut recreate (`_drop_all`) |
| `legacy.py` | `obsolete_artifacts` — detect superseded flat files (not migration) |
| `__init__.py` | Public API re-exports |

## Local Contracts

- **`tables.py` (`METADATA`) is the only place schema exists.** Add or change
  tables there — never hand-write `CREATE TABLE`. Nullability (including
  SQLite's bare-`PRIMARY KEY` notnull-0 quirk), `AUTOINCREMENT`, and
  server-default conventions are documented in `tables.py`'s module docstring;
  `tests/unit/test_store.py` freezes the generated schema against the retired
  hand-written DDL. Reference tables through the `*_TABLE` constants — never
  inline a table literal. FTS5/trigger/index DDL that has no Core object rides
  `AFTER_CREATE_DDL` (fires on every `create_all`, so repair-on-reopen holds).
- **Core DDL reaches a raw connection through `schema._core_engine`.** The
  throwaway engine binds the caller's `sqlite3` connection and is never
  `dispose()`d — disposing would close the caller's handle. Consumers still
  receive a raw `sqlite3.Connection` from `open_store`.
- **Additive changes ride `reconcile_columns`; a shape change needs a
  migration.** New columns/tables are picked up on the next open with no
  version bump. A primary-key change cannot be reconciled, so it is migrated
  in place instead of dropping the store — `migrate_tier_state_identity`
  (`schema.py`) is the one example: `tier_state` moved from a bare-stem key to
  the content-addressed `<stem>-<hash8>` identity
  (`kompend.core.identity.source_key`) and kept every row's tier, hit counters
  and `ingested_by`. A `SCHEMA_VERSION` bump remains reserved for incompatible
  changes.
- **Clean-cut only (policy B3).** A version mismatch drops and recreates the
  store. There are **no** importers and **no** legacy readers — old
  `.kompend/*.json*` files are never read. `legacy.py` only *reports* them for
  `kompend doctor` (Phase 2).
- **One opener.** All access goes through `open_store` / `connect` so PRAGMAs
  (WAL, `foreign_keys`, `busy_timeout`) and the version check always apply.
- **Row per record.** Typed columns for queried fields; a `TEXT` JSON payload
  for nested/graph content. FTS5 only where the design doc says it earns its
  place (`memory_log`; not `wiki_snapshot`).
- `references` is a SQLite keyword — the table is `bib_references`.
- **The vector manifest row names the embedding model** (`embedding_model`).
  Vectors of one model are not vectors of another, so the store must know which
  model wrote them. The column is additive: `reconcile_columns` adds it in
  place, no `SCHEMA_VERSION` bump (a bump would drop the whole store).
- Paths come from `kompend.core.paths` (`VaultPaths.kompend_db_file`); pathlib only.

## Verification

- `uv run pytest tests/unit/test_store.py -q`
- `uv run ruff check src/kompend/core/store/` and `uv run ty check src/kompend/core/store/`
