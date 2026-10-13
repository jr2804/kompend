# AGENTS.md — Files

Path constants, VaultPaths, and naming conventions. Load when touching files.

## General Files

### Pattern: Single Source of Truth for Paths

Project artifact paths should be defined in a single constants module. No other
module hard-codes directory or file names. Always import from the constants
module.

### Pattern: Resolved Path Object

Use a path-resolver class that takes a base directory and exposes every artifact
path as a property. This ensures consistent path construction across the codebase.

### Extension

| Who | When | What |
|-----|------|------|
| Project maintainer | Adding a new artifact file/directory | Add constant row |
| Project maintainer | Renaming an artifact | Update constant value |
| Project maintainer | Changing the paths module | Update module reference |

---

## Project-specific Files — KOMPEND

### Path Constants — Single Source of Truth

All on-disk artifact names are `Final[str]` constants in **`src/kompend/core/paths.py`**.

**Rule:** `".kompend"`, `"cache"`, `"config.toml"`, `"vectors"`, `"memory.db"`,
`"last_wiki.json"`, etc. must **never** appear as a literal in
any `.py` file outside `paths.py`. Always import from `kompend.core.paths`.

| Constant | Value | Usage |
|----------|-------|-------|
| `KOMP_DIR` | `".kompend"` | Root metadata directory |
| `KOMP_DB_FILE` | `"kompend.db"` | Single non-vector SQLite store (`.kompend/kompend.db`) |
| `CACHE_SUBDIR` | `"cache"` | Cache component within `KOMP_DIR` |
| `CACHE_DIR` | `".kompend/cache"` | Full relative cache path (for DEFAULTS) |
| `CONFIG_FILE` | `"config.toml"` | Project-local config |
| `VECTORS_DIR` | `"vectors"` | Embedding vector store |
| `VECTORS_MANIFEST_FILE` | `"vectors.manifest.json"` | Vector id sidecar (legacy — now the `vector_manifest` row) |
| `DOWNLOADED_FILE` | `"downloaded.json"` | URL download manifest (legacy — now `downloaded` table) |
| `MEMORY_DB_FILE` | `"memory.db"` | Semantic query-memory collection **directory** (zvec, not SQLite) |
| `MEMORY_LOG_FILE` | `"memory.jsonl"` | Memory append-only log (legacy — now `memory_log` table) |
| `MEMORY_STORE_FILE` | `"memory.json"` | Memory state export (legacy — now `memory_states` table) |
| `PROVENANCE_DIR` | `"provenance"` | Provenance trace directory (legacy — now `provenance` table) |
| `SUMMARIES_SUBDIR` | `"summaries"` | Per-source summary pages (`wiki/summaries/`) |
| `EXPORTED_DIR_NAME` | `"exported"` | Full-source markdown exports (top-level, on demand) |
| `QUALITY_FILE` | `"quality.md"` | Verification summary page (`wiki/quality.md`) |
| `VERIFICATION_FILE` | `"verification.json"` | Verification results |
| `PRIVATE_FILE` | `"private.json"` | Private page manifest |
| `COMPILE_STATE_FILE` | `"compile_state.json"` | Tier state checkpoint |
| `LINT_HISTORY_FILE` | `"lint_history.jsonl"` | Append-only lint history log |
| `STATE_FILE` | `"state.json"` | General state file |

**Configurable directory names** (`"sources"`, `"wiki"`) live in `DEFAULTS`
under `["paths"]["sources_dir"]` and `["paths"]["output_dir"]` — not in
`paths.py`, because users can override them.

### Project root naming

The project root directory is always called `base_dir`. Never use `target_dir`,
`resolved_base`, `project_root`, or `sources_dir.parent`.

### VaultPaths

`VaultPaths` (in `paths.py`) resolves concrete paths inside a vault:

```python
from kompend.core.paths import VaultPaths

paths = VaultPaths(base_dir)
paths.cache_dir  # base_dir/.kompend/cache
paths.compile_state_file  # base_dir/.kompend/compile_state.json
paths.wiki_cache_file  # base_dir/.kompend/last_wiki.json
paths.extract_dir  # base_dir/.kompend/cache/extract (legacy — now extract cache layer in kompend.db)
paths.images_dir  # base_dir/.kompend/cache/images
```

| Context | Pattern |
|---------|---------|
| Module has `config: PipelineConfig` | `config.paths.cache_dir` |
| Module has `base_dir: Path` | `VaultPaths(base_dir).cache_dir` |
| Module has `output_dir: Path` | `VaultPaths(output_dir.parent).cache_dir` |
| Test with `tmp_path` | `VaultPaths(tmp_path).cache_dir` |

### Test conventions

- All tests use `pytest` and `tmp_path` for filesystem operations.
- Unit tests must mock external dependencies (LLM, network, filesystem).
- Integration tests calling LLM APIs must use `@pytest.mark.skipif` checking
  for API key availability.
- Test files follow `test_<module>.py` naming.
