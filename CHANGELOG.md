# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `knox init` command — scaffold new wiki projects with standard directory structure
- `knox config get/set/list/path` subcommands — inspect and manage configuration
- Hierarchical project config discovery — walk up from CWD merging `.knox/config.toml` and `knox.toml` files (mise-style)
- Configurable user notes file (`NOTES.md`) — injected into LLM templates as `{{ user_notes }}`
- `[paths]` config section with `wiki_dir`, `sources_dir`, `output_dir`, `cache_dir`, `state_dir`
- `[notes]` config section with `file` key
- `KNOX_WIKI_DIR`, `KNOX_CACHE_DIR`, `KNOX_STATE_DIR`, `KNOX_NOTES_FILE` environment variables

### Changed

- CLI refactored: monolithic `commands.py` split into domain submodules (`pipeline_cmds`, `quality_cmds`, `query_cmds`, `capture_cmds`, `pack_cmds`, `serve_cmds`)
- Config precedence now supports hierarchical project config (multiple `.knox/config.toml` files merged)
- **Breaking:** `compile.chunk_size` default changed from `512` to `1024`. Previous default was silently ignored due to a parameter name mismatch (`chunk_size=` was passed to `**kwargs` instead of `max_tokens=`). Now that the config is actually wired, 1024 tokens produces better embedding quality for 768-dim models.

### Removed

- **Breaking:** Per-command `--sources-dir`/`-s` and `--output-dir`/`-o` flags removed from all commands. Use `--wiki-dir` (global) or `[paths]` config section instead.
- **Breaking:** `KNOX_SOURCES_DIR` and `KNOX_OUTPUT_DIR` env vars now map to `paths.sources_dir`/`paths.output_dir` (were `compile.sources_dir`/`compile.output_dir`). Deprecated `compile.sources_dir`/`compile.output_dir` keys are auto-migrated.
- **Breaking:** `HashEmbeddingProvider` removed from production code. Two embedding providers remain: `litellm:<model>` prefix (LiteLLM) and bare model name (sentence-transformers). Config default for `compile.embedding.model` is now `null` (vector indexing disabled until configured).

### Migration

If you used `--sources-dir` or `--output-dir`, switch to:

```bash
# CLI
knox --wiki-dir ./my-project compile

# Or in .knox/config.toml
[paths]
sources_dir = "my_docs"
output_dir = "output"
```
