# Compilation Guide

KNOX compiles documents into a knowledge wiki through a five-tier pipeline.
This guide explains the compilation modes, two-phase execution, prompt
budgeting, and per-source state.

## Pipeline tiers

| Tier | Name | What runs | Cost | Default |
|------|------|-----------|------|---------|
| 0 | Index | Parse + persist state | Free | ✅ |
| 1 | Chunk | Index + chunk documents | Free | ✅ |
| 2 | Resolve | Chunks → entities / references | Free | ✅ |
| 3 | Compile | Resolve → wiki pages | ~$0.05/doc | ✅ |
| 4 | Output | Write wiki to disk | Free | ✅ |

## Compilation modes

Use `--mode` to skip tiers you don't need:

```bash
knox compile *.pdf                           # full
knox compile *.pdf --mode index-only         # re-index only
knox compile *.pdf --mode resolve-only       # skip re-parsing
knox compile *.pdf --mode output-only        # re-render only
```

| Mode | Tiers run |
|------|-----------|
| `full` | 0, 1, 2, 3, 4 |
| `index-only` | 0, 1 |
| `resolve-only` | 2, 3, 4 |
| `output-only` | 4 |

The pipeline loads its tier checkpoint from `.knox/compile_state.json`
inside the output directory. On restart, unchanged sources are skipped.

## Two-phase compilation

The default compiler is `TwoPhaseCompiler` (N3). It runs:

1. **Extraction pass** — one LLM call per source, returns structured JSON
   (`{summary, entities, concepts, tags}`).
2. **Generation pass** — concepts from all sources are merged by name and
   one page is generated per unique concept. Cross-source edges are added
   for shared entities.

To use the simpler one-pass mode:

```bash
knox compile *.pdf --single-pass
```

## Prompt budget

Each concept page in phase 2 sees at most `--prompt-budget` characters
of source content (default 200 000). When a concept draws from many
sources, the content is **fair-share truncated** — each source gets
an equal slice, not first-N-chunks. Excess truncation is logged.

```bash
knox compile *.pdf --prompt-budget 100000
```

## Per-source state

KNOX persists a tier checkpoint for every source file:

```json
{
  "entries": {
    "paper": {
      "stem": "paper",
      "source_path": "sources/paper.pdf",
      "content_hash": "a1b2c3...",
      "current_tier": 3,
      "last_compile": "2026-06-08T12:34:56+00:00",
      "search_hits": 0
    }
  }
}
```

The file lives at `<output>/.knox/compile_state.json`. Delete it (or
pass `--force`) to force a full recompile.

## Tier listing

```bash
knox compile *.pdf --list-tiers
```

Prints a JSON list of every tier with the count of sources that have
reached it. Useful in CI for "how complete is this wiki?" checks.

## FAQ

**Q: How do I re-compile a single source?**
A: `knox compile sources/foo.pdf --force`

**Q: Why is my source at tier 0 after a failed compile?**
A: The tier is updated only after the chunk succeeds. Re-run `knox compile`.

**Q: How does merge work on re-ingest?**
A: In two-phase mode, entities/concepts from the new source are merged
with existing pages — old content is preserved, new provenance is added.

**Q: Can I run just the LLM stages?**
A: Yes: `knox compile *.pdf --mode resolve-only` skips re-parsing.

**Q: How do I get a JSON status report?**
A: `knox compile *.pdf --list-tiers` outputs JSON.
