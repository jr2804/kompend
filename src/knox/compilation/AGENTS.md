# DOX — knox.compilation

<!-- on entry: loaded when agent enters src/knox/compilation -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Tier 4 compilation — build an interlinked knowledge graph (`CompiledWiki`)
from resolved chunks. Two-phase extraction-then-generation with cross-source
merging and an optional iterative WiCER refinement loop.

## Ownership

- `src/knox/compilation/` — compilers, graph construction, schema, checkpoints

## Architecture

| Module                  | Contents                                                                 |
| ----------------------- | ------------------------------------------------------------------------ |
| `base.py`               | `BaseCompiler` ABC — compiler interface with progress hook               |
| `two_phase.py`          | `TwoPhaseCompiler` — extract-first then generate with cross-source merge |
| `_two_phase_core.py`    | Core extract/generate loops extracted from `two_phase.py`                |
| `wicer.py`              | `WiCERCompiler` — iterative correctness/entity/consistency refinement    |
| `graph.py`              | `GraphCompiler` — dependency-graph-aware compilation                     |
| `schema.py`             | Wiki graph schema: predicates, entity types, relation types              |
| `_checkpoints.py`       | Compilation checkpoint/restore                                           |
| `flows/`                | Flow-based compilation building blocks                                   |
| `flows/assembly.py`     | Cross-source assembly flow                                               |
| `flows/extraction.py`   | Extraction flow                                                          |
| `flows/generation.py`   | Generation flow; `build_known_titles` (canonical titles for wikilinks), `learn_wikilink_aliases` (post-generation alias map persisted as `MetadataField.WIKILINK_ALIASES`) |
| `flows/references.py`   | Reference-extraction flow (pre-extraction pass)                          |
| `reference_excision.py` | Reference-section detection + excision (regex-based)                     |

## Rules

- Import from the package root: `from knox.compilation import GraphCompiler`,
  `from knox.compilation.wicer import WiCERCompiler`.
- `two_phase.py` is the primary entry point; `_two_phase_core.py` holds
  extracted core logic to keep the main module focused.
- `WiCERCompiler` wraps a `TwoPhaseCompiler` and adds iterative probing.
- No prompt text belongs here — templates live in `src/knox/templates/`.

### Page kinds (category taxonomy)

Every generated page carries a `kind` (OKF `type`): the **mandatory taxonomy**
(`concept`, `entity`, `summary`, `comparison` — `MANDATORY_KINDS` in
`core/config/keys.py`) is always allowed and cannot be deactivated; extra
kinds come from `compile.extraction.categories`. The LLM assigns `kind` per
page in `generation_prompt.md.j2`; `_allowed_kind` in `flows/generation.py`
validates it at node creation — anything outside the allowed set falls back
to `concept` (permissive-consumption model, never a hard failure).
Exception: extracted entities become mandatory pages — `ExtractionSet.entity_pages()`
merges them with a pre-assigned `kind: entity`, and `_resolve_page_kind` in `flows/generation.py`
overrides the LLM's label for those entries. Entity specs colliding with an existing
concept key are dropped to avoid duplicate pages.
