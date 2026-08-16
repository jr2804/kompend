# Category extraction: mandatory taxonomy + configurable categories

Created: 2026-08-16
Status: **implemented** (2026-08-16). All phases 0-4 done; 1691 unit tests pass, ruff at baseline.

## Problem (verified in code)

The two-phase compile produces **only `concept` pages**. Verified:

- `src/knox/compilation/flows/generation.py:331` hardcodes `node_type="concept"` for every generated page.
- `src/knox/compilation/schema.py:88` — `GenerationPage` has `{title, slug, content, provenance, related}`; **no kind/type field** — the LLM is never asked for one.
- `src/knox/templates/compilation/generation_prompt.md.j2` asks for "one comprehensive wiki page per concept".
- `src/knox/compilation/graph.py:173` has an `_infer_page_kind` heuristic (overview/comparison/entity) but it's only reachable in the single-phase flow; `graph.py:410` hardcodes `"concept"` in the other path.
- Evidence: p863 `last_wiki.json` — 64/64 nodes `node_type == "concept"`; index shows a single `## Concept` group.

Karpathy: the wiki index is "organized by category (entities, concepts, sources, etc.)"; OKF §4.1 requires a non-empty `type` but leaves the taxonomy producer-defined. Both expect multiple page kinds; KNOX emits one.

(Note: the old "30%" lint score was rumdl style noise, not knowledge completeness — every cached node *is* on disk. The gap is category *richness*, not persistence.)

## Design

### Mandatory taxonomy (cannot be deactivated)

| kind | When the LLM must emit it | Page content contract |
|---|---|---|
| `concept` | default — any topic/idea/approach | current behavior (unchanged) |
| `entity` | a named thing: standard (P.863, POLQA), body (ITU-T), organization, product, person | canonical definition, aliases, role in sources |
| `summary` | one digest per significant source (or per clearly-delimited topic cluster) | what the source contributes, key claims, relation to other pages |
| `comparison` | page that evaluates alternatives ("X vs Y", "A versus B", criteria-based) | dimensions compared, verdict, per-side evidence |

The LLM assigns `kind` per page; invalid/unknown values fall back to `concept` (no hard failure — matches the permissive-consumption model, §10).

### Configurable extras

```toml
[compile.extraction.categories]
# name → instruction injected into the generation prompt; the LLM may tag
# pages with these kinds in addition to the mandatory set.
procedure = "Step-by-step how-to pages: prerequisites, numbered steps, verification."
```

- Keys must not shadow the mandatory taxonomy (validation error at config load).
- Empty/missing section → mandatory set only.
- Each configured category becomes a prompt instruction + an allowed `kind` value.

### Why these four

- **concept/entity/summary/comparison** cover Karpathy's "entities, concepts, sources" plus the comparative pages the p863 corpus clearly wants (dozens of "X vs Y" pages exist, all mis-typed as `concept`).
- `summary` pages give the index a source-anchored category (Karpathy: ingest "writes a summary page in the wiki").
- `comparison` already has a heuristic in `graph.py` — evidence the taxonomy was intended.

## Phases

### Phase 0 — schema + prompt

- `GenerationPage.kind: str = "concept"` (schema.py:88).
- `generation_prompt.md.j2`: add a `kind` field to the output object spec; list mandatory kinds with the one-line contracts above; configured extras appended from `compile.extraction.categories`.
- `generation.py:324-331`: `node_type=page.kind if page.kind in allowed else "concept"`.

### Phase 1 — output grouping

- `_KIND_FOLDERS` (vault.py:71): add `"summary": "concepts"` (keep one folder; categories are index groups, not directories). `overview` stays at root.
- Index `## {type}` grouping already keys off `node_type` — no change; verify pluralized headings.
- Checkpoints (`_checkpoints.py:102`) already round-trip `node_type` — no change.

### Phase 2 — config

- New key `compile.extraction.categories` (keys.py + `_configs.py`), dict[str, str].
- Validation: reserved names rejected; empty values rejected.
- `knox.toml.example`: document mandatory set + example extra.

### Phase 3 — tests

- Schema coercion: garbage/unknown `kind` → `concept`; valid extra accepted.
- Generation wiring: `node_type` comes from `page.kind` (unit, mocked LLM output).
- Config: reserved-name rejection, empty dict = mandatory only.
- Index: multi-kind wiki renders one `## {Type}` group per kind.

### Phase 4 — docs

- `src/knox/AGENTS.md` (output section: taxonomy is producer-defined per OKF §4.1; mandatory kinds list), `docs/guides/output.md`, `docs/concepts/open-knowledge-format.md` (category examples), `knox.toml.example`.

## Out of scope (flagged)

- **Source category pages**: `references.md` (type `Reference`) already exists; a separate `source` kind would duplicate it.
- **WiCER/quality interplay**: whether extra kinds affect the WiCER probe loop — follow-up.
- **Entity-page cost**: entity pages inflate page counts/LLM spend; the mandatory set is fixed but the *minimum* entity threshold (page vs related-link-only) may need a knob later.
- **`_infer_page_kind`**: single-phase heuristic stays as-is; two-phase now gets its kinds from the LLM.
