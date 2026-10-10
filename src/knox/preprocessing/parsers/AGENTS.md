# DOX — knox.preprocessing.parsers

<!-- on entry: loaded when agent enters src/knox/preprocessing/parsers -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Pluggable ingestion backends. Each backend is a `BaseParser` subclass plus a
factory that builds an instance from `ParserBuildOptions`. Backends
self-register at import time via `register_backend`; a backend's module is
imported on demand (`_ensure_backend`), never at package-import time, so CLI
`--help` stays free of the heavy optional dependencies.

## Architecture

```text
base.py            BaseParser, FileExtension (extensions only), ParserRouter, DPI/IMAGE_SCALE constants
markdown.py        markdown_to_elements() — THE Markdown→element converter used by every
                   markdown-producing backend; element_to_markdown()/markdown_to_element()
                   are the inverse pair used for rumdl normalisation
_registry.py       ParserBuildOptions, PARSER_BACKENDS, KNOWN_BACKENDS, register_backend/get_backend,
                   build_default_router() — the single factory for a ready-to-use router
parsecraft_parser.py  ParsecraftParser — THE ingestion backend; per-page routing is
                   parsecraft's, knox keeps none of its own
marker_parser.py   MarkerParser  (manual install: high-accuracy PDF)
mineru_parser.py   MinerUParser  (manual install: layout-aware PDF/Office + OCR)
```

`build_pipeline` builds a `ParserRouter` from `config.parser` (`ingest.parser`
config key; `--parser` CLI flag on `ingest`/`parse`). Exactly **one** backend is
registered — `DEFAULT_PARSER`, parsecraft, unless the operator names another.
parsecraft routes each page across its own sub-backends, so knox keeps no chain,
no ordering and no hand-off between backends.

`build_default_router` raises for an unknown name, and for a known backend whose
optional dependency is missing (naming the install command). A vault whose
configured parser is absent must fail loudly instead of starting with an empty
router.

`get_all_extensions(names)` takes the backend names to union suffixes for.
Discovery passes `[config.parser]`, so a vault ingests exactly what its backend
can read — parsecraft's native-text backend reads `.c/.h/.py/…`, a PDF-only
backend does not.

`ParserRouter.fingerprint()` hashes the registered backend class names plus any
per-backend `cache_identity()` — the identity of the parser *with its routing
inputs*. The parse and chunk caches key on it, so switching `ingest.parser` or a
backend's own routing configuration (`ingest.parsecraft_judge` / `_model` /
`_preference`) re-parses instead of silently reusing the previous setup. A
backend that carries routing inputs defines `cache_identity()`; the router looks
it up structurally (the parsecraft adapter deliberately is not a `BaseParser`),
and an absent hook contributes an empty identity, so adding one never invalidates
a cache by itself.

## Rules

### One backend, ever

A second knox-side backend competes with parsecraft's own per-page routing
rather than complementing it. Add capabilities to parsecraft, not to knox. The
knox-side backends that duplicated it (`docling`, `liteparse`, `pandoc`,
`glm-ocr`, `web`) were deleted; `marker` and `mineru` are placeholders until
knox declares their parsecraft extras (beads pc-u4g / pc-gv8), and each is
removed at that point. Extraction-pass defaults now live in parsecraft.

### Registration — cycle avoidance

`_registry.py` imports **no parser class** (that would cycle). Each parser module
imports `register_backend, ParserBuildOptions` from `_registry` and calls
`register_backend(...)` at module level. `_ensure_backend` imports the module on
demand, so a backend whose optional dependency is missing simply stays
unregistered — and selecting it then raises with its install hint.

### One Markdown converter — no per-backend copies

Any backend that produces Markdown MUST route it through
`markdown.py:markdown_to_elements()`. Do not add a local
`_markdown_to_elements` / `_parse_table` / `_build_image_element` — those were
duplicated across four modules and drifted: every copy recognised only headings,
images, tables and paragraphs, so `$$…$$` equations, fenced code and list items
were silently folded into `TEXT`.

Backend-specific behaviour belongs in a single `resolve_image` callback passed
via `MarkdownContext` (ref → bytes). That is the *only* part of the walk that
legitimately differs between backends.

### Parse-time markdown normalisation (rumdl)

After parsing, `knox.core.text.rumdl.normalize_element_markdown()` rewrites
markdown element contents through **one batched `rumdl` call per document**, so
cached content is canonical and backend formatting differences are levelled at
the part level. Called from the parse tier, `ingest_documents`, and text
attachments. Content-only: page/reading-order/metadata are preserved; tables and
images are skipped. Best-effort — failure keeps the original content.

### Adding a backend

Only when parsecraft cannot host the capability.

1. Create `preprocessing/parsers/<name>_parser.py` with a `BaseParser`
   subclass. Set `supported_extensions` (a set of `FileExtension`) and
   `figure_enrichment` (capability flag, see below).
2. Add a module-level `_factory(options: ParserBuildOptions) -> <Parser>`, call
   `register_backend("<name>", _factory, parser_cls=<Parser>)`, and add the name
   to `KNOWN_BACKENDS` / `_BACKEND_MODULES` in `_registry.py`.
3. Declare the dependency as an **optional** extra or a manual install. No
   parser backend may be a main dependency — parsecraft is the ingestion path,
   and the backends it drives arrive transitively through it.
4. Add tests to `tests/unit/test_parser_registry.py`.

### `figure_enrichment` is informational, NOT a hard VLM skip

`BaseParser.figure_enrichment: bool` advertises whether a backend emits figure
descriptions. It is shown in `knox list-parsers` and metadata only. `VLMEnricher`
(Tier 2) **always** runs when `compile.llm.vlm_model` is set and does
**per-figure** bypass instead: a figure carrying `chunk.metadata["description"]`
(set by an enriching backend) is attached as an `image_description` entity and
skipped; figures the backend missed are still VLM-captioned.

### Parser output contract

Image elements set `metadata["image_file"]` (file mode) or
`metadata["image_base64"]` (base64 mode) — use `MetadataField.IMAGE_PATH` /
`IMAGE_BASE64` symbols, never bare literals. An enriching backend writes its
description to `metadata["description"]`, which flows via `extractors.py` →
`StructuredFigure.description` and via `hierarchical.py` → `chunk.metadata`.

## Verification

- `uv run pytest tests/unit/test_parser_registry.py tests/unit/test_parser_options_surfaces.py tests/unit/test_parsecraft_parser.py -x`
- `uv run python -c "from knox.preprocessing.parsers._registry import PARSER_BACKENDS; print(sorted(PARSER_BACKENDS))"` — registration fires without error
