# DOX — kompend.preprocessing.parsers

<!-- on entry: loaded when agent enters src/kompend/preprocessing/parsers -->
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
markdown.py        element_to_markdown()/markdown_to_element() — the wrap/unwrap pair used for
                   rumdl whole-part normalisation; no Markdown→element converter lives here
                   anymore (parsecraft maps its IR directly)
_registry.py       ParserBuildOptions, PARSER_BACKENDS, KNOWN_BACKENDS, register_backend/get_backend,
                   build_default_router() — the single factory for a ready-to-use router
parsecraft_parser.py  ParsecraftParser — THE ingestion backend; per-page routing is
                   parsecraft's, kompend keeps none of its own
```

`build_pipeline` builds a `ParserRouter` from `config.parser_options`. Exactly
**one** backend is registered, `DEFAULT_PARSER` (parsecraft). There is no config
key and no CLI flag selecting a backend: parsecraft routes each page across its
own sub-backends, so a kompend-side selector would have nothing to choose between.
kompend keeps no chain, no ordering and no hand-off.

`build_default_router` raises for an unknown name, and for a known backend whose
optional dependency is missing (naming the install command). A vault whose parser
is absent must fail loudly instead of starting with an empty router. The `name`
parameter still exists as registry API — the tests pin it — but no caller passes
it.

`ingest.parsecraft_backend_options` is the only parser-side config key. It
carries the backend options the user wants to override (a mapping of option
name → value); `ParserBuildOptions.parsecraft_backend_options` hands them to
the adapter, which merges them over its derived defaults. The options are part
of the parser's `cache_identity()`, so changing them re-parses.

`get_all_extensions(names)` takes the backend names to union suffixes for.
Discovery passes `[DEFAULT_PARSER]`, so a vault ingests exactly what its parser can
read — parsecraft's native-text backend reads `.c/.h/.py/…`.

`ParserRouter.fingerprint()` hashes the registered backend class names plus any
per-backend `cache_identity()` — the identity of the parser *with its routing
inputs*. The parse and chunk caches key on it, so changing a backend's own
routing configuration (`ingest.parsecraft_judge` / `_model` / `_preference` / `_effort` / `_allow_ocr` / `_engines` / `_extract_vector_graphics` / `_preferred_backend` / `_only_backend`, or
the `max_passes` the adapter passes) re-parses instead of silently reusing the
previous setup. A backend that carries routing inputs defines `cache_identity()`;
the router looks it up structurally (the parsecraft adapter deliberately is not a
`BaseParser`), and an absent hook contributes an empty identity, so adding one
never invalidates a cache by itself.

## Rules

### One backend, ever

A second kompend-side backend competes with parsecraft's own per-page routing
rather than complementing it. Add capabilities to parsecraft, not to kompend. The
kompend-side backends that duplicated it (`docling`, `liteparse`, `pandoc`,
`glm-ocr`, `web`, `marker`, `mineru`) are all deleted. Two engines stay
reachable without kompend backends: `marker` is bring-your-own — parsecraft's
marker backend detects a manual `marker-pdf` install by import (ADR-0006; no
marker-pdf release can enter kompend's graph: 1.x pins anthropic<0.47 against
pydantic-ai's ≥0.108, and 2.0.0 resolves only by downgrading anthropic and
pillow graph-wide) — and `mineru` enters through kompend's `parsecraft[mineru]`
extra (licence opt-in, ADR-0007 — see the comment in `pyproject.toml`).
Extraction-pass defaults now live in parsecraft.

### Registration — cycle avoidance

`_registry.py` imports **no parser class** (that would cycle). Each parser module
imports `register_backend, ParserBuildOptions` from `_registry` and calls
`register_backend(...)` at module level. `_ensure_backend` imports the module on
demand, so a backend whose optional dependency is missing simply stays
unregistered — and selecting it then raises with its install hint.

### No per-backend Markdown converters

A backend that produces Markdown MUST NOT walk it with a local converter.
The per-backend copies this project once carried (each recognising only
headings, images, tables and paragraphs — `$$…$$` equations, fenced code and
list items silently folded into `TEXT`) are the cautionary tale; the shared
converter that replaced them (`markdown_to_elements` + `MarkdownContext`)
was deleted together with the last kompend-side markdown backend. parsecraft
maps its IR to elements directly. If a future backend emits Markdown, the
walk belongs in `markdown.py` beside the shared patterns — never next to the
SDK call.

### Parse-time markdown normalisation (rumdl)

After parsing, `kompend.core.text.rumdl.normalize_element_markdown()` rewrites
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
descriptions. It is shown in `kompend list-parsers` and metadata only. `VLMEnricher`
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
- `uv run python -c "from kompend.preprocessing.parsers._registry import PARSER_BACKENDS; print(sorted(PARSER_BACKENDS))"` — registration fires without error
