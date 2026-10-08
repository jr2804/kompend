# DOX — knox.preprocessing.parsers

<!-- on entry: loaded when agent enters src/knox/preprocessing/parsers -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Pluggable ingestion backends. Each backend is a `BaseParser` subclass plus a
factory that builds an instance from `ParserBuildOptions`. Backends self-register
at import time via `register_backend` (same shape as the zvec/embedding provider factories);
importing the `knox.preprocessing.parsers` package triggers registration.

## Architecture

```text
base.py            BaseParser, FileExtension (extensions only), ParserRouter, DPI/IMAGE_SCALE constants
markdown.py        markdown_to_elements() — THE Markdown→element converter used by every
                   markdown-producing backend; element_to_markdown()/markdown_to_element()
                   are the inverse pair used for rumdl normalisation
_registry.py       ParserBuildOptions, PARSER_BACKENDS, register_backend/get_backend,
                   build_default_router() — the single factory for a ready-to-use router
docling_parser.py  DoclingParser  (default backend; supported_extensions owned by parser)
pandoc_parser.py   PandocParser   (legacy formats + LaTeX + binaryornot text fallback;
                                   _READER_FOR owned by parser, not the enum)
liteparse_parser.py LiteParseParser (fast Rust-core parser; no figure enrichment)
```

`build_pipeline` builds a `ParserRouter` from `config.parser`
(`ingest.parser` config key; `--parser` CLI flag on `ingest`/`parse`).
`config.parser` is a comma-separated priority list (e.g. `"liteparse,docling"`):
backends are registered in that order and the first one whose
:meth:`BaseParser.can_parse` matches a file wins. `pandoc` is always
appended last (legacy formats + LaTeX + universal text fallback), so formats
no listed backend handles never regress.

With `ingest.parser_fallback` (default **true**) a backend that raises during
:meth:`parse` hands off to the *next* configured backend that also claims the
file, logging a WARNING naming the file, the failed backend, the error, and the
backend that succeeded. This is chain fallback *between configured backends* —
distinct from the *text* fallback above, which applies only to files no backend
claims. `parser_fallback = false` restores one-shot routing (first match only,
the failure propagates). The parse cache keys on it, so flipping the flag
invalidates instead of reusing output produced under the other policy.

`ParserRouter.fingerprint()` returns a hash of the ordered backend class
names plus the text fallback, the fallback policy, and any per-backend
`cache_identity()` — the identity of the parser chain *with its routing
inputs*. The parse cache and chunk cache key on it, so switching
`ingest.parser`, `ingest.parser_fallback`, or a backend's own routing
configuration (`ingest.parsecraft_judge` / `_model` / `_preference`) re-parses
and re-chunks instead of silently reusing output from the previous setup.

A backend that carries routing inputs contributes them by defining
`cache_identity()`. The router looks the method up structurally (the
parsecraft adapter deliberately is not a `BaseParser`), and a backend without
the hook contributes nothing — an empty identity keeps existing cache keys
byte-identical, so adding the hook never invalidates a cache by itself.

The canonical file-extension enum (`FileExtension`, in `base.py`) is the
single source of truth for *which extensions exist*. Each parser declares its
own `supported_extensions` set and owns any parser-specific metadata — e.g.
`PandocParser._READER_FOR` maps extensions to pandoc reader names without
coupling the enum to pandoc terminology.

## Rules

### Pandoc backend — lazy binary, on-demand download

`PandocParser` depends on `pypandoc` (NOT the `pandoc` PyPI package). The
legacy `pandoc` package launched the pandoc binary at *import time*
(`pandoc.configure(auto=True)` in `pandoc.types`), which crashed CLI startup
on misconfigured systems (e.g. a mise shim that isn't a valid Win32 exe →
`WinError 193`) and added startup latency even when pandoc wasn't used.

`pypandoc` defers binary resolution to the first conversion call and never
touches the binary at import. `parse()` calls `_ensure_pandoc_available()`
once per process: a fast no-op when pandoc is on PATH, or an on-demand
download (`pypandoc.ensure_pandoc_installed()`) when it's missing. The JSON
AST (`pypandoc.convert_file(path, 'json')`) replaces the legacy typed-AST —
blocks are classified by their `"t"` tag string instead of `isinstance`.

Input reader resolution never returns `""`: `FileExtension.pandoc_format`
maps each known extension to an explicit pandoc reader
(`_PANDOC_READERS` in `base.py`), and `_resolve_format` falls back to
`"markdown"` for unknown text files (source code, arbitrary suffixes) so any
non-binary file is parsed like `.txt`. Returning `""` made pypandoc infer
the reader from the file extension and crash on extensions it cannot read
(`.c`, `.log`, `.yaml`, `.ltx`, …).

### Registration — cycle avoidance

`_registry.py` imports **no parser class** (that would cycle). Each parser
module imports `register_backend, ParserBuildOptions` from `_registry` and calls
`register_backend(...)` at module level. Importing `parsers/__init__.py` (which
imports every parser module) fires all registrations before any
`build_default_router` call. Re-export `build_default_router` only after the
parser imports so backends are registered first.

### One Markdown converter — no per-backend copies

Any backend that produces Markdown (liteparse, marker, mineru, glmocr, web)
MUST route it through `markdown.py:markdown_to_elements()`. Do not add a local
`_markdown_to_elements` / `_parse_table` / `_build_image_element` — those were
duplicated across four modules and drifted: every copy recognised only
headings, images, tables and paragraphs, so `$$…$$` equations, fenced code and
list items were silently folded into `TEXT`.

Backend-specific behaviour belongs in a single `resolve_image` callback passed
via `MarkdownContext` (ref → bytes). That is the *only* part of the walk that
legitimately differs between backends.

### Parse-time markdown normalisation (rumdl)

After parsing, `knox.core.text.rumdl.normalize_element_markdown()` rewrites
markdown element contents through **one batched `rumdl` call per document**,
so cached content is canonical and backend formatting differences are levelled
at the part level. Called from the parse tier, `ingest_documents`, and text
attachments. Content-only: page/reading-order/metadata are preserved; tables
and images are skipped. Best-effort — failure keeps the original content.

### Full capability by default

A backend's `parse()` must enable that backend's full extraction capability by
default (figures, tables, formulas, code). Leaving a capability off because it
costs time is a silent data-loss bug — the user cannot tell an empty result
from a document that has no such content. Docling's `PdfPipelineOptions` is
the cautionary example: passing only picture options let docling's own
`do_formula_enrichment=False` / `do_code_enrichment=False` defaults win, so
every equation was dropped. Expose a `[compile.pdf]` knob to turn a pass off
and default it on.

The one exception is a pass that only *re-derives* content the format already
carries. Docling's `do_code_enrichment` runs CodeFormulaV2 over every
`CodeItem` to re-OCR it; a born-digital PDF's code is already correct in its
text layer, so the pass is pure cost (~140 s per code block on CPU — measured
9 blocks adding ~1250 s to a 16-page range, and >25 min for one code-heavy
manual). `code_enrichment` therefore defaults `False`. Tables
(`table_structure`) and equations (`formula_enrichment`) have no equivalent
native representation and stay on. OCR follows the same rule through
`ocr = "auto"`: it runs only when the PDF has no text layer.

### Adding a backend

1. Create a module `preprocessing/parsers/<name>_parser.py` with a
   `BaseParser` subclass. Set `supported_extensions` (a set of `FileExtension`)
   and `figure_enrichment` (capability flag, see below).
2. Append a module-level `_factory(options: ParserBuildOptions) -> <Parser>`
   that maps `ParserBuildOptions` → constructor kwargs, then
   `register_backend("<name>", _factory, parser_cls=<Parser>)`.
3. Import the class in `parsers/__init__.py` and add it to `__all__` (this
   import is what triggers registration).
4. Install the dependency as a **regular** dep (`uv add <pkg>`), never optional.
   No `try/except ImportError` guards — core deps are unconditional (see
   `src/knox/AGENTS.md`).
5. Add tests to `tests/unit/test_parser_registry.py`.

### `figure_enrichment` is informational, NOT a hard VLM skip

`BaseParser.figure_enrichment: bool` advertises whether a backend emits figure
descriptions. It is shown in `knox list-parsers` and metadata only.
`VLMEnricher` (Tier 2) **always** runs when `compile.llm.vlm_model` is set and
does **per-figure** bypass instead: a figure carrying
`chunk.metadata["description"]` (set by an enriching backend) is attached as an
`image_description` entity and skipped; figures the backend missed are still
VLM-captioned. Backends that do not enrich (Docling, LiteParse, Pandoc) leave
`figure_enrichment=False`; VLM captions all their figures.

### Parser output contract

Image elements set `metadata["image_file"]` (file mode) or
`metadata["image_base64"]` (base64 mode) — use `MetadataField.IMAGE_PATH` /
`IMAGE_BASE64` symbols, never bare literals. An enriching backend writes its
description to `metadata["description"]`, which flows via `extractors.py` →
`StructuredFigure.description` and via `hierarchical.py` → `chunk.metadata`.

### Full-capability default

Parsers MUST request *every* extraction pass the backend exposes by default:
turning them on is the contract, turning them off needs a config key.
Disabling silently drops content from the wiki and the user has no signal.

- **LiteParseParser**: `extract_images=True` is set explicitly; `image_mode`
  controls embedding. Figure refs are normalized via `_ref_to_id`, which
  strips both the new (`img_`) and legacy (`image_`) liteparse prefixes; a
  missing prefix slipped past and dropped every figure (bug #1410).
- **DoclingParser**: `PdfPipelineOptions` builds from `[compile.pdf]`
  (`PdfConfig`): `do_table_structure` and `do_formula_enrichment` (default
  `True`), `do_code_enrichment` (default `False` — see "Full capability by
  default"), `do_ocr` resolved by a text-layer probe (`PdfOcrMode.AUTO`),
  `table_mode` (TableFormer `accurate`/`fast`), and
  `accelerator_options.num_threads` (`0` = all usable cores; Docling otherwise
  hard-defaults to 4 regardless of host). The `image_mode="none"` branch still
  passes enrichment so equations survive even when the user opts out of picture
  extraction.
- Formula/code enrichment hard-requires `accelerate`; it is a regular `docling`
  extra in `pyproject.toml`.

## Verification

- `uv run pytest tests/unit/test_parser_registry.py tests/unit/test_parser_capabilities.py tests/unit/test_pdf_chunking.py -x`
- `uv run python -c "from knox.preprocessing.parsers._registry import PARSER_BACKENDS; print(sorted(PARSER_BACKENDS))"` — registration fires without error
