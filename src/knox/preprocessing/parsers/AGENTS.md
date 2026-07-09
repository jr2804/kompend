# DOX — knox.preprocessing.parsers

## Purpose

Pluggable ingestion backends. Each backend is a `BaseParser` subclass plus a
factory that builds an instance from `ParserBuildOptions`. Backends self-register
at import time via `register_backend` (mirror of `knox.llm.providers`);
importing the `knox.preprocessing.parsers` package triggers registration.

## Architecture

```text
base.py            BaseParser, FileExtension, ParserRouter, DPI/IMAGE_SCALE constants
_registry.py       ParserBuildOptions, PARSER_BACKENDS, register_backend/get_backend,
                   build_default_router() — the single factory for a ready-to-use router
docling_parser.py  DoclingParser  (default backend; structure-preserving via Docling)
pandoc_parser.py   PandocParser   (legacy formats + LaTeX + binaryornot text fallback)
liteparse_parser.py LiteParseParser (fast Rust-core parser; no figure enrichment)
```

`build_pipeline` builds a `ParserRouter` from `config.parser`
(`ingest.parser` config key; `--parser` CLI flag on `ingest`/`parse`).
`config.parser` is a comma-separated priority list (e.g. `"liteparse,docling"`):
backends are registered in that order and the first one whose `can_parse()`
matches a file wins. `pandoc` is always appended last (legacy formats + LaTeX +
universal text fallback), so formats no listed backend handles never regress.

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

### Registration — cycle avoidance

`_registry.py` imports **no parser class** (that would cycle). Each parser
module imports `register_backend, ParserBuildOptions` from `_registry` and calls
`register_backend(...)` at module level. Importing `parsers/__init__.py` (which
imports every parser module) fires all registrations before any
`build_default_router` call. Re-export `build_default_router` only after the
parser imports so backends are registered first.

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

Image elements set `metadata["image_path"]` (file mode) or
`metadata["image_base64"]` (base64 mode) — use `MetadataField.IMAGE_PATH` /
`IMAGE_BASE64` symbols, never bare literals. An enriching backend writes its
description to `metadata["description"]`, which flows via `extractors.py` →
`StructuredFigure.description` and via `hierarchical.py` → `chunk.metadata`.

## Verification

- `uv run pytest tests/unit/test_parser_registry.py -x`
- `uv run python -c "from knox.preprocessing.parsers._registry import PARSER_BACKENDS; print(sorted(PARSER_BACKENDS))"` — registration fires without error
