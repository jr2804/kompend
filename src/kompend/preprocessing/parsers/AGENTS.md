# DOX — kompend.preprocessing.parsers

<!-- on entry: loaded when agent enters src/kompend/preprocessing/parsers -->
<!-- parent: ../../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

The ingestion backend: parsecraft, and nothing else. parscraft routes each page
across its own sub-backends (native text/markdown/HTML, liteparse, docling,
pandoc, OCR extras, mineru), so a kompend-side backend selector or chain would
have nothing left to select between. This subtree maps parsecraft's IR onto
kompend tier-1 elements and owns the construction path from kompend config to a
ready adapter.

## Architecture

```text
base.py              FileExtension (+ is_text), known_extension(), ParserBackend Protocol,
                     DocumentParser (guard + identity), assert_text_source()/BinarySourceError,
                     scrub_document(), DPI_DEFAULT/IMAGE_SCALE_DEFAULT
options.py           PARSER_PARSECRAFT, ParserOptions (+ from_config) — the construction inputs
document_parser.py   build_parser(options) — the single factory, returns a guarded DocumentParser
extensions.py        supported_extensions() — the ingestible suffix set
parsecraft_routing.py  load_parsecraft_config(), constraints_for(), ParsecraftRoutingConfig —
                     parsecraft's own routing config, read through parsecraft's engine
parsecraft_parser.py ParsecraftParser — the adapter: IR → elements, figures, diagnostics
markdown.py          element_to_markdown()/markdown_to_element() — the wrap/unwrap pair used for
                     rumdl whole-part normalisation
```

`../images.py` (sibling module `preprocessing/images.py`) holds figure storage:
`ImageStore` turns an IR figure asset into a cached file or an inline payload.
The mapping lives here; the store lives one level up because the captioning
stage reads it too.

`PARSER_PARSECRAFT` is a plain constant, not an enum: there is nothing to select
between, and a name vocabulary would invite a second backend.

## Local Contracts

### Routing is parsecraft's configuration

kompend keeps no copy of parsecraft's routing vocabulary. Three files, in
ascending priority, are read by parsecraft's own engine:

1. parsecraft's schema defaults
2. `~/.config/parsecraft/config.toml`
3. `<wiki>/.kompend/parsecraft.toml` (the vault path comes from
   `VaultPaths.parsecraft_config_file`; the `<cwd>/parsecraft.toml` project file
   is appended after it, so the vault outranks a file in the working directory)
4. `PARSECRAFT_CONFIG_*` environment variables

`ParserOptions.parsecraft_config_files` carries the file list;
`parsecraft_routing.load_parsecraft_config()` builds the engine, injects
kompend's one policy value (`KOMPEND_MAX_PASSES = 3`, at the default layer), and
returns a `ParsecraftRoutingConfig` — parsecraft's settings plus the set of
`routing.*` keys a config layer actually wrote. Never read parsecraft's TOML
here by hand, and never re-declare a parsecraft key as a kompend config field:
the six mirrored `[ingest] parsecraft_*` routing keys are in `REMOVED_KEYS` for
exactly that reason.

### Host facts fill blanks, they never overwrite

`constraints_for()` resolves four host-fact keys — `engines`, `allow_ocr`,
`vram_budget_gb`, `gpu_usable` — and only when the config layers left them
unwritten. Two of them are load-bearing, not cosmetic:

- `engines` gates every backend that requires an external tool. An empty set
  (the unwritten default) would silently disable `pandoc` and the tesseract OCR
  backend on a host that has them.
- `allow_ocr` is `None` when unwritten, and `RoutingSettings.routing_constraints`
  omits the kwarg rather than deriving it, so `RoutingConstraints`' own `True`
  applies. The derivation the settings meant is `bool(environment.ocr_backends)`.

The loaded settings themselves stay exactly what the files say. The parse-cache
identity is derived from them, so folding a probe result in would make every
vault re-parse whenever the local machine gained a tool.

### `build_parser` is the only factory, and it defers the native stack

`build_parser(options)` imports the adapter through
`importlib.import_module(_ADAPTER_MODULE)` at the call site. That is the
runtime-only import `.agents/CODE.md` requires (no `TYPE_CHECKING` block, no
`# noqa: PLC0415` — the latter is a BANNED-NOQA violation): parsecraft's native
stack costs ~1.4 s, which `kompend --help` must not pay. Callers that only need
the suffix table (`list-parsers`, extension discovery) pass no options and never
instantiate a backend.

### Identity: what a change re-parses

`DocumentParser.backend_identity()` composes the backend's
`backend_identity()` with its `cache_identity()`, and that string — not
`cache_identity()` alone — is the parse-cache key
(`PARSE_CACHE_PARSER_KEY`). The adapter's `backend_identity()` names what can
parse here (registered backends, installed extras, the resolved
`[routing]` pins); `cache_identity()` names the routing inputs (judge spec,
preference, OCR permission, engines, backend options, image mode). A parsecraft
version bump is deliberately absent: a bug-fix release must not re-parse a
vault, while a changed backend set or a changed routing value must.

### The format table classifies, it never gates

`FileExtension` lists the formats kompend knows about; parsecraft knows more
(its native-text backend reads `.c/.h/.py/…`). Suffix resolution goes through
`known_extension()`, which returns `None` for a suffix outside the table instead
of raising: an unknown suffix is a fact about the format, not an error. Upstream
issue #3 — a raise on an unknown suffix dropped every code file the backend
reads.

`supported_extensions()` merges the adapter's own suffix set with the canonical
`FileExtension` table and is the single answer to "what can this vault ingest":
discovery, the status report and `list-parsers` all read it.

One gate exists, at intake: `assert_text_source()` — called at the top of
`DocumentParser.parse()` — rejects a suffix the table knows as *text* whose bytes
sniff as binary (binaryornot). A PDF named `.md` must fail loudly
(`BinarySourceError`, a `ValueError`), not succeed as U+FFFD soup (knox-rtmn).
The gate is deliberately narrow: unknown suffixes and honest binary containers
pass untouched, and text judgement accepts UTF-16 so the encoding layer's
charset-normalizer fallback keeps working.

### Parse-time markdown normalisation (rumdl)

`kompend.core.text.rumdl.normalize_element_markdown()` rewrites markdown element
contents through **one batched `rumdl` call per document**, so cached content is
canonical and formatting differences are levelled at the part level. Called from
the parse tier, `ingest_documents`, and text attachments. Content-only: page,
reading order and metadata are preserved; tables and images are skipped.
Best-effort — a failure keeps the original content.

### No per-backend Markdown converters

A backend that produces Markdown must not walk it with a local converter. The
per-backend copies this project once carried (each recognising only headings,
images, tables and paragraphs — `$$…$$` equations, fenced code and list items
silently folded into `TEXT`) are the cautionary tale. parsecraft maps its IR to
elements directly. If a future backend emits Markdown, the walk belongs in
`markdown.py` beside the shared patterns — never next to the SDK call.

### `figure_enrichment` is informational, not a hard VLM skip

`ParserBackend.figure_enrichment: bool` advertises whether a backend emits
figure descriptions. It is shown in `kompend list-parsers` and metadata only.
`VLMEnricher` (Tier 2) always runs when `compile.llm.vlm_model` is set and does
**per-figure** bypass instead: a figure carrying `chunk.metadata["description"]`
is attached as an `image_description` entity and skipped; figures the backend
missed are still VLM-captioned.

### Parser output contract

Image elements set `metadata["image_file"]` (file mode) or
`metadata["image_base64"]` (base64 mode) — use `MetadataField.IMAGE_FILE` /
`MetadataField.IMAGE_BASE64`, never bare literals. The figure's own text rides
along as `metadata["description"]`, which reaches `StructuredFigure.description`
and `chunk.metadata`, so the captioning stage can use it instead of calling a
VLM. A figure without a caption is still a figure: the image is its content, a
caption is only ever a description.

Figures come from the IR: parsecraft hands a crop over as an `ExtractedAsset` on
the page, and the chunk that owns it names the asset in `metadata["asset_id"]`
(`MetadataField.ASSET_ID`). `ImageStore` decides where the bytes land, and
`compile.image_mode` picks the mode (`none` extracts no images at all). A figure
with no image to point at stays what it is: its text becomes a text element,
never an image element without a reference.

The mode also decides whether the backend is asked for crops at all: when the
run stores figures, `liteparse.extract_images` is derived as `true` (liteparse is
the backend behind the default route for a text PDF, and crops nothing unless
asked). No kompend key exists for it — `[routing.backend_options]` in
parsecraft's config is the escape hatch, and a written value wins over the
derived one.

The image mode is part of the parse-cache identity (`images:file`), because a run
that extracts images produces a different document than one that does not. The
segment is emitted only when the run extracts images, so a vault that never had
them is not re-parsed for nothing.

Extracted figure files are content-addressed:
`<source-stem>__<sha256(bytes)[:16]><suffix>` (the stem is for readability only).
The same figure re-parsed keeps its file instead of accumulating copies, and
changed bytes land under a new name instead of overwriting the old crop.

## Work Guidance

Adding a capability means adding it to parsecraft, not to kompend. Only when
parsecraft genuinely cannot host it: create `<name>_parser.py` implementing
`ParserBackend` (`can_parse`, `parse`, `supported_extensions`,
`figure_enrichment`), extend `build_parser` to choose it, declare the dependency
as an optional extra, and add tests to `tests/unit/test_parser_registry.py`. Do
not reintroduce a name vocabulary or a registry: one factory, one adapter.

## Verification

- `uv run pytest tests/unit/test_parser_registry.py tests/unit/test_parser_options_surfaces.py tests/unit/test_parsecraft_parser.py tests/unit/test_intake_binary_guard.py -x`
- `uv run python -c "from kompend.preprocessing.parsers.extensions import supported_extensions; print(sorted(supported_extensions()))"` — discovery works without instantiating a backend
- `uv run python scripts/checks/all.py` (`antipatterns`, `enum_cache`, `field_consistency`)
