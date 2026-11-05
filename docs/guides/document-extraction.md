# Document Extraction in kompend

kompend aims to parse a wide variety of document formats
into a structured, LLM-aware representation. This document explains which
frameworks are used, why, and what was considered before settling on the
current architecture.

See also:

- [Architecture overview](../architecture/index.md) — 5-tier pipeline
- [Pipeline Tiers](../concepts/pipeline-tiers.md) — Per-tier component reference
- [LLM Configuration](llm-config.md) — Provider setup and model selection

## Parser Backends

parsecraft is the only parser path; the selector is gone. The only parser
decision that stays kompend's is which provider credential may be spent on the
routing judge (`[ingest] parsecraft_judge`, `parsecraft_judge_model`) plus the
host-resource bounds on a heavy pass — see [Routing settings](#routing-settings).
Everything else about routing is parsecraft's own configuration, read from
parsecraft's files, never mirrored into `[ingest]`.

- `parsecraft` — an external document-intelligence pipeline that routes each
  page across its own sub-backends (native text/markdown/HTML, LiteParse,
  docling, pandoc, OCR extras) and returns one document per source. It is a
  **core dependency** of kompend, together with its `docling` and `web`
  sub-backends, so installing kompend installs it.

`kompend list-parsers` shows the backend and the formats it claims.

Two opt-in extensions ride on top of parsecraft rather than being kompend
backends:

- **`kompend[mineru]`** — adds MinerU 4.x as a parsecraft sub-backend
  (`parsecraft[mineru]>=2026.10.14`), eligible for layout-heavy and OCR
  pages. The extra pulls model weights, not the `mineru` package: parsecraft
  runs MinerU natively. It still opts into MinerU's own licence terms — read
  the note in `pyproject.toml` before using it commercially.
- **marker (bring-your-own)** — `marker-pdf` cannot be locked or pinned by
  kompend at all (see [below](#why-marker-cannot-be-a-locked-extra)); install
  it yourself and parsecraft counts it among its routing candidates.

The judge tier needs no extra: `typesafe-sdk` — shared by the `ollama`, `zen`
and `typesafe-ai` providers — is base-install in parsecraft ≥ 2026.10.5, so it
arrives with kompend itself. The judge is enabled purely by config
(`[ingest] parsecraft_judge`, `parsecraft_judge_model`; the ranking axis it
works against is `[routing] preference` in parsecraft's config —
see [Judge tier](#judge-tier-model-assisted-per-page-routing)).

### Why `marker` cannot be a locked extra

Measured fresh (kompend-3, 2026-10-23) against the current lock: the tight pin
`marker-pdf>=1.10.2,<2` **hard-fails** — `marker-pdf` 1.10.2 pins
`anthropic>=0.46,<0.47` while `pydantic-ai` 2.60.0 needs `anthropic>=1.12.1`,
and the two ranges do not overlap. Dropping the upper bound only makes it
"resolve" by downgrading `anthropic` 2.1.0 → 0.125.0 and `pillow`
12.3.0 → 10.4.0 — silently regressing kompend's LLM stack.

That is the substantive reason the marker backend was deleted rather than
kept behind an extra (ADR-0006): kompend cannot own the `marker-pdf` pin, so
marker is parsecraft's **bring-your-own** sub-backend.

## Supported Formats

Discovery takes the registry default (parsecraft) and asks it which suffixes
it claims; the source directory is scanned for exactly those. parsecraft's
current 34-suffix table is what a default `kompend ingest` covers, plus eight
legacy suffixes from kompend's own table (`.adoc .bmp .doc .eml .gif .ltx
.msg .webp`) that parsecraft rejects at parse time — such a file is skipped
with a message rather than crashing. `.kompend/`
is excluded regardless of claimed formats, so state artifacts such as
`.kompend/rumdl.toml` are never ingested even though parsecraft claims `.toml`.

| Format | Handled by | Notes |
|--------|------------|-------|
| `.pdf` | `parsecraft` | Routed per page across its sub-backends; scanned pages escalate to an OCR backend when one is installed |
| `.docx`, `.pptx`, `.xlsx` | `parsecraft` | Office formats via its sub-backends |
| `.epub`, `.odt`, `.rtf` | `parsecraft` | E-book and legacy text containers |
| `.md`, `.markdown`, `.html`, `.htm` | `parsecraft` | Native text and markup backends |
| `.csv`, `.tsv` | `parsecraft` | Deterministic table IR: one TABLE element per file, header row first, ragged rows kept |
| `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` | `parsecraft` | Images; content comes from an OCR/VLM sub-backend when installed |
| `.txt`, `.text`, `.log`, `.rst`, `.ini`, `.cfg`, `.toml`, `.c`, `.h`, `.hpp`, `.cpp`, `.cc`, `.py`, `.m`, `.sh`, `.tex` | `parsecraft` | Plain-text source family |

`.html`/`.htm` are handled by parsecraft's own native/trafilatura backends —
there is no separate web parser. `kompend list-parsers` prints the live
extension list. A file whose suffix the backend does not claim is never
picked up by discovery.

## Parser Strategy: Speed vs. Accuracy

parsecraft is the only parser path, and it does not pick one engine for a
whole corpus: it analyses each page and routes it, so a born-digital text
page stays on the cheap native/LiteParse path while a scanned or
table-heavy page in the same document can escalate to a stronger backend.
The speed/accuracy trade-off is decided per page rather than per
configuration.

### One backend, no chain

There is no comma-separated list and no `parser_fallback`. parsecraft performs
its own per-page routing across its registered sub-backends (native, docling,
liteparse, pandoc, OCR), and produces one `ParsedDocument` per source however
many internal passes fired. Degraded pages (parsecraft downgraded an OCR route
to native text) surface as kompend metadata (`degraded_pages`), so the choice stays
visible after the parse.

A second kompend-side level of ordering and hand-off would compete with that
routing rather than complement it, so kompend keeps none. The routing inputs
(judge, preference, engine list, backend pins) are parsecraft's own
configuration — see [Routing settings](#routing-settings). An unknown backend
name raises; a known backend whose optional dependency is missing raises with
its install command — a vault whose configured parser is absent fails loudly
instead of parsing nothing.

## Routing settings

kompend builds one parser — parsecraft's — and hands it the two decisions that
are genuinely kompend's: which provider credential may be spent on the routing
judge (`ingest.parsecraft_judge`, `ingest.parsecraft_judge_model`) and how much
of the host a docling pass may use (`ingest.parsecraft_docling_threads`,
`ingest.parse_min_free_commit_mb`, `ingest.max_concurrent_heavy_parses`).
Everything else about routing is parsecraft's config, not a kompend key.

The precedence, lowest first:

1. parsecraft's schema defaults
2. `~/.config/parsecraft/config.toml` (global)
3. `<wiki>/.kompend/parsecraft.toml` (this vault)
4. `./parsecraft.toml` (working directory)
5. `PARSECRAFT_CONFIG_*` environment variables

Within the project layer a later file wins per key (deep merge), and
`parsecraft config show` names the exact file that supplied each value. An
unknown key is a config error — parsecraft's `[routing]` section forbids
extras — so a typo fails at load instead of silently doing nothing.

```toml
# <wiki>/.kompend/parsecraft.toml
[routing]
preference = "quality"            # speed | balanced | quality
preferred_backend = "docling"     # a preference; ineligible pages route elsewhere
max_passes = 2                    # fallback-pass budget
# allow_ocr = true                # omit the key to let eligibility derive it

[routing.backend_options.docling] # forwarded verbatim; keys are backend-owned
do_ocr = false
```

kompend keeps no copy of these keys. A vault that still sets one of the six
retired `[ingest] parsecraft_*` routing keys (preference, allow_ocr, engines,
preferred_backend, only_backend, backend_options) loads, is warned by
`kompend config check`, and ignores it — they live in `REMOVED_KEYS`, and the
table above is the only place they are read from now.

## Sub-Backends: what parsecraft routes between

There is one selectable backend; the comparison that matters is between
parsecraft's sub-backends, and the extras decide which are present:

| Sub-backend | What it wins | Arrives via |
|------------|--------------|-------------|
| native (text/markdown/HTML) | The fast path for born-digital content | core (`web` for HTML) |
| LiteParse | Default PDF engine | core |
| docling | Structure-heavy pages: equations, tables, complex layouts | core |
| pandoc | Legacy containers (EPub, ODT, RTF) | core |
| PyMuPDF | Light PDF text extraction (AGPL — see below) | `kompend[pymupdf]` |
| OCR models | Scanned pages (per-page escalation) | `kompend[ocr]` |
| mineru | Layout-aware PDF/Office with OCR | `kompend[mineru]` |
| PyMuPDF + MinerU | Both PDF engines in one extra | `kompend[pdf-full]` |
| marker | Highest reported benchmark accuracy | bring-your-own install |

### Choosing

- **Default:** stock kompend covers text, office, PDF and container formats
  with per-page routing.
- **Layout-heavy or scanned corpora:** add `kompend[mineru]`, or
  `kompend[ocr]` for the OCR model stack.
- **An AGPL PDF engine:** `kompend[pymupdf]` adds PyMuPDF (AGPL-3.0 or
  commercial); `kompend[pdf-full]` adds it alongside MinerU.
- **Maximum accuracy on hard layouts:** install `marker-pdf` yourself
  (GPU preferred) — kompend cannot pin it (see
  [Why `marker` cannot be a locked extra](#why-marker-cannot-be-a-locked-extra)).

## Part-level markdown normalisation

Every backend that produces Markdown after its SDK call goes through one shared
converter (`kompend/preprocessing/parsers/markdown.py`) that maps the Markdown into
tier-1 elements. It recognises headings, images, tables, **equations** (`$$…$$`,
`\[…\]`), **fenced code** (with language), and **list items**.

Before that consolidation each backend had its own copy of the walk, and all of
them recognised only headings, images, tables and paragraphs — so equations,
code and list items from sub-backends that *do* emit them (docling, marker,
mineru) were folded into `TEXT` and reached the chunker as prose.

After conversion, each document's markdown-bearing elements are normalised
through `rumdl` in **one batched call per document**, at parse time — so the
cached parsed content is already canonical and backend formatting differences
are levelled before anything downstream reads it. Only the element *content*
changes; page numbers, reading order and metadata are preserved. Tables and
images are skipped (their representation is already canonical).

This is best-effort: if `rumdl` is not installed or the batch cannot be split
back losslessly, the original content is kept.

## ParsecraftParser: External Adapter

[Parsecraft](https://pypi.org/project/parsecraft/) is an external
document-intelligence pipeline (public API only, separate project from kompend).
kompend consumes it through a single adapter (`preprocessing/parsers/parsecraft_parser.py`)
and maps parsecraft's intermediate representation onto kompend's element model.
It is a core dependency, so installing kompend is all that is required:

```bash
uv add kompend    # pulls parsecraft[docling,web]>=2026.10.16
```

The floor is load-bearing: 2026.10.5 moved `auto`, `liteparse`, `pandoc` and
`openpyxl` into parsecraft's base install, 2026.10.6 split the old `pdf` extra
into `pymupdf` and `pdf-full`, and 2026.10.8 made `extract_vector_graphics`
usable — its `VectorGraphics` read was dict-style on a dataclass until then, so
the option crashed the backend and the table path was unreachable. 2026.10.14
withdrew docling's text routes (markdown and plain text are the native family's,
which is also what takes the docling `Table.match` spin off the markdown path),
made docling report author heading levels, and added the CSV/TSV tier. 2026.10.15
swapped the docling engine for `docling-rs`, which takes the Python docling tree
(`docling`, `docling-ibm-models`, `docling-parse`, `docling-slim`) out of every
install — torch and transformers stay, because `crawl4ai[all]` is a core
dependency and pulls them for its own extras. 2026.10.16 restores the `docling`
capability probe (`docling_rs`), which 2026.10.15 had broken, and drops the
`mineru` package from the `mineru` extra, so the `openai` conflict that forced an
override is gone. Raising it
is a reviewed event (see `docs/architecture/decisions.md`, "Decision:
parsecraft as the ingestion backend"). kompend
never imports parsecraft internals, private modules, or test helpers — only the
documented public surface (`parsecraft.pipeline` / `parsecraft.backends` / IR).

!!! warning "A resolution failure moments after a release is usually a stale index cache"
    `No solution found … because there is no version of kompend[pdf-full]==<version>`
    seconds after a release is almost always a stale index cache, not a missing
    artifact. Check the version on PyPI, then retry with the cache bypassed and
    an explicit version — `UV_NO_CACHE=1 uv tool install --force --refresh "kompend[pdf-full]==2026.10.15"`.

The message is word-for-word what a genuinely empty release produces, so
confirm it against PyPI's JSON before concluding the release itself broke. Note
PyPI answers inconsistently while a release propagates: the `…/<package>/json`
endpoint can still report the previous version, while
`…/<package>/<version>/json` already lists it. For the same cause on the
lockfile path, `uv lock --refresh` re-reads the index — a floor bump can fail
resolution with "only <package><=<previous> is available" even when the new
release is metadata-identical apart from its version.

### What the parsecraft adapter does

The adapter runs unconditionally; there is no backend selector to set. The
remaining `[ingest]` knobs are the parsecraft parameters.

The adapter calls `pipeline.execute()` — analyze → `plan_route` → per-range
dispatch with fallback passes → aggregate `DocumentResult` — so routing,
selective OCR escalation, and per-range fallback passes are inherited rather
than reimplemented in kompend. Figure enrichment is on (`figure_enrichment = True`)
and parsecraft's own figure extraction flows through `VLMEnricher` like every
other backend's.

### Formats parsecraft claims

parsecraft speaks MIME (`text/plain`, `application/pdf`, `image/jpeg`, …);
kompend discovers files by extension, and asks the configured backend which
suffixes it handles (`supported_extensions()`). With parsecraft
that is its 32-suffix table — `.txt`, `.text`, `.log`, `.rst`, `.ini`, `.cfg`,
`.toml`, `.c`, `.h`, `.hpp`, `.cpp`, `.cc`, `.py`, `.m`, `.sh`, `.tex`, plus the
usual PDF/HTML/image set — so a parsecraft vault can ingest source files. A
PDF-only backend would contribute no text suffixes at all. `.kompend/` is excluded
from source discovery regardless of claimed formats: state artifacts like
`.kompend/rumdl.toml` are never ingested even though parsecraft claims `.toml`.

### Fallback interaction

There is no kompend-level chain. One backend is configured, and parsecraft runs
its own per-page routing and fallback passes between its registered
sub-backends (native text/markdown/html/pdf, OCR backends, LiteParse); the
result is one `ParsedDocument` per source regardless of how many internal
passes fired.

If parsecraft exhausts every pass on a page, the adapter raises
`ParsecraftParseError` carrying the per-page cause, and the parse fails — there
is no second backend behind it to fall through to.

### Diagnostics carried into kompend metadata

Five classes of parsecraft signal survive the parse into `.kompend/kompend.db`:

- `metadata.parser` — `"parsecraft"`.
- `metadata.backends` — comma-separated list of parsecraft sub-backends that
  won each page group (parse provenance).
- `metadata.quality_signals` — names of `QualitySignal` entries parsecraft
  attached to the document (route degradation, missing extras, etc.).
- `metadata.degraded_pages` — page numbers whose route was downgraded inside
  parsecraft (e.g. OCR → native text because no eligible OCR backend was
  installed). This is the trace that proves a parse silently downgraded,
  rather than hiding the choice behind a structurally identical result.
- `metadata.empty_pages` — page numbers that yielded no text at all. An
  image-only page served by a text-layer backend comes back with zero content
  instead of raising, so the document "parses" and is simply shorter than the
  source. kompend records the page numbers and logs one warning per file naming
  the remedy ([`[routing] allow_ocr = true`], an OCR-capable backend, or a
  forced backend that reads images). See [Silent empty pages](#silent-empty-pages).

### Silent empty pages

A parse that loses pages without complaint is worse than one that fails, so
empty pages are not left to the log: the page numbers are persisted with the
document and news of them is emitted once per file, not once per page.

Two different things produce an empty page, and they are recorded separately:

| Field | Meaning | Log |
|-------|---------|-----|
| `metadata.blank_pages` | Nothing on the page in the source — a separator, an empty annex, an intentional gap. `parsecraft inspect` reports `chars=0 images=0 blank=True`. | DEBUG |
| `metadata.unreadable_pages` | The page holds images but yielded no text, so content was probably lost. | WARNING |

`metadata.empty_pages` lists both, in page order. The split is not guessed: it
takes parsecraft's own per-page verdict, `PageRoute.degradation_code ==
"degraded-blank-page"` with score `0.0`, which parsecraft emits when it
degrades a page to native because there was *nothing to lose*. A page that
still comes back empty without that verdict is treated as unread.

```text
WARNING  scan-2024.pdf: 2 page(s) hold images but produced no text (4, 5).
         Routing said: page 4: no usable native text (blank/garbled); OCR pass docling;
         page 5: ... Engine said: OCR models are not installed (docling-ocr-unavailable)
```

A page can also come back empty *without* the adapter saying anything, and
parsecraft reports that on a third surface: a `degraded-empty-output` quality
signal (`score=0.0`, detail `"<backend> returned no content for a page that
needs OCR"`) on the document. It has no markdown projection, so the quality
list is where it has to be read from, and it is suppressed on any page whose
adapter already explained itself — the same page never reports both. kompend
quotes it as `Executor said: …` next to the routing reason.

The whole split, in parsecraft's own terms:

| Situation | Surface | Exit |
|-----------|---------|------|
| Routing refused OCR or chose native | `degraded-blank-page` page route | 0 |
| OCR-capable parser ran, returned nothing, adapter silent | `degraded-empty-output` quality signal | 0 |
| OCR-capable parser ran, returned nothing, adapter explained why | page `diagnostics` (`docling-ocr-unavailable`) | 0 |
| Every candidate raised (OOM, missing extra) | page `pipeline-all-passes-failed` diagnostic + document quality signal | 1 |

The last row is a hard failure, not an empty page: kompend raises instead of
storing a placeholder, so a crashed backend never looks like a blank page.

The quoted reason is parsecraft's, so the message names the actual cause
instead of suggesting a remedy that may already be installed:

- **`blank_pages`, nothing in the log**: nothing to do. No content was lost,
  and a corpus of standards documents is full of such pages.
- **`unreadable_pages` on a scanned document**: the OCR tier needs a backend
  that can run on this machine. Installing the `ocr` extra is not enough on
  its own — a GPU-only harness is excluded when the torch build has no CUDA, and
  `ocr-tesseract` needs a system engine. Check `parsecraft inspect`: it prints
  the per-page reason and the host's `installed_extras` / `gpu_usable` facts.
  A convert-tier backend that reads images itself (`docling`, `mineru`,
  `marker`) is the fallback, named with `only_backend = "docling"` (or
  `preferred_backend`) in `[routing]` alongside `allow_ocr = true`.
- **"But the OCR backends are installed"**: with the `ocr` extra, two further
  things decide whether a page is read. *Memory*: the shipped OCR backends
  rasterise at 300 dpi and parsecraft's own benchmarks show 4 of 5 requesting
  52–118 GiB on an 8 GB card (pc-da0), so an eligible harness can still OOM —
  `docling` reads scans on CPU and is the safe choice there (safe by
  availability, not by speed: `ocr-qianfan` also completes the fixture set,
  at 689–939 s per two-page document). *Model files*:
  docling-rs downloads layout and OCR weights separately, so a cache can hold
  `layout_heron.onnx` and none of `ocr_det.onnx`, `ocr_rec.onnx`,
  `ocr_rec_en.onnx`, `ocr_textline_ori.onnx`, `ppocr_keys_v1.txt` — pages then
  come back empty with no exception. That state is now reported: the warning
  quotes the engine's own `docling-ocr-unavailable` diagnostic.
- **Every page empty** on a file that clearly has text: the source is an image
  bundle, not a document — kompend raises rather than storing an empty document
  when no element at all survives the parse.

### Bounding a stalled parse

A backend can wedge and never return. Nothing inside the parser can interrupt
that, so `[ingest] parse_timeout_s` puts the bound outside the process: with a
positive value every ingest parse runs in a worker process that is killed when
the budget expires. The file is counted as failed, the run continues with the
next source, and the next run retries it because nothing was written to the
cache.

```toml
[ingest]
parse_timeout_s = 300      # seconds per source; 0 (default) = no bound
```

The worker is created once per run and reused, so the corpus pays one
interpreter start-up rather than one per file; a killed worker is replaced on
the next source. Two consequences worth knowing:

- A parse that legitimately needs longer than the budget is killed, so set it
  well above the slowest file you expect (a layout-aware parse of a long PDF can
  take minutes on CPU).
- `0` keeps the in-process path exactly as before, including its exception
types. Turn the bound on when a corpus has already stalled once.

Separately, the CLI caps the numeric thread pools before importing any numeric
library (`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` — default `min(8, cores)`, or
`KOMP_BLAS_THREADS`). OpenBLAS multiplies every large allocation by its thread
count, and on an allocation failure it **aborts the process** instead of raising
`MemoryError` — no handler, no traceback, just `OpenBLAS error: Memory allocation
still failed after 10 retries, giving up`. Capping the pool turns that abort
into slower arithmetic.

### Bounding a docling pass's memory

A docling conversion is the heaviest thing an ingest does. Its cost is **fixed
per pass, not per page**: docling-rs loads its layout, table-structure and OCR
models plus an ONNX arena inside the Rust extension, so one 8-page PDF and one
113-page PDF both peak around 4.2–5.4 GB of working set. Two knobs bound it,
and one default assumes them.

```toml
[ingest]
parsecraft_docling_threads = 4          # ONNX threads; default 4 on a big host
parse_min_free_commit_mb = 10240        # refuse a heavy parse below this; 0 = off
max_concurrent_heavy_parses = 1         # parallel docling/mineru passes

# ~/.config/parsecraft/config.toml or <wiki>/.kompend/parsecraft.toml
[routing.backend_options.docling]
do_ocr = false                          # only when the PDF has a text layer
do_table_structure = false              # skip the table model (~1.6 GB)
```

- **`parsecraft_docling_threads`** sets the engine's own
  `DOCLING_RS_PDF_THREADS` at parser construction, whether or not this run
  routes a page to docling (setting it late changes nothing — the engine reads
  it when it builds its converter). The default is `min(4, cores)`. Measured on
  Windows with docling-rs 2.65.1, one 8-page PDF: engine default 7.1 s /
  2790 MB peak commit, 4 threads 7.4 s / 1830 MB, 1 thread 16.3 s / 1533 MB —
  four is the knee. An explicit `DOCLING_RS_PDF_THREADS` in the environment
  wins over the config key.
- **`[routing.backend_options]`** (in parsecraft's config) forwards options to
  one sub-backend's converter. Skipping the OCR and table models on a
  text-layer PDF is the single biggest win measured: 8.0 GB / 23.6 s down to
  2.9 GB / 10.3 s. Only do it for sources with real text; a scan needs both
  models.
- **`parse_min_free_commit_mb`** and **`max_concurrent_heavy_parses`** are the
  guard, not the cure: a heavy parse (a non-text source: docling, mineru,
  pandoc, pymupdf) only starts with that much free commit charge, and at most
  that many run at once. A source that does not get the room is logged as
  skipped (`Skipped <file>: …`) and retried next run — never silently dropped,
  and never allowed to take the whole run down. Light sources (native text,
  code) are unaffected.

With GPU-accelerated docling there is nothing to tune yet on Windows: the PyPI
wheel is CPU-only. `docling-rs-cuda` exists for Linux x86_64 only, and the
Windows binary ships DirectML DLLs but was built without the provider enabled,
so `DOCLING_RS_EP=directml` answers `built without that provider … using CPU`.

### Judge tier: model-assisted per-page routing

By default parsecraft routes with its **deterministic rule table** (format,
VRAM budget, page intent) and never calls a model. `[ingest] parsecraft_judge`
opts into a *judge*: a System One model that re-ranks the candidates the hard
constraints already admitted, so a text-native PDF stays on the fast path
while a scanned, table-heavy one is sent to a strong backend — the speed vs.
accuracy trade-off in [Parser Strategy](#parser-strategy-speed-vs-accuracy),
decided per page instead of per configuration.

```toml
[ingest]
parsecraft_judge = "ollama"            # none | ollama | zen | typesafe-ai
parsecraft_judge_model = "parable-tinyjev"

# ~/.config/parsecraft/config.toml or <wiki>/.kompend/parsecraft.toml
[routing]
preference = "balanced"                # speed | balanced | quality
# allow_ocr = false                    # unset = derive from extras/engine; false bans OCR
# engines = ["pandoc"]                 # unset = the engines this host has

[routing.backend_options.mineru]
effort = "flash"                       # flash | medium | high | xhigh
[routing.backend_options.liteparse]
extract_vector_graphics = true         # recover table grids from vector records
```

- **`ollama`** is the offline choice: it talks to your local daemon
  (`http://localhost:11434`), needs no credential, and the model must be a
  *decision* model — ollama must list `decision` under
  `ollama show <model>` → `capabilities`. Completion-only GGUFs are rejected
  by the daemon with `400 … does not support decision`. Use the bare ollama
  name without the `:tag` suffix (`parable-tinyjev`, not
  `parable-tinyjev:latest`): the SDK reads the suffix as a variant token and
  refuses it. The SDK is base-install with parsecraft (≥ 2026.10.5), so no
  extra is needed.
- **`zen`** (`OPENCODE_API_KEY`) and **`typesafe-ai`** (`TYPESAFE_API_KEY`) are
  the cloud choices; the credential is read from the environment only.
- **`[routing] preference`** is the tiebreak axis *inside* a backend family,
  after the hard constraints (format, GPU/VRAM budget, installed extras) have
  admitted the candidates. `speed` and `balanced` coincide by design — both
  take the candidate with the smallest `estimated_vram_gb` — while `quality`
  inverts the tiebreak to the strongest. Two routing outcomes, not three. The
  axis applies with or without a provider judge: parsecraft's own
  `resolve_judge(None, preference=…)` constructs the deterministic judge with
  it, and it is read from the same config either way.
- **`[routing.backend_options.mineru] effort`** picks the MinerU sub-backend's
  conversion effort (`flash` | `medium` | `high` | `xhigh`, default `flash`).
  `flash` is the weight-free CPU path for text PDFs; `high` and `xhigh` engage
  MinerU's VLM path and recover equations, tables and figure captions `flash`
  omits, at the cost of loading the VLM checkpoint. It is part of the
  parse-cache fingerprint (a change re-parses) and reaches the backend as a
  per-backend constructor option.
- **`[routing] allow_ocr`** decides whether parsecraft may route a page to an
  OCR sub-backend. Unset, kompend supplies the probe's answer: OCR is permitted
  when an `ocr-*` extra is installed **or** the host has a usable OCR engine.
  Set `false` to ban OCR outright — an all-native corpus should, because a blank
  or faint page otherwise pays a full GPU VLM pass — or `true` to permit it
  regardless of the probe. The value is part of the parse-cache fingerprint
  whenever it is written.
- **`[routing] preferred_backend`** ranks one backend first where it is
  eligible — a preference, never a guarantee: a page the named backend cannot
  serve still falls back. **`[routing] only_backend`** is the hard pin: pages
  the named backend cannot *cover* fail loudly instead of silently falling
  back, while a blank page still degrades (coverage and content are different
  things). Both are validated against parsecraft's registered backends at
  config load and fold into the parse-cache fingerprint.

  First-parse warmup: the first document after flipping to a heavier backend
  pays the backend's own one-time cost — model checkpoint load, and download
  where the weights are not cached — which can be tens of seconds per page.
  That is warmup, not steady state; subsequent documents run at the backend's
  normal per-page cost. Judge the setting by the second run, not the first.
- **`[routing] engines`** restricts the external engines parsecraft may use,
  overriding the probed set. Unset (the default) keeps the probe's answer.
  Names are parsecraft's own engine ids — this is open capability data, not a
  closed value domain, so it is a list of strings rather than an enum, and an
  unknown name fails at load. Part of the parse-cache fingerprint.
- **`[routing.backend_options.liteparse] extract_vector_graphics`** asks the
  liteparse backend to recover table structure from a page's vector graphics.
  With it on, parsecraft populates `StructuredChunk.rows` and a table arrives
  as structured `list[list[str]]` instead of a Markdown grid; without it,
  tables stay Markdown and the markdown converter rebuilds the grid
  downstream. A page with no vector records still yields no table, and the
  option is inert on pages another backend serves. Keys and value domains
  belong to each sub-backend; kompend forwards them verbatim and declares no
  option of its own except the derived `liteparse.extract_images`, which
  follows `compile.image_mode` and yields to a written value.

A judge never turns a routing problem into a failed ingest: an unresolvable
provider or missing credential degrades to the rule table at resolution time,
and a judge that resolves but fails on its first call is retried once with the
rule table and then remembered for the rest of the run (one `WARNING` per run).
The judge configuration is part of the parse-cache fingerprint, so
changing it re-parses rather than silently reusing pages routed under the old
settings.

### Error mapping

parsecraft raises typed `AnalysisError` subclasses; the adapter maps them to
kompend behaviour:

| parsecraft exception | kompend behaviour |
|---|---|
| `UnsupportedSourceError` | Input — skip the file with a warning naming the suffix |
| `NoAnalyzerError` | Dependency/config — "install the matching extra" |
| `DependencyUnavailableError` | Dependency/config — "install the named extra" |
| `BackendError` (other) | Backend failure with cause preserved |
| `RoutingError` | kompend-side error with cause preserved |
| Total exhaustion (`pipeline-all-passes-failed`) | `ParsecraftParseError` with per-page cause |

Exit codes from the parsecraft CLI are **not** part of the contract — kompend
imports `parsecraft.pipeline` as a library, so typed exceptions cross the
boundary, never exit codes.

### Limitations

- Adds an external dependency with a minimum parsecraft release; upgrading
  parsecraft is a coordinated event, not a silent update.
- Per-page routing and fallback are entirely parsecraft's responsibility —
  kompend cannot intervene mid-pipeline.
- Adapter calls `execute()` **serially**, matching parsecraft's 8 GB one-
  instance discipline (`pipeline/AGENTS.md`). kompend-level parallelism across
  documents is preserved (the existing parse worker pool is unchanged).

## Investigated and Rejected: opendataloader-pdf[hybrid]

### Why it looked promising

[opendataloader-pdf](https://github.com/opendataloader-project/opendataloader-pdf)
(Apache 2.0) ranks #1 in public benchmarks with an overall score of 0.907
versus docling's 0.882. Its "hybrid" mode routes complex pages (tables,
scanned documents, formulas, charts) to an AI backend for significantly
better accuracy than the purely local mode.

### Why it was rejected

Despite the accuracy advantage, operational concerns outweigh the benefit:

### 1. Java 11+ runtime requirement

The package is 85.9% Java. Every kompend user would need a JDK installed and
`JAVA_HOME` configured — a serious friction point for a project that is
otherwise pure Python.

### 2. Separate hybrid-mode server

Hybrid mode requires a background process:

```bash
opendataloader-pdf-hybrid --port 5002
```

kompend would need to spawn, monitor, and shut down this server — error-prone
on Windows, and a crash would cause silent parse failures.

### 3. JVM-per-call overhead

Each `convert()` invocation spawns a new JVM process. When ingesting
dozens of documents, this adds seconds of overhead per file.

### 4. PDF-only

opendataloader handles only PDF. kompend needs multi-format support
(DOCX, PPTX, HTML, MD, EPUB, etc.) — all handled natively by docling.

### 5. Different data model

opendataloader's JSON output is page-oriented with bounding boxes.
Docling's model is content-oriented (document tree). Supporting both
would require a second adapter layer, increasing maintenance surface.

### 6. Marginal real-world gain

The benchmark gap (0.907 vs 0.882) is 2.8%. Docling actually beats
opendataloader on heading accuracy (0.824 vs 0.821). For kompend's use
case — extracting structured content for LLM consumption, not visual
page layout — docling's accuracy is sufficient.

### Future consideration

If docling's PDF accuracy proves insufficient for specific document types,
the parser infrastructure supports plugging in alternatives. A future
version could add opendataloader as an optional PDF backend — but the
operational complexity (Java, sidecar server, JVM overhead) means this
should only be done when there is clear evidence of content loss, not
based on benchmarks alone.

## See also

- [Attachments](attachments.md) — Supplementary files per source
  (photos, binaries, source code) with opt-in ingestion.
