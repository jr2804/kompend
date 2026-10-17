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

parsecraft is the only parser path; the selector is gone. The remaining
`[ingest]` keys are the parsecraft parameters
(`parsecraft_judge`, `parsecraft_judge_model`, `parsecraft_preference`,
`parsecraft_effort`, `parsecraft_allow_ocr`, `parsecraft_engines`,
`parsecraft_extract_vector_graphics`, `parsecraft_preferred_backend`,
`parsecraft_only_backend` —
see [Judge tier](#judge-tier-model-assisted-per-page-routing)).

- `parsecraft` — an external document-intelligence pipeline that routes each
  page across its own sub-backends (native text/markdown/HTML, LiteParse,
  docling, pandoc, OCR extras) and returns one document per source. It is a
  **core dependency** of kompend, together with its `docling` and `web`
  sub-backends, so installing kompend installs it.

`kompend list-parsers` shows the backend and the formats it claims.

Two opt-in extensions ride on top of parsecraft rather than being kompend
backends:

- **`kompend[mineru]`** — adds MinerU 4.x as a parsecraft sub-backend
  (`parsecraft[mineru]>=2026.10.8`), eligible for layout-heavy and OCR
  pages. The extra opts into MinerU's own licence terms — read the note in
  `pyproject.toml` before using it commercially.
- **marker (bring-your-own)** — `marker-pdf` cannot be locked or pinned by
  kompend at all (see [below](#why-marker-cannot-be-a-locked-extra)); install
  it yourself and parsecraft counts it among its routing candidates.

The judge tier needs no extra: `typesafe-sdk` — shared by the `ollama`, `zen`
and `typesafe-ai` providers — is base-install in parsecraft ≥ 2026.10.5, so it
arrives with kompend itself. The judge is enabled purely by config
(`[ingest] parsecraft_judge`, `parsecraft_judge_model`,
`parsecraft_preference`, `parsecraft_effort`, `parsecraft_allow_ocr`,
`parsecraft_engines`, `parsecraft_extract_vector_graphics`,
`parsecraft_preferred_backend`, `parsecraft_only_backend`).

### Why `marker` cannot be a locked extra

Measured fresh (kompend-3, 2026-10-10) against the current lock: the tight pin
`marker-pdf>=1.10.2,<2` **hard-fails** — `marker-pdf` 1.10.2 pins
`anthropic>=0.46,<0.47` while `pydantic-ai` 2.31.1 needs `anthropic>=0.108`,
and the two ranges do not overlap. Dropping the upper bound only makes it
"resolve" by downgrading `anthropic` 1.13.0 → 0.125.0 and `pillow`
12.3.0 → 10.4.0 — silently regressing kompend's LLM stack.

That is the substantive reason the marker backend was deleted rather than
kept behind an extra (ADR-0006): kompend cannot own the `marker-pdf` pin, so
marker is parsecraft's **bring-your-own** sub-backend.

## Supported Formats

Discovery takes the registry default (parsecraft) and asks it which suffixes
it claims; the source directory is scanned for exactly those. parsecraft's
current 32-suffix table is what a default `kompend ingest` covers. `.kompend/`
is excluded regardless of claimed formats, so state artifacts such as
`.kompend/rumdl.toml` are never ingested even though parsecraft claims `.toml`.

| Format | Handled by | Notes |
|--------|------------|-------|
| `.pdf` | `parsecraft` | Routed per page across its sub-backends; scanned pages escalate to an OCR backend when one is installed |
| `.docx`, `.pptx`, `.xlsx` | `parsecraft` | Office formats via its sub-backends |
| `.epub`, `.odt`, `.rtf` | `parsecraft` | E-book and legacy text containers |
| `.md`, `.markdown`, `.html`, `.htm` | `parsecraft` | Native text and markup backends |
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
routing rather than complement it, so kompend keeps none. An unknown backend name
raises; a known backend whose optional dependency is missing raises with its
install command — a vault whose configured parser is absent fails loudly
instead of parsing nothing.

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
uv add kompend    # pulls parsecraft[docling,web]>=2026.10.8
```

The floor is load-bearing: 2026.10.5 moved `auto`, `liteparse`, `pandoc` and
`openpyxl` into parsecraft's base install, 2026.10.6 split the old `pdf` extra
into `pymupdf` and `pdf-full`, and 2026.10.8 made `extract_vector_graphics`
usable — its `VectorGraphics` read was dict-style on a dataclass until then, so
the option crashed the backend and the table path was unreachable. Raising it
is a reviewed event (see `.agents/plans/parsecraft-integration.md`). kompend
never imports parsecraft internals, private modules, or test helpers — only the
documented public surface (`parsecraft.pipeline` / `parsecraft.backends` / IR).

!!! warning "A resolution failure moments after a release is usually a stale index cache"

    `No solution found … because there is no version of kompend[pdf-full]==<version>`
    seconds after a release is almost always a stale index cache, not a missing
    artifact. Check the version on PyPI, then retry with the cache bypassed and an
    explicit version:

    ```bash
    UV_NO_CACHE=1 uv pip install --force-reinstall "kompend[pdf-full]==2026.10.15"
    ```

    The message is word-for-word what a genuinely empty release produces, so
    confirm it against PyPI's JSON before concluding the release itself broke.
    Note PyPI answers inconsistently while a release propagates: the
    `…/<package>/json` endpoint can still report the previous version, while
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
suffixes it handles (`get_all_extensions([config.parser])`). With parsecraft
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

Four classes of parsecraft signal survive the parse into `.kompend/kompend.db`:

- `metadata.parser` — `"parsecraft"`.
- `metadata.backends` — comma-separated list of parsecraft sub-backends that
  won each page group (parse provenance).
- `metadata.quality_signals` — names of `QualitySignal` entries parsecraft
  attached to the document (route degradation, missing extras, etc.).
- `metadata.degraded_pages` — page numbers whose route was downgraded inside
  parsecraft (e.g. OCR → native text because no eligible OCR backend was
  installed). This is the trace that proves a parse silently downgraded,
  rather than hiding the choice behind a structurally identical result.

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
parsecraft_preference = "balanced"     # speed | balanced | quality
parsecraft_effort = "flash"            # flash | medium | high | xhigh (MinerU sub-backend)
# parsecraft_allow_ocr = false         # unset = derive from extras/engine; false bans OCR
# parsecraft_engines = ["mineru"]      # empty = keep the probed engine set
# parsecraft_extract_vector_graphics = true  # recover table grids from vector records
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
- **`parsecraft_preference`** is the tiebreak axis *inside* a backend family,
  after the hard constraints (format, GPU/VRAM budget, installed extras) have
  admitted the candidates. `speed` and `balanced` coincide by design — both
  take the candidate with the smallest `estimated_vram_gb` — while `quality`
  inverts the tiebreak to the strongest. Two routing outcomes, not three. The
  axis applies with or without a provider judge: parsecraft's own
  `resolve_judge(None, preference=…)` constructs the deterministic judge with
  it, and kompend passes it through on the no-judge path too.
- **`parsecraft_effort`** picks the MinerU sub-backend's conversion effort
  (`flash` | `medium` | `high` | `xhigh`, default `flash`). `flash` is the
  weight-free CPU path for text PDFs; `high` and `xhigh` engage MinerU's VLM
  path and recover equations, tables and figure captions `flash` omits, at the
  cost of loading the VLM checkpoint. The key is schema-validated and folds
  into the parse-cache fingerprint (a change re-parses). It reaches the backend
  as a per-backend constructor option, injected only when non-default so an
  explicit `flash` does not churn the conversion cache.
- **`parsecraft_allow_ocr`** overrides whether parsecraft may route a page to an
  OCR sub-backend. Unset, parsecraft derives it: OCR is permitted when an
  `ocr-*` extra is installed **or** the host has a usable OCR engine. Set
  `false` to ban OCR outright — an all-native corpus should, because a blank or
  faint page otherwise pays a full GPU VLM pass — or `true` to permit it
  regardless of the derivation. The value is part of the parse-cache
  fingerprint, and it reaches the planner immediately.
- **`parsecraft_preferred_backend`** ranks one backend first where it is
  eligible — a preference, never a guarantee: a page the named backend cannot
  serve still falls back. **`parsecraft_only_backend`** is the hard pin: pages
  the named backend cannot *cover* fail loudly instead of silently falling
  back, while a blank page still degrades (coverage and content are different
  things). Both are validated against parsecraft's registered backends at
  config load and fold into the parse-cache fingerprint.

  First-parse warmup: the first document after flipping to a heavier backend
  pays the backend's own one-time cost — model checkpoint load, and download
  where the weights are not cached — which can be tens of seconds per page.
  That is warmup, not steady state; subsequent documents run at the backend's
  normal per-page cost. Judge the setting by the second run, not the first.
- **`parsecraft_engines`** restricts the external engines parsecraft may use,
  overriding the probed set. Empty (the default) keeps the probe's answer.
  Names are parsecraft's own engine ids and the env form is a comma-separated
  list — this is open capability data, not a closed value domain, so it is a
  list of strings rather than an enum. Part of the parse-cache fingerprint.
- **`parsecraft_extract_vector_graphics`** asks the liteparse backend to recover
  table structure from a page's vector graphics. With it on, parsecraft
  populates `StructuredChunk.rows` and a table arrives as structured
  `list[list[str]]` instead of a Markdown grid; without it, tables stay
  Markdown and the markdown converter rebuilds the grid downstream. A page with
  no vector records still yields no table, and the option is inert on pages
  another backend serves.

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
