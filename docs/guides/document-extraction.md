# Document Extraction in KNOX

KNOX ("eXtended documents") aims to parse a wide variety of document formats
into a structured, LLM-aware representation. This document explains which
frameworks are used, why, and what was considered before settling on the
current architecture.

See also:

- [Architecture overview](../architecture/index.md) — 5-tier pipeline
- [Pipeline Tiers](../concepts/pipeline-tiers.md) — Per-tier component reference
- [LLM Configuration](llm-config.md) — Provider setup and model selection

## Parser Backends

`[ingest] parser` names exactly one backend.

- `parsecraft` — the default. An external document-intelligence pipeline that
  routes each page across its own sub-backends (native text/markdown/HTML/PDF,
  LiteParse, docling, pandoc, OCR extras) and returns one document per source.
  It ships as the `parsecraft` extra, which pulls the sub-backends that make it
  useful: `uv add knox[parsecraft]`.
- `web` — trafilatura HTML parser, installed with the core package.

Two further backends exist but cannot be locked extras — their transitive
dependencies conflict (pillow, transformers, ONNX) with knox's LLM stack.
Install them manually:

```bash
uv pip install marker-pdf   # deep-learning PDF/office/HTML
uv pip install mineru       # layout-aware PDF/Office with OCR (torch, ONNX)
```

Then select the one you want:

```toml
[ingest]
parser = "parsecraft"   # or "web" | "marker" | "mineru"
```

`knox list-parsers` shows which backends are installed and the formats each
claims. An unknown name raises; a known backend whose optional dependency is
missing raises with its install command, so a vault configured against an
absent parser fails loudly instead of parsing nothing.

The judge tier needs the `systemone` extra on top of parsecraft:

```bash
uv add knox[systemone]
```

> **Installing marker/mineru into a knox project:** run the `uv pip install`
> commands *outside* the project's resolved environment (a separate venv or a
> container), then point knox at that environment. They cannot be added to the
> project's dependency graph without regressing the core LLM stack — see
> [Why `marker` cannot be a locked extra](#why-marker-cannot-be-a-locked-extra).

### Why `marker` cannot be a locked extra

Measured against the current lock, `marker-pdf` 2.0.0 resolves only by
downgrading eight packages, five of them backwards past what knox's LLM stack
requires:

| Package | knox | marker forces | Declared by | Movable? |
|---------|------|---------------|-------------|----------|
| `anthropic` | 1.8.0 | **<1** | `marker-pdf` | ❌ |
| `openai` | 3.19.0 | **<3** | `marker-pdf` | ❌ |
| `pillow` | 12.3.0 | `<11` | `marker-pdf` | ✅ |
| `opencv-python-headless` | 5.0.0.93 | `==4.11.0.86` | `surya-ocr` | ✅ relax our floor |
| `pypdfium2` | 5.13.0 | `==5.10.1` | `pdftext` | ✅ above our floor |
| `filelock` | 4.0.1 | `<4` | `surya-ocr` | ✅ |
| `pydantic-ai` | 2.48.0 | 2.31.1 | consequence of `anthropic<1` | ❌ |
| `pydantic-ai-harness` | 0.34.0 | 0.24.0 | consequence of `anthropic<1` | ❌ |

`surya-ocr` and `pdftext` are listed in marker's own `pyproject.toml`, so this
is one hop away, not transitive.

The two unmovable conflicts are `marker-pdf` itself: `anthropic>=0.69.0,<1`
and `openai>=2.2.0,<3`. Their intersection with `pydantic-ai`'s
`anthropic>=1.8.0` and `openai>=3.8.0` (declared in `pydantic-ai`'s own
default `pydantic-ai-slim[anthropic,openai]`) is **empty** — verified with
`uv pip install "anthropic>=1.8.0,<1"` and the same for `openai`. Relaxing
our own `opencv-python-headless`/`pypdfium2` floors doesn't help: those
remove two conflicts and leave the two that make a resolution impossible.
`mineru` installs cleanly and remains the usable manual backend; `marker` is
usable only if the downgrades above are acceptable.

## Supported Formats

Discovery follows the configured backend: knox asks `[ingest] parser` which
suffixes it claims (`get_all_extensions([config.parser])`) and scans the source
directory for exactly those. With the default backend that is parsecraft's
31-suffix table. `.knox/` is excluded regardless of claimed formats, so state
artifacts such as `.knox/rumdl.toml` are never ingested even though parsecraft
claims `.toml`.

| Format | Handled by | Notes |
|--------|------------|-------|
| `.pdf` | `parsecraft` | Routed per page across its sub-backends; scanned pages escalate to an OCR backend when one is installed |
| `.docx`, `.pptx`, `.xlsx` | `parsecraft` | Office formats via its sub-backends |
| `.epub`, `.odt`, `.rtf` | `parsecraft` | E-book and legacy text containers |
| `.md`, `.markdown`, `.html`, `.htm` | `parsecraft` | Native text and markup backends |
| `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` | `parsecraft` | Images; content comes from an OCR/VLM sub-backend when installed |
| `.txt`, `.text`, `.log`, `.rst`, `.ini`, `.cfg`, `.toml`, `.c`, `.h`, `.hpp`, `.cpp`, `.cc`, `.py`, `.sh`, `.tex` | `parsecraft` | Plain-text source family |
| `.html` | `web` | Trafilatura reader; the only suffix `web` claims |

`marker` and `mineru`, when installed and selected, claim their own sets.
`knox list-parsers` prints the live extension list per installed backend. A
file whose suffix no installed backend claims is never picked up by discovery.

## Parser Strategy: Speed vs. Accuracy

parsecraft is the default backend, and it does not pick one engine for a whole
corpus: it analyses each page and routes it, so a born-digital text page stays
on the cheap native/LiteParse path while a scanned or table-heavy page in the
same document can escalate to a stronger backend. The speed/accuracy trade-off
is decided per page rather than per configuration.

`[ingest] parser` selects the single backend knox hands files to:

```toml
[ingest]
parser = "parsecraft"
```

### One backend, no chain

There is no comma-separated list and no `parser_fallback`. parsecraft performs
its own per-page routing across its registered sub-backends (native, docling,
liteparse, pandoc, OCR), and produces one `ParsedDocument` per source however
many internal passes fired. Degraded pages (parsecraft downgraded an OCR route
to native text) surface as KNOX metadata (`degraded_pages`), so the choice stays
visible after the parse.

A second knox-side level of ordering and hand-off would compete with that
routing rather than complement it, so knox keeps none. An unknown backend name
raises; a known backend whose optional dependency is missing raises with its
install command — a vault whose configured parser is absent fails loudly
instead of parsing nothing.

## Backend Comparison: Strengths and Weaknesses

> Capabilities below reflect what each backend **actually emits into the
tier-1 element model** through knox's shared Markdown converter, not just what
the upstream project advertises. Backends that mark a capability "—" produce
that content in their Markdown, but knox previously discarded it (see
[Part-level markdown normalisation](#part-level-markdown-normalisation)).

### Capabilities at a glance

| Backend | Figures | Tables | Equations (LaTeX) | Code blocks | OCR engine | Speed |
|---------|---------|--------|-------------------|-------------|------------|-------|
| `parsecraft` | yes | yes | yes | yes | per page, when an OCR sub-backend is installed | medium |
| `marker` | yes | yes | yes | yes | block OCR | medium (GPU-friendly) |
| `mineru` | yes | yes | yes | yes | layout + OCR | medium–slow |
| `web` (trafilatura) | no | yes | n/a | yes | no | fastest |

`parsecraft`'s row is the union of its sub-backends: which of figures, tables,
equations and OCR it can actually produce depends on the extras present.

### Where each backend wins

| Backend | Strong points | Weak points |
|---------|---------------|-------------|
| `parsecraft` | Per-page routing instead of one engine per corpus: a born-digital page stays on the cheap native/LiteParse path while a scanned or table-heavy page in the same file escalates. Inherits selective OCR, per-page fallback passes and route-degradation metadata. | Adds an external dependency with a minimum parsecraft release; routing is parsecraft's to decide and knox cannot intervene mid-pipeline. On parsecraft 2026.10.1 the candidate set is narrowed by its own installed-extra probe (see [Limitations](#limitations)). |
| `marker` | Highest reported benchmark accuracy of the selectable backends; equations and tables as first-class output; device-aware modes; optional LLM processors for extra quality when an LLM service is configured. | Cannot be a locked extra — its own `anthropic<1` / `openai<3` caps contradict `pydantic-ai`'s `anthropic>=1.8.0` / `openai>=3.8.0` (see [above](#why-marker-cannot-be-a-locked-extra)); model download + GPU strongly preferred; slowest on CPU; internal figure captions only via the opt-in LLM path. |
| `mineru` | Layout-aware with selectable backends (`pipeline` OCR + layout, `vlm`, `hybrid`); `formula_enable`/`table_enable` on by default; parse-method control (`auto`/`ocr`/`txt`) lets you force or skip OCR per corpus. | Cannot be a locked extra; heavy install (torch + ONNX); its Markdown carries no page markers, so `page_count` is reported as unknown (`0`). |
| `web` (trafilatura) | Purpose-built HTML reader: strips chrome/nav/boilerplate and returns clean GFM (headings, lists, tables, links, code) much faster than a PDF parser on HTML. | HTML only; no figures (image refs are dropped — knox's web parser has no image resolver), no OCR, no equations. |

### Choosing

- **Default:** `parsecraft` — per-page routing covers text-native, office and
  scanned documents with one configuration.
- **Highest local accuracy on hard layouts:** `marker` (manual install, GPU
  preferred).
- **Image-only or OCR-heavy corpora:** `parsecraft` with an OCR sub-backend
  installed, or `mineru` (`parse_method = "ocr"`).
- **Web pages:** `web`.

## Part-level markdown normalisation

Every backend that produces Markdown after its SDK call goes through one shared
converter (`knox/preprocessing/parsers/markdown.py`) that maps the Markdown into
tier-1 elements. It recognises headings, images, tables, **equations** (`$$…$$`,
`\[…\]`), **fenced code** (with language), and **list items**.

Before that consolidation each backend had its own copy of the walk, and all of
them recognised only headings, images, tables and paragraphs — so equations,
code and list items from backends that *do* emit them (marker, mineru) were
folded into `TEXT` and reached the chunker as prose.

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
document-intelligence pipeline (public API only, separate project from KNOX).
KNOX consumes it through a single adapter (`preprocessing/parsers/parsecraft_parser.py`)
and maps parsecraft's intermediate representation onto KNOX's element model.
Install it as an optional extra:

```bash
uv add knox[parsecraft]    # carries parsecraft>=2026.10.1 (Python ≥3.13)
```

The extra carries a minimum parsecraft release; raising it is a reviewed event
(see `.agents/plans/parsecraft-integration.md`). KNOX never imports parsecraft
internals, private modules, or test helpers — only the documented public
surface (`parsecraft.pipeline` / `parsecraft.backends` / IR).

### What selecting parsecraft does

Set `ingest.parser = "parsecraft"` to route every file it claims through
parsecraft's full pipeline:

```toml
[ingest]
parser = "parsecraft"   # the ingestion backend; no chain, no fallback list
```

The adapter calls `pipeline.execute()` — analyze → `plan_route` → per-range
dispatch with fallback passes → aggregate `DocumentResult` — so routing,
selective OCR escalation, and per-range fallback passes are inherited rather
than reimplemented in KNOX. Figure enrichment is on (`figure_enrichment = True`)
and parsecraft's own figure extraction flows through `VLMEnricher` like every
other backend's.

### Formats parsecraft claims

parsecraft speaks MIME (`text/plain`, `application/pdf`, `image/jpeg`, …);
KNOX discovers files by extension, and asks the configured backend which
suffixes it handles (`get_all_extensions([config.parser])`). With parsecraft
that is its 31-suffix table — `.txt`, `.text`, `.log`, `.rst`, `.ini`, `.cfg`,
`.toml`, `.c`, `.h`, `.hpp`, `.cpp`, `.cc`, `.py`, `.sh`, `.tex`, plus the
usual PDF/HTML/image set — so a parsecraft vault can ingest source files. A
PDF-only backend would contribute no text suffixes at all. `.knox/` is excluded
from source discovery regardless of claimed formats: state artifacts like
`.knox/rumdl.toml` are never ingested even though parsecraft claims `.toml`.

### Fallback interaction

There is no KNOX-level chain. One backend is configured, and parsecraft runs
its own per-page routing and fallback passes between its registered
sub-backends (native text/markdown/html/pdf, OCR backends, LiteParse); the
result is one `ParsedDocument` per source regardless of how many internal
passes fired.

If parsecraft exhausts every pass on a page, the adapter raises
`ParsecraftParseError` carrying the per-page cause, and the parse fails — there
is no second backend behind it to fall through to.

### Diagnostics carried into KNOX metadata

Four classes of parsecraft signal survive the parse into `.knox/knox.db`:

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
parser = "parsecraft"
parsecraft_judge = "ollama"            # none | ollama | zen | typesafe-ai
parsecraft_judge_model = "parable-tinyjev"
parsecraft_preference = "balanced"     # speed | balanced | quality
```

- **`ollama`** is the offline choice: it talks to your local daemon
  (`http://localhost:11434`), needs no credential, and the model must be a
  *decision* model — ollama must list `decision` under
  `ollama show <model>` → `capabilities`. Completion-only GGUFs are rejected
  by the daemon with `400 … does not support decision`. Use the bare ollama
  name without the `:tag` suffix (`parable-tinyjev`, not
  `parable-tinyjev:latest`): the SDK reads the suffix as a variant token and
  refuses it. `uv add knox[systemone]` provides the SDK.
- **`zen`** (`OPENCODE_API_KEY`) and **`typesafe-ai`** (`TYPESAFE_API_KEY`) are
  the cloud choices; the credential is read from the environment only.
- **`parsecraft_preference`** is the tiebreak axis among candidates that pass
  the constraint check — `speed` prefers the cheapest eligible backend,
  `quality` the strongest, `balanced` (default) capability bounded by the
  host budget.

A judge never turns a routing problem into a failed ingest: an unresolvable
provider or missing credential degrades to the rule table at resolution time,
and a judge that resolves but fails on its first call is retried once with the
rule table and then remembered for the rest of the run (one `WARNING` per run).
The judge configuration is part of the parse-cache fingerprint, so
changing it re-parses rather than silently reusing pages routed under the old
settings.

### Error mapping

parsecraft raises typed `AnalysisError` subclasses; the adapter maps them to
KNOX behaviour:

| parsecraft exception | KNOX behaviour |
|---|---|
| `UnsupportedSourceError` | Input — skip the file with a warning naming the suffix |
| `NoAnalyzerError` | Dependency/config — "install the matching extra" |
| `DependencyUnavailableError` | Dependency/config — "install the named extra" |
| `BackendError` (other) | Backend failure with cause preserved |
| `RoutingError` | KNOX-side error with cause preserved |
| Total exhaustion (`pipeline-all-passes-failed`) | `ParsecraftParseError` with per-page cause |

Exit codes from the parsecraft CLI are **not** part of the contract — KNOX
imports `parsecraft.pipeline` as a library, so typed exceptions cross the
boundary, never exit codes.

### Limitations

- Adds an external dependency with a minimum parsecraft release; upgrading
  parsecraft is a coordinated event, not a silent update.
- Per-page routing and fallback are entirely parsecraft's responsibility —
  KNOX cannot intervene mid-pipeline.
- Adapter calls `execute()` **serially**, matching parsecraft's 8 GB one-
  instance discipline (`pipeline/AGENTS.md`). KNOX-level parallelism across
  documents is preserved (the existing parse worker pool is unchanged).
- **The judge can only rank candidates the environment probe admitted.**
  parsecraft 2026.10.1 detects installed backends through a declared map
  (`parsecraft/environment/probe.py: EXTRA_IMPORTS`) whose
  `optional_dependency_group` value it does not list is treated as *not
  installed* — `extra_present("docling")` returns `False` even when the
  `docling` module resolves, because the map has no `"docling"` entry (nor
  `"pandoc"`). With docling filtered out, a PDF has exactly one candidate
  (`liteparse`) and the judge has nothing to choose, so `parsecraft_judge`
  changes nothing. Confirm the candidate set before reading anything into a
  judge run: an empty `metadata.degraded_pages` and an unchanged
  `metadata.backends` are expected, not a regression.

## MinerUParser: Layout-Aware PDF/Office Parsing

[MinerU](https://github.com/opendatalab/MinerU) (Apache 2.0) is a
high-accuracy document parser with three selectable backends:

- **pipeline** (default) — traditional OCR + layout-detection model.
- **vlm** — vision-language model backend (needs `mineru-vl-utils` + GPU server).
- **hybrid** — pipeline for text-heavy pages, VLM for complex layouts.

MinerU handles PDF, DOCX, PPTX, and XLSX. It has no internal figure
captioning, so `figure_enrichment` is `False` and every figure flows through
`VLMEnricher`. The parser drives MinerU's `do_parse` API, which converts a
source file to Markdown (with extracted images), then walks the Markdown into
KNOX `DocumentElement` values.

MinerU requires heavy dependencies (torch, transformers, ONNX models). It is
**not** installed by default — install it manually:

```bash
uv pip install mineru
```

Then select it: `parser = "mineru"`.

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

The package is 85.9% Java. Every KNOX user would need a JDK installed and
`JAVA_HOME` configured — a serious friction point for a project that is
otherwise pure Python.

### 2. Separate hybrid-mode server

Hybrid mode requires a background process:

```bash
opendataloader-pdf-hybrid --port 5002
```

KNOX would need to spawn, monitor, and shut down this server — error-prone
on Windows, and a crash would cause silent parse failures.

### 3. JVM-per-call overhead

Each `convert()` invocation spawns a new JVM process. When ingesting
dozens of documents, this adds seconds of overhead per file.

### 4. PDF-only

opendataloader handles only PDF. KNOX needs multi-format support
(DOCX, PPTX, HTML, MD, EPUB, etc.) — all handled natively by docling.

### 5. Different data model

opendataloader's JSON output is page-oriented with bounding boxes.
Docling's model is content-oriented (document tree). Supporting both
would require a second adapter layer, increasing maintenance surface.

### 6. Marginal real-world gain

The benchmark gap (0.907 vs 0.882) is 2.8%. Docling actually beats
opendataloader on heading accuracy (0.824 vs 0.821). For KNOX's use
case — extracting structured content for LLM consumption, not visual
page layout — docling's accuracy is sufficient.

### Future consideration

If docling's PDF accuracy proves insufficient for specific document types,
the parser infrastructure supports plugging in alternatives. A future
version could add opendataloader as an optional PDF backend — but the
operational complexity (Java, sidecar server, JVM overhead) means this
should only be done when there is clear evidence of content loss, not
based on benchmarks alone.

## Marker Parser

[Marker](https://github.com/datalab-to/marker) (Apache-2.0) is a deep-learning
document converter that renders PDF, DOCX, PPTX, XLSX, EPUB, and HTML into
Markdown with structured tables, equations, and extracted figures. It reports
the highest benchmark accuracy among the backends KNOX can select directly
(95.67 heuristic accuracy).

Two conversion modes (configured via `mode`):

| Mode | Use case |
|------|----------|
| `fast` | CPU/MPS — lightweight detectors, block-OCRs garbled content only |
| `balanced` | CUDA GPU — full VLM layout model + full-page OCR |

Install:

```bash
uv pip install marker-pdf
```

Marker has no internal figure captioning, so `figure_enrichment` is `False`
and every extracted figure still flows through `VLMEnricher`.

## See also

- [Attachments](attachments.md) — Supplementary files per source
  (photos, binaries, source code) with opt-in ingestion.
