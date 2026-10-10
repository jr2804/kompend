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

`[ingest] parser` names exactly one backend: `parsecraft`, the default.

- `parsecraft` — an external document-intelligence pipeline that routes each
  page across its own sub-backends (native text/markdown/HTML/PDF, LiteParse,
  docling, pandoc, OCR extras) and returns one document per source. It ships
  as the `parsecraft` extra, which pulls the sub-backends that make it
  useful: `uv add knox[parsecraft]`.

```toml
[ingest]
parser = "parsecraft"   # the only knox backend
```

`knox list-parsers` shows the backend and the formats it claims. An unknown
name raises, so a vault configured against a backend that does not exist
fails loudly instead of parsing nothing.

Two opt-in extensions ride on top of parsecraft rather than being knox
backends:

- **`knox[mineru]`** — adds MinerU 4.x as a parsecraft sub-backend
  (`parsecraft[mineru]>=2026.10.2`), eligible for layout-heavy and OCR
  pages. The extra opts into MinerU's own licence terms — read the note in
  `pyproject.toml` before using it commercially.
- **marker (bring-your-own)** — `marker-pdf` cannot be locked or pinned by
  knox at all (see [below](#why-marker-cannot-be-a-locked-extra)); install
  it yourself and parsecraft counts it among its routing candidates.

The judge tier needs the `systemone` extra on top of parsecraft:

```bash
uv add knox[systemone]
```

### Why `marker` cannot be a locked extra

Measured fresh (knox-3, 2026-10-10) against the current lock: the tight pin
`marker-pdf>=1.10.2,<2` **hard-fails** — `marker-pdf` 1.10.2 pins
`anthropic>=0.46,<0.47` while `pydantic-ai` 2.31.1 needs `anthropic>=0.108`,
and the two ranges do not overlap. Dropping the upper bound only makes it
"resolve" by downgrading `anthropic` 1.13.0 → 0.125.0 and `pillow`
12.3.0 → 10.4.0 — silently regressing knox's LLM stack.

That is the substantive reason the marker backend was deleted rather than
kept behind an extra (ADR-0006): knox cannot own the `marker-pdf` pin, so
marker is parsecraft's **bring-your-own** sub-backend.

## Supported Formats

Discovery follows the configured backend: knox asks `[ingest] parser` which
suffixes it claims (`get_all_extensions([config.parser])`) and scans the source
directory for exactly those. With the default backend that is parsecraft's
32-suffix table. `.knox/` is excluded regardless of claimed formats, so state
artifacts such as `.knox/rumdl.toml` are never ingested even though parsecraft
claims `.toml`.

| Format | Handled by | Notes |
|--------|------------|-------|
| `.pdf` | `parsecraft` | Routed per page across its sub-backends; scanned pages escalate to an OCR backend when one is installed |
| `.docx`, `.pptx`, `.xlsx` | `parsecraft` | Office formats via its sub-backends |
| `.epub`, `.odt`, `.rtf` | `parsecraft` | E-book and legacy text containers |
| `.md`, `.markdown`, `.html`, `.htm` | `parsecraft` | Native text and markup backends |
| `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` | `parsecraft` | Images; content comes from an OCR/VLM sub-backend when installed |
| `.txt`, `.text`, `.log`, `.rst`, `.ini`, `.cfg`, `.toml`, `.c`, `.h`, `.hpp`, `.cpp`, `.cc`, `.py`, `.m`, `.sh`, `.tex` | `parsecraft` | Plain-text source family |

`.html`/`.htm` are handled by parsecraft's own native/trafilatura backends —
there is no separate web parser. `knox list-parsers` prints the live
extension list. A file whose suffix the backend does not claim is never
picked up by discovery.

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

## Sub-Backends: what parsecraft routes between

There is one selectable backend; the comparison that matters is between
parsecraft's sub-backends, and the extras decide which are present:

| Sub-backend | What it wins | Arrives via |
|------------|--------------|-------------|
| native (text/markdown/HTML) | The fast path for born-digital content | core parsecraft |
| LiteParse | Default PDF engine | `knox[parsecraft]` |
| docling | Structure-heavy pages: equations, tables, complex layouts | `knox[parsecraft]` |
| pandoc | Legacy containers (EPub, ODT, RTF) | `knox[parsecraft]` |
| OCR backends | Scanned pages (per-page escalation) | `parsecraft[ocr-*]` extras |
| mineru | Layout-aware PDF/Office with OCR | `knox[mineru]` |
| marker | Highest reported benchmark accuracy | bring-your-own install |

### Choosing

- **Default:** `knox[parsecraft]` covers text, office, PDF and container
  formats with per-page routing.
- **Layout-heavy or scanned corpora:** add `knox[mineru]` or a
  `parsecraft[ocr-*]` extra.
- **Maximum accuracy on hard layouts:** install `marker-pdf` yourself
  (GPU preferred) — knox cannot pin it (see
  [Why `marker` cannot be a locked extra](#why-marker-cannot-be-a-locked-extra)).

## Part-level markdown normalisation

Every backend that produces Markdown after its SDK call goes through one shared
converter (`knox/preprocessing/parsers/markdown.py`) that maps the Markdown into
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
that is its 32-suffix table — `.txt`, `.text`, `.log`, `.rst`, `.ini`, `.cfg`,
`.toml`, `.c`, `.h`, `.hpp`, `.cpp`, `.cc`, `.py`, `.m`, `.sh`, `.tex`, plus the
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

## See also

- [Attachments](attachments.md) — Supplementary files per source
  (photos, binaries, source code) with opt-in ingestion.
