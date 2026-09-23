# Document Extraction in KNOX

KNOX ("eXtended documents") aims to parse a wide variety of document formats
into a structured, LLM-aware representation. This document explains which
frameworks are used, why, and what was considered before settling on the
current architecture.

See also:

- [Architecture overview](../architecture/index.md) — 5-tier pipeline
- [Pipeline Tiers](../concepts/pipeline-tiers.md) — Per-tier component reference
- [LLM Configuration](llm-config.md) — Provider setup and model selection

## Optional Parser Backends

Installed by default:

- `liteparse` — the default PDF/office parser
- `web` — trafilatura HTML parser

Additional backends install as package extras:

```bash
uv add knox[docling]        # high-accuracy PDF/EPUB/HTML (recommended for complex layouts)
uv add knox[pandoc]         # legacy text formats (RST, LaTeX, ODT, EPUB, TEX)
```

Or several at once:

```bash
uv add knox[docling,pandoc]
```

> For `pip` users: `pip install knox[docling,pandoc]`.

A few backends cannot ship as extras because their transitive dependencies
conflict (pillow, transformers, ONNX) and cannot be locked. Install them
manually:

```bash
uv pip install marker-pdf   # deep-learning PDF/office/HTML
uv pip install glmocr       # multimodal OCR for images and PDFs
uv pip install mineru       # layout-aware PDF/Office with OCR (torch, ONNX)
```

List currently available backends with `knox list-parsers`. Uninstalled
backends configured in `parser` are silently skipped (with a warning).

> **Installing them into a knox project:** run the `uv pip install` commands
> above *outside* the project's resolved environment (a separate venv or a
> container), then point knox at that environment. They cannot be added to the
> project's dependency graph without regressing the core LLM stack — see
> below.

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
`mineru` and `glmocr` install cleanly and remain the two usable manual
backends.

## Supported Formats

> **Default coverage:** `liteparse` handles `.pdf`, `.docx`, `.pptx`, and
> `.xlsx`. The remaining formats below require `docling` or `pandoc` (extras)
> or `web` (trafilatura, default for HTML).

| Format | Parser | Notes |
|--------|--------|-------|
| `.pdf` | LiteParseParser / DoclingParser | LiteParse fast (default); docling higher accuracy, incl. tables, images, headings |
| `.docx` | LiteParseParser / DoclingParser | LiteParse fast (default); docling higher accuracy |
| `.pptx` | LiteParseParser / DoclingParser | LiteParse fast (default); docling higher accuracy |
| `.html` | WebPageParser / DoclingParser | Trafilatura HTML→Markdown (default); docling alternative |
| `.md`, `.markdown` | DoclingParser | Markdown via docling |
| `.epub` | DoclingParser | E-book format via docling |
| `.tex`, `.ltx` | DoclingParser | LaTeX via docling |
| `.csv` | DoclingParser | Tabular data via docling |
| `.xlsx` | LiteParseParser / DoclingParser | LiteParse fast (default); docling higher accuracy |
| `.adoc` | DoclingParser | AsciiDoc via docling |
| `.eml`, `.msg` | DoclingParser | Email via docling |
| `.doc` | LiteParseParser / DoclingParser | Legacy Word — converted via LibreOffice (requires a local install) |
| `.odt` | PandocParser | OpenDocument text |
| `.rst` | PandocParser | reStructuredText |
| `.rtf` | PandocParser | Rich Text Format |
| `.txt` | PandocParser | Plain text, treated as markdown |

Unknown file extensions are checked with
[`binaryornot`](https://github.com/binaryornot/binaryornot). If the file
is determined to be plain text, it is fed to `PandocParser` as a last
resort. Truly binary files with unknown extensions are skipped.

## Parser Strategy: Speed vs. Accuracy

KNOX ships **LiteParse** as the default parser. LiteParse is built on a fast
Rust core: it is much faster than Docling while being only slightly less
accurate, which makes it the right default for most ingest workloads. It covers
PDF, DOCX, DOC, PPTX, and XLSX out of the box.

For the highest possible fidelity, install **Docling** (an optional extra) and
put it ahead of `liteparse` in the `parser` priority list:

```toml
[ingest]
parser = "docling,liteparse,web"
```

## Backend Comparison: Strengths and Weaknesses

> Capabilities below reflect what each backend **actually emits into the
tier-1 element model** through knox's shared Markdown converter, not just what
the upstream project advertises. Backends that mark a capability "—" produce
that content in their Markdown, but knox previously discarded it (see
[Part-level markdown normalisation](#part-level-markdown-normalisation)).

### Capabilities at a glance

| Backend | Figures | Tables | Equations (LaTeX) | Code blocks | OCR engine | Speed |
|---------|---------|--------|-------------------|-------------|------------|-------|
| `liteparse` (default) | yes | yes¹ | **no** | — | Tesseract | fastest |
| `docling` | yes | yes | yes² | yes | RapidOCR | slow |
| `marker` | yes | yes | yes | yes | block OCR | medium (GPU-friendly) |
| `mineru` | yes | yes | yes | yes | layout + OCR | medium–slow |
| `glmocr` | yes (crops) | yes | yes | yes | VLM (cloud/self-host) | network-bound |
| `pandoc` | no | yes | yes | yes | no | fastest (no ML) |
| `web` (trafilatura) | no | yes | n/a | yes | no | fastest |

¹ GFM tables only — column detection degrades on multi-column layouts, where
contiguous text can be mis-detected as one very wide table row.
² Formula/code recognition needs `accelerate` (bundled with the `docling`
extra) and downloads the CodeFormulaV2 model on first use (~4 min on CPU).
Turn it off with `[compile.pdf] formula_enrichment = false` if you have a
reason; doing so silently drops every equation from every PDF, so it defaults
to on.

### Where each backend wins

| Backend | Strong points | Weak points |
|---------|---------------|-------------|
| `liteparse` | Fastest by a wide margin (Rust core, no Python ML stack); native PDF via PDFium; office formats via an external LibreOffice install; figures extracted with real bytes; the right default for high-volume, born-digital text. | **No equation support at all** — its Markdown never contains `$$`, so no FORMULA element can ever be produced. Table detection is layout-blind and can produce garbage rows on multi-column pages. No internal captioning (every figure needs the VLM pass). |
| `docling` | Broadest format coverage (20+, incl. email/EPUB/AsciiDoc); the only backend here that is a single locked extra with no manual install; structure-preserving headings/lists/figures; table structure + formula + code enrichment all available and on by default; RapidOCR handles scanned pages with no GPU. | Slowest of the local backends (ML models per document); table accuracy good but not best-in-class for complex borderless tables; no bounding-box/page-layout coordinates exposed; first parse downloads models. |
| `marker` | Highest reported benchmark accuracy of the local backends; equations and tables as first-class output; device-aware modes; optional LLM processors for extra quality when an LLM service is configured. | Cannot be a locked extra — its own `anthropic<1` / `openai<3` caps contradict `pydantic-ai`'s `anthropic>=1.8.0` / `openai>=3.8.0` (see [above](#why-marker-cannot-be-a-locked-extra)); model download + GPU strongly preferred; slowest on CPU; internal figure captions only via the opt-in LLM path. |
| `mineru` | Layout-aware with selectable backends (`pipeline` OCR + layout, `vlm`, `hybrid`); `formula_enable`/`table_enable` on by default; parse-method control (`auto`/`ocr`/`txt`) lets you force or skip OCR per corpus. | Cannot be a locked extra; heavy install (torch + ONNX); its Markdown carries no page markers, so `page_count` is reported as unknown (`0`); slower than docling on typical corpora. |
| `glmocr` | Strong on image-only and raster content (photos, scans, screenshots) via a multimodal VLM; emits a structured per-element JSON (labels + bounding boxes) rather than a flat Markdown blob; no local model, so CPU-only hosts work. | Needs `ZHIPU_API_KEY` (cloud) or a running self-hosted endpoint; network latency per page; cannot be a locked extra; per-element page attribution is fixed to page 1 in the crop path. |
| `pandoc` | No ML, no model downloads; excellent for legacy text formats (RST, LaTeX, ODT, EPUB, TEX) and any text file without a dedicated parser; equation and code passthrough from LaTeX sources. | No figures, no OCR, no layout analysis — it re-renders whatever the source format already encodes; the binary is downloaded on first use. |
| `web` (trafilatura) | Purpose-built HTML reader: strips chrome/nav/boilerplate and returns clean GFM (headings, lists, tables, links, code) much faster than a PDF parser on HTML. | HTML only; no figures (image refs are dropped — knox's web parser has no image resolver), no OCR, no equations. |

### Choosing

- **Default/high volume, born-digital text:** `liteparse`. Accept that equations
  and multi-column tables will be poor.
- **Equations, formulas, or technical PDFs:** `docling` (locked, no manual
  install, formula enrichment on by default) or `marker` (higher accuracy, needs
  a manual install and preferably a GPU).
- **Scanned or image-only documents:** `docling` (RapidOCR) or `mineru`
  (`parse_method = "ocr"`); `glmocr` if a VLM endpoint is acceptable.
- **Legacy text formats:** `pandoc`.
- **Web pages:** `web`.

Chain them — first matching backend per file wins:

```toml
[ingest]
parser = "docling,mineru,liteparse,pandoc,web"
```

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

## Docling

[Docling](https://github.com/docling-project/docling) (MIT license) is KNOX's
highest-accuracy document parser. It is a pure-Python library (with C++
acceleration) that requires no external runtimes or sidecar processes. Install
it with `uv add knox[docling]`.

## Strengths

- Supports 20+ input formats (PDF, DOCX, PPTX, HTML, MD, EPUB, LaTeX,
  CSV, XLSX, AsciiDoc, email, and more).
- Structure-preserving output with heading hierarchy, tables, lists,
  formulas, and image references.
- Handles scanned PDFs via built-in OCR (RapidOCR).
- No GPU required.
- Active development, MIT license.

## Limitations

- Bounding-box / page-layout coordinates are not exposed (docling targets
  content structure, not visual layout).
- Does not produce Tagged PDF or PDF/UA output.
- Table extraction accuracy is good but not best-in-class for complex
  borderless tables.

## PandocParser: Legacy Format Support

[Pandoc](https://pandoc.org/) (the Python `pandoc` package) handles formats
that docling does not support: OpenDocument (`.odt`), reStructuredText
(`.rst`), LaTeX (`.tex`, `.ltx`), and Rich Text Format (`.rtf`). It also
handles plain text files (`.txt`) and serves as the binaryornot text
fallback. Legacy binary Word (`.doc`) is **not** pandoc's job — pandoc has
no reader for the OLE container format; LiteParse and Docling read `.doc`
via LibreOffice instead.

PandocParser is registered **after** DoclingParser in the parser router, so
docling always wins for formats both support (e.g. `.docx` is handled by
docling, not pandoc).

## LiteParseParser: Fast PDF/Office Parsing

[LiteParse](https://github.com/run-llama/liteparse) (Rust-core, Apache 2.0) is
the recommended backend for PDF and Office formats when speed matters more than
layout-structure accuracy. It reconstructs documents to Markdown with embedded
image bytes but has no LLM captioning of its own, so every extracted figure
flows through `VLMEnricher`.

Configure via the priority list: `parser = "web,liteparse,docling"`.

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

Then enable it in the priority list: `parser = "web,mineru,docling"`.

## Investigated and Rejected: opendataloader-pdf[hybrid]

### Why it looked promising

[opendataloader-pdf](https://github.com/opendataloader-project/opendataloader-pdf)
(Apache 2.0) ranks #1 in public benchmarks with an overall score of 0.907
versus docling's 0.882. Its "hybrid" mode routes complex pages (tables,
scanned documents, formulas, charts) to an AI backend for significantly
better accuracy than the purely local mode.

### Why it was rejected

Despite the accuracy advantage, operational concerns outweigh the benefit:

## 1. Java 11+ runtime requirement

The package is 85.9% Java. Every KNOX user would need a JDK installed and
`JAVA_HOME` configured — a serious friction point for a project that is
otherwise pure Python.

## 2. Separate hybrid-mode server

Hybrid mode requires a background process:

```bash
opendataloader-pdf-hybrid --port 5002
```

KNOX would need to spawn, monitor, and shut down this server — error-prone
on Windows, and a crash would cause silent parse failures.

## 3. JVM-per-call overhead

Each `convert()` invocation spawns a new JVM process. When ingesting
dozens of documents, this adds seconds of overhead per file.

## 4. PDF-only

opendataloader handles only PDF. KNOX needs multi-format support
(DOCX, PPTX, HTML, MD, EPUB, etc.) — all handled natively by docling.

## 5. Different data model

opendataloader's JSON output is page-oriented with bounding boxes.
Docling's model is content-oriented (document tree). Supporting both
would require a second adapter layer, increasing maintenance surface.

## 6. Marginal real-world gain

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

## GlmOcrParser: Multimodal OCR for Images and PDFs

[GLM-OCR](https://github.com/zai-org/GLM-OCR) (Zhipu AI, 0.9B params) is a
lightweight multimodal OCR model. It accepts raster images (PNG, JPG, TIFF,
BMP, GIF, WEBP) and PDFs (rendered to images internally), returning
structured elements labelled `text`, `title`, `table`, `formula`, and
`figure` — which map directly onto KNOX `DocumentElement` types.

Two execution modes:

- **Cloud** (default): calls Zhipu MaaS. Requires `ZHIPU_API_KEY` env var.
  No local GPU needed.
- **Self-hosted**: set `mode="selfhosted"` and point `api_url` at a vLLM /
  SGLang / Ollama endpoint.

The `glmocr` package is an optional dependency:

```bash
uv pip install glmocr
```

GLM-OCR does no internal captioning (`figure_enrichment=False`); every
extracted figure still flows through `VLMEnricher`.

### Strengths

- OmniDocBench V1.5: 94.62 — top accuracy at launch
- Apache 2.0 (code) + MIT (weights) — very permissive
- Cloud API mode needs no GPU at all
- Rich structured output (label + bbox per element)

### Limitations

- Image-based only — no native DOCX/PPTX/XLSX support
- PDF handling requires rasterising to images first
- Newer project (Feb 2026, limited release history)

## Binaryornot Text Fallback

When a file's extension is not recognised by any registered parser,
[`binaryornot`](https://github.com/binaryornot/binaryornot) checks whether
the file contains binary data. If it is text, the file is routed to
`PandocParser` as a last-resort parser. This handles edge cases like
extensionless files or files with unusual extensions that happen to be
plain text.

## Optional Parser Frontends

KNOX ships with **LiteParse** and **web** (trafilatura) as the default
parsers (always installed). Additional parser backends install as extras or
manually:

```bash
uv add knox[docling]        # high-accuracy PDF/office (Docling)
uv add knox[pandoc]         # legacy formats (.odt, .tex, .rtf, .rst)
uv pip install marker-pdf   # deep-learning PDF/office/HTML (Marker, highest benchmark accuracy)
```

Configure the parser order in `knox.toml` under `[ingest]`:

```toml
[ingest]
# Comma-separated list; first matching backend per file wins.
parser = "liteparse,docling,pandoc"
```

Backends that are known but not installed are silently skipped (with a
warning); unknown names raise `ValueError`.

## Marker Parser

[Marker](https://github.com/datalab-to/marker) (Apache-2.0) is a deep-learning
document converter that renders PDF, DOCX, PPTX, XLSX, EPUB, and HTML into
Markdown with structured tables, equations, and extracted figures. It reports
the highest benchmark accuracy among KNOX-supported parsers (95.67 heuristic
accuracy vs docling 86.7).

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

## Previously Removed Parsers

- **PdfPlumberParser** (pdfplumber) — docling covers `.pdf` first in the
  router, making this fallback unreachable in practice. PdfPlumber was
  not used for any other purpose (page counting, title extraction, etc.).
- **VLMParser** — standalone image parser for `.png`/`.jpg`/etc. Never
  wired into any router. Image content embedded in PDFs is handled by
  docling's own image extraction; standalone images lack the VLM client
  needed for captioning at Tier 1.
- **ExternalParser / HTTPParser** — framework for delegating parsing to
  CLI tools or HTTP endpoints. Never wired into any config path and
  lacked documentation.

## See also

- [Attachments](attachments.md) — Supplementary files per source
  (photos, binaries, source code) with opt-in ingestion.
