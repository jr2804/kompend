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

Each parser backend is an **optional dependency**. Only `liteparse` (the
default) is installed out of the box. Install additional backends as needed:

```bash
uv sync --group docling    # high-accuracy PDF/EPUB/HTML (recommended for complex layouts)
uv sync --group mineru     # layout-aware PDF/Office with OCR (heavy deps: torch, ONNX)
uv sync --group pandoc     # legacy text formats (RST, LaTeX, ODT, EPUB, TEX)
uv sync --group web        # trafilatura HTML parser for web pages
```

Or install multiple at once:

```bash
uv sync --group docling --group pandoc
```

List currently available backends with `knox list-parsers`. Uninstalled
backends configured in `parser` are silently skipped (with a warning).

## Supported Formats

| Format | Parser | Notes |
|--------|--------|-------|
| `.pdf` | DoclingParser | Structure-preserving, incl. tables, images, headings |
| `.docx` | DoclingParser | Office format via docling |
| `.pptx` | DoclingParser | PowerPoint via docling |
| `.html` | DoclingParser | Web pages via docling |
| `.md`, `.markdown` | DoclingParser | Markdown via docling |
| `.epub` | DoclingParser | E-book format via docling |
| `.tex`, `.ltx` | DoclingParser | LaTeX via docling |
| `.csv` | DoclingParser | Tabular data via docling |
| `.xlsx` | DoclingParser | Excel via docling |
| `.adoc` | DoclingParser | AsciiDoc via docling |
| `.eml`, `.msg` | DoclingParser | Email via docling |
| `.doc` | PandocParser | Legacy Word (docling does not support this) |
| `.odt` | PandocParser | OpenDocument text |
| `.rst` | PandocParser | reStructuredText |
| `.rtf` | PandocParser | Rich Text Format |
| `.txt` | PandocParser | Plain text, treated as markdown |

Unknown file extensions are checked with
[`binaryornot`](https://github.com/binaryornot/binaryornot). If the file
is determined to be plain text, it is fed to `PandocParser` as a last
resort. Truly binary files with unknown extensions are skipped.

## Primary Parser: Docling

[Docling](https://github.com/DS4SD/docling) (MIT license) is KNOX's primary
document parser. It is a pure-Python library (with C++ acceleration) that
requires no external runtimes or sidecar processes.

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
that docling does not support: legacy Word (`.doc`), OpenDocument (`.odt`),
reStructuredText (`.rst`), and Rich Text Format (`.rtf`). It also handles
plain text files (`.txt`) and serves as the binaryornot text fallback.

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
**not** installed by default — install it explicitly:

```bash
uv sync --group mineru
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

KNOX ships with **LiteParse** as the default parser (always installed).
Additional parser backends are available as optional dependency groups.
Install the ones you need:

```bash
uv sync --group docling    # high-accuracy PDF/office (Docling)
uv sync --group marker     # deep-learning PDF/office/HTML (Marker, highest benchmark accuracy)
uv sync --group pandoc     # legacy formats (.doc, .odt, .tex, .epub)
uv sync --group web        # HTML/web-page extraction (trafilatura)
```

Configure the parser order in `knox.toml` under `[ingest]`:

```toml
[ingest]
# Comma-separated list; first matching backend per file wins.
parser = "liteparse,marker,docling,pandoc"
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
uv sync --group marker
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
