---
icon: lucide/book-open
---

# KNOX

**Knowledge Organizer for eXtended documents**

A compiled LLM-wiki system for scientific and technical documents.
Parse, chunk, resolve, and compile PDFs, DOCX, spreadsheets, and
web sources into an interlinked, queryable knowledge base.

---

## 5-Tier Pipeline

KNOX follows a parser-agnostic, structure-preserving pipeline:

``` mermaid
graph LR
  T1[Preprocessing] --> T2[Chunking]
  T2 --> T3[Resolution]
  T3 --> T4[Compilation]
  T4 --> T5[Output]
```

### Quick Start

```bash
# Compile documents into a wiki
knox compile *.pdf --sources-dir ./sources

# Query the compiled wiki
knox query "what is the main result?"
```

### Key Features

- **Parser-agnostic** — Docling (PDF), Pandoc (DOCX/EPUB/HTML), crawl4ai (web)
- **Hybrid chunking** — Atomic elements + recursive token-aware splitting
- **WiCER compilation** — Iterative compile-evaluate-refine loop for quality
- **Multi-format output** — Markdown (incl. Obsidian vault), JSON, Typst
- **LLM-powered linting** — 8 structural rules + 3 LLM rules

---

[Getting Started](guides/cli.md){ .md-button }
[Architecture](architecture/index.md){ .md-button }
[GitHub](https://github.com/anomalyco/knox){ .md-button }
