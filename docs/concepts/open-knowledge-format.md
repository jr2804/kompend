# Open Knowledge Format (OKF) v0.1

KNOX produces **OKF v0.1**-compliant knowledge bundles by default. OKF is an
open, human- and agent-friendly format for representing knowledge — a directory
of markdown files with YAML frontmatter.

**Spec:** <https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md>

## Why OKF?

OKF is designed for **portability and interoperability**:

- **Human-readable** — plain markdown files, no proprietary binary format
- **Agent-friendly** — structured YAML frontmatter with typed fields
- **Portable** — a directory of files, easy to version, diff, and share
- **Extensible** — producers may add any additional frontmatter keys
- **Permissive consumption** — consumers MUST tolerate missing optional fields,
  unknown types, unknown keys, broken links, and missing index files

## Conformance

KNOX meets all 3 hard requirements for OKF v0.1 conformance (§9):

1. **Every concept file** has parseable YAML frontmatter with a non-empty
   ``type`` field
2. **Reserved filenames** (``index.md``, ``log.md``) follow OKF structure
3. **Cross-links** use standard markdown absolute bundle-relative paths

## Bundle Structure

A KNOX wiki compiled with ``compile.output.format = "okf"`` produces:

```text
wiki/
├── index.md              ← table of contents with grouped sections
├── log.md                ← change log with ISO date headings
├── concepts/             ← concept/entity/comparison pages
│   ├── audio-processing.md
│   ├── python-fundamentals.md
│   └── ...
└── write/                ← one-shot pages (knox write)
    └── my-notes.md
```

## Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| ``type`` | ✅ | OKF concept type (mapped from KNOX ``node_type``) |
| ``title`` | ✅ | Human-readable page title |
| ``description`` | ✅ | One-line summary (first line of content) |
| ``tags`` | ✅ | Source filenames as cross-cutting tags |
| ``timestamp`` | ✅ | ISO 8601 compile timestamp |
| ``kind`` | — | KNOX extension: original node type |
| ``compiled_at`` | — | KNOX extension: compile date |
| ``sources`` | — | KNOX extension: source filenames |

## Cross-Links

OKF recommends **absolute** (bundle-relative) links starting with ``/`` (§5.1):

```markdown
See the [audio processing](/concepts/audio-processing.md) page for details.
```

These are stable when documents move within their subdirectory. KNOX converts
``[[wikilinks]]`` to this format during compilation.

## Index Page

The root ``index.md`` has grouped sections with descriptions:

```markdown
---
okf_version: "0.1"
---

# Knowledge Base

## Concepts

- [Audio Processing](/concepts/audio-processing.md) — Signal processing techniques
- [Python Fundamentals](/concepts/python-fundamentals.md) — Core language concepts

## Overviews

- [System Architecture](/concepts/system-architecture.md) — High-level design
```

## Change Log

The ``log.md`` uses ISO 8601 date headings with bold action prefix (§7):

```markdown
## 2026-07-10

- **Creation**: Initial compilation (42 pages, 15 sources)
- **Update**: Added audio processing concepts
```

## Citations

Provenance information is rendered as a ``# Citations`` section with numbered
``[N]`` refs (§8):

```markdown
# Citations

[1] smith2026_audio.pdf — "Advanced Audio Processing" (p. 12-15)
[2] ISO_standard.pdf — "ISO/IEC 23000-5" (p. 3)
```

## Legacy Formats

Select a different output format with ``compile.output.format``:

| Format | Config value | Description |
|--------|-------------|-------------|
| OKF (default) | ``okf`` | Open Knowledge Format v0.1 |
| Markdown | ``markdown`` | Obsidian vault mode (``[[wikilinks]]``) |
| JSON | ``json`` | Single ``wiki.json`` file |
| Typst | ``typst`` | Single ``wiki.typ`` file |
