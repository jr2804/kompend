# Open Knowledge Format (OKF) v0.2

KNOX produces **OKF v0.2**-compliant knowledge bundles by default. OKF is an
open, human- and agent-friendly format for representing knowledge — a directory
of markdown files with YAML frontmatter.

**Spec:** <https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md>

## Why OKF?

OKF is designed for **portability and interoperability**:

- **Human-readable** — plain markdown files, no proprietary binary format
- **Agent-friendly** — structured YAML frontmatter with typed fields, plus trust
  signals (``generated``/``verified``) that record how content was produced
- **Portable** — a directory of files, easy to version, diff, and share
- **Extensible** — producers may add any additional frontmatter keys
- **Permissive consumption** — consumers MUST tolerate missing optional fields,
  unknown types, unknown keys, broken links, and missing index files

## Conformance

KNOX meets all 3 hard requirements for OKF v0.2 conformance (§11):

1. **Every concept file** has parseable YAML frontmatter with a non-empty
   ``type`` field
2. **Reserved filenames** (``index.md``, ``log.md``) follow OKF structure
3. **Cross-links** use standard markdown absolute bundle-relative paths

## Migrating from v0.1

v0.2 is a superset of v0.1 with two **breaking field migrations** (§13.1):

| v0.1 | v0.2 | Notes |
|------|------|-------|
| ``timestamp`` | ``generated: {by, at}`` | ``by`` follows the §7 actor convention (``knox/<version>``) |
| Body ``# Citations`` list | Structured ``sources`` objects + per-claim footnotes | Attribution via ``[^id]`` keyed to ``sources[].id`` (§5.1) |

v0.2 consumers fall back to v0.1 fields gracefully (§3.1), so v0.1 bundles
remain *consumable*, but only the v0.2 fields above are *canonical*.

## Bundle Structure

A KNOX wiki compiled with the default markdown output produces an OKF v0.2 bundle:

```text
wiki/
├── index.md              ← table of contents with grouped sections
├── log.md                ← change log with ISO date headings
├── references.md         ← formatted bibliography (type: Reference)
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
| ``description`` | — | One-line summary (first line of content) |
| ``tags`` | — | Cross-cutting **topic** tags (from compilation) |
| ``status`` | — | Lifecycle: ``stable`` (default), ``draft``, ``deprecated`` (§5.4) |
| ``generated`` | ✅ | ``{by, at}`` — producer actor + ISO timestamp (§5.2) |
| ``verified`` | — | List of ``{by, at}`` trust events; emitted when a verification pass runs (§5.2) |
| ``sources`` | — | Structured provenance objects ``{id, resource, title, author}`` (§5.1) |
| ``kind`` | — | KNOX extension: original KNOX node type |
| ``source_files`` | — | KNOX extension: provenance filename stems |

!!! note "``tags`` vs ``source_files`` vs ``sources``"
    These are deliberately distinct to avoid the v0.1 overload:
    **``tags``** = cross-cutting topics; **``source_files``** = KNOX extension
    listing provenance filename stems; **``sources``** = OKF v0.2 structured
    provenance objects from the reference store. Source filenames never appear
    in ``tags`` or as bare strings in ``sources``.

## Cross-Links

OKF recommends **absolute** (bundle-relative) links starting with ``/`` (§6.1):

```markdown
See the [audio processing](/concepts/audio-processing.md) page for details.
```

These are stable when documents move within their subdirectory. KNOX converts
``[[wikilinks]]`` to this format during compilation.

## Per-Claim Attribution

Instead of a v0.1 body ``# Citations`` list, OKF v0.2 attributes claims with
markdown footnotes keyed to the structured ``sources`` entries (§5.1):

```markdown
---
sources:
  - id: smith2026
    resource: https://doi.org/10.1234/audio
    title: "Advanced Audio Processing"
---

This technique improves SNR by 12 dB [^smith2026].

[^smith2026]: Advanced Audio Processing
```

## Trust Signals

The ``generated`` and ``verified`` fields record how content was produced and
checked, enabling trust tiers (§5.3):

- **``generated.by``** — always ``knox/<version>`` (the compiling agent)
- **``verified``** — appended with a ``{by, at}`` event each time
  ``knox compile --verify`` runs; absent when no verification pass has occurred
  (honest "unverified" tier)

## Index Page

The root ``index.md`` has grouped sections with descriptions. It is the only
file permitted to declare ``okf_version`` in frontmatter (§12 exception):

```markdown
---
okf_version: "0.2"
---

# Knowledge Base

# Concept

- [Audio Processing](/concepts/audio-processing.md) — Signal processing techniques
- [Python Fundamentals](/concepts/python-fundamentals.md) — Core language concepts

# Overview

- [System Architecture](/concepts/system-architecture.md) — High-level design
```

## Change Log

The ``log.md`` uses ISO 8601 date headings with bold action prefix (§9):

```markdown
## 2026-07-10

**Creation**: Compiled 42 page(s) from 15 source(s).
```

## Legacy Formats

Select a different output format with ``compile.output.format``:

| Format | Config value | Description |
|--------|-------------|-------------|
| OKF (default) | ``okf`` | Open Knowledge Format v0.2 |
| Markdown | ``markdown`` | Obsidian vault mode (``[[wikilinks]]``) |
| JSON | ``json`` | Single ``wiki.json`` file |
| Typst | ``typst`` | Single ``wiki.typ`` file |
