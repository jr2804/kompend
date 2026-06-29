# Output Architecture

KNOX's output layer converts a compiled wiki into publishable documents.
It is designed around three principles:

1. **Flavor-agnostic intermediate model** — the `Document` dataclass carries content in canonical form
2. **Single rendering path** — `DocumentRenderer` is the only way to produce Markdown
3. **rumdl for flavor support** — all Markdown flavor transformations go through `rumdl fmt`, never hand-rolled

```text
              ┌──────────────────┐
              │   CompiledWiki   │
              │  (core/models)   │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  MarkdownWriter  │
              │   (thin wrapper) │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │     Document     │
              │  (output/models) │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │ DocumentRenderer │
              │  (output/render) │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │ rumdl fmt        │
              │ --flavor <X>     │
              └────────┬─────────┘
                       │
                       ▼
                ┌──────────────┐
                │ wiki.md      │
                │ index.md     │
                │ log.md       │
                └──────────────┘
```

## Document model

Defined in `src/knox/output/models.py`:

| Field | Type | Description |
|-------|------|-------------|
| `title` | `str` | Page title (used as the heading) |
| `body` | `str` | Markdown body with `[[wikilinks]]` |
| `frontmatter` | `dict[str, Any]` | YAML frontmatter fields |
| `heading_level` | `int` | Heading level for the title (1 = `#`) |
| `source_chunks` | `list[str] | None` | Source chunk IDs for provenance |
| `toc_entries` | `list[dict] | None` | TOC entries contributed by this document |
| `page_slug` | `str` | URL-friendly slug |

## DocumentRenderer

Defined in `src/knox/output/renderer.py`.

The renderer produces canonical Markdown (frontmatter `---` block + heading + body)
and then optionally pipes through `rumdl fmt --flavor <FLAVOR>` for flavor-specific
formatting.

```python
renderer = DocumentRenderer(flavor="obsidian")
doc = Document(title="My Page", body="Content", frontmatter={"lang": "en"})
result = renderer.render(doc)
# => ---
#    lang: en
#    ---
#
#    # My Page
#
#    Content
```

### Rumdl integration

When rumdl is installed, `render()` pipes the raw Markdown through:

```text
rumdl fmt --flavor <FLAVOR> --stdin --stdin-filename page.md --silent
```

If rumdl is not on `PATH`, or if the subprocess fails, the raw Markdown is returned
with a warning logged.

## BaseOutputWriter

Defined in `src/knox/output/base.py`.

All output writers extend `BaseOutputWriter` and must implement `write(wiki, output_dir)`.
The class provides a plugin registry:

```python
writer_class = BaseOutputWriter.get("markdown")  # MarkdownWriter
```

### Current implementations

| Name | Class | Description |
|------|-------|-------------|
| `markdown` | `MarkdownWriter` | Multi-flavor Markdown via DocumentRenderer |
| `json` | `JsonWriter` | JSON array of `[page, content, source]` |
| `typst` | `TypstWriter` | Typst document format (experimental) |

## Flavor auto-detection

When `compile.output.flavor` is not explicitly set, KNOX inspects the output directory:

| File/dir present | Flavor detected |
|-----------------|----------------|
| `.obsidian/` | `obsidian` |
| `mkdocs.yml` | `mkdocs` |
| `_quarto.yml` or `*.qmd` files | `quarto` |
| none | `obsidian` (default) |

Explicit `compile.output.flavor` always overrides auto-detection.
