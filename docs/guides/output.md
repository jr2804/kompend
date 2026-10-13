# Output Configuration Guide

kompend supports multiple output formats for the compiled wiki, controlled by the
`compile.output.format` config key (default: ``markdown``).

## Markdown (OKF v0.2-compliant, default)

The default markdown output is a **OKF v0.2**-compliant knowledge bundle. OKF
is an open, human- and agent-friendly **format spec** — a directory of
markdown files with YAML frontmatter. It is **not** a separate output format:
`compile.output.format` offers only `markdown` (OKF-compliant), `json`, and
`typst`. See the [OKF concept page](../concepts/open-knowledge-format.md)
for the full field reference, trust signals, and the v0.1→v0.2 migration table.

**Spec:** <https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md>

**Conformance (§11):**

- Frontmatter is built once by `VaultWriter.okf_frontmatter()`; every concept
  file has parseable YAML with a non-empty ``type`` field
- Trust fields: ``generated: {by, at}`` (supersedes v0.1 ``timestamp``),
  ``status`` (default ``stable``), optional ``verified``
- ``tags`` = topic tags only; ``source_files`` = provenance filename stems
  (kompend extension); ``sources`` = §5.1 structured objects — never bare strings
- Cross-links are portable folder-relative markdown paths
  (``[Title](concepts/title.md)`` from the bundle root) — they render in
  Obsidian, GitHub, and any Markdown viewer
- ``index.md`` has grouped sections with descriptions; root declares ``okf_version: "0.2"``
- Pages carry a ``kind`` from the mandatory taxonomy — ``Concept``, ``Entity``,
  ``Summary``, ``Comparison`` — plus any ``compile.extraction.categories``
  extras; unknown kinds fall back to ``Concept`` (permissive consumption)
- ``log.md`` follows OKF §9: date-grouped ``## YYYY-MM-DD`` headings, newest
  first; entries are ``- [HH:mm:ss] **Action**: prose`` list items. Never
  wiped — compiles, ingests, queries, and lint runs all append entries to the
  same file (one shared timeline)
- Per-claim attribution via footnotes ``[^id]`` keyed to ``sources[].id``:
  inline ``[sources/x.pdf]`` citations are rewritten to ``[^id]`` refs, with
  definitions emitted only for referenced ids (replaces the v0.1 body
  ``# Citations`` numbered list)

## Other formats

### JSON

Select with ``compile.output.format = "json"``. Single ``wiki.json`` file.

### Typst

Select with ``compile.output.format = "typst"``. Single ``wiki.typ`` file.

## Formatting (rumdl)

Compile runs `rumdl fmt` over the generated output, and `kompend lint format`
re-runs it on demand (`--check` for CI). The per-vault rumdl config lives at
`.kompend/rumdl.toml` (auto-materialized from the shipped default) and can be
overridden via `compile.output.rumdl_config`.

## Flavor configuration

kompend supports multiple Markdown flavors for the legacy Markdown output via the
``compile.output.flavor`` config key (set with ``kompend config set compile.output.flavor <flavor>``
or override one run with ``--set compile.output.flavor=<flavor>``).

| Key / flag | Description | Default |
|------|-------------|---------|
| ``compile.output.flavor`` (``--set compile.output.flavor=``) | Markdown output flavor | ``obsidian`` (auto-detected) |
| ``compile.output.frontmatter`` | Frontmatter fields | — |
| ``--synthesize/--no-synthesize`` | Generate LLM overview page | ``True`` (default-on) |

## Examples

```bash
# OKF v0.2 bundle (default)
kompend compile

# Legacy Obsidian vault mode
kompend config set compile.output.format markdown

# GitHub-flavored Markdown (legacy mode)
kompend config set compile.output.format markdown
kompend config set compile.output.flavor gfm

# MkDocs site (with frontmatter field)
kompend config set compile.output.format markdown
kompend config set compile.output.flavor mkdocs
kompend config set compile.output.frontmatter '{language="en"}'
```

## Config file (`.kompend/config.toml`)

The `output` section configures flavor, frontmatter defaults, and rumdl behavior:

```toml
[compile.output]
flavor = "obsidian"        # auto-detect if omitted
frontmatter_defaults = { language = "en", domain = "general" }
rumdl_config = ".rumdl.toml"   # custom rumdl config (optional)
```

### Flavor auto-detection

Without an explicit `compile.output.flavor`, kompend inspects the output directory:

| Marker | Flavor | Use case |
|--------|--------|----------|
| `.obsidian/` directory | `obsidian` | Obsidian vault |
| `mkdocs.yml` file | `mkdocs` | MkDocs site |
| `_quarto.yml` or `*.qmd` files | `quarto` | Quarto project |
| none | `obsidian` | Default fallback |

Explicit `compile.output.flavor` always overrides auto-detection.

## Available flavors

| Flavor | Alias | Description |
|--------|-------|-------------|
| `obsidian` | — | Obsidian.md vaults with `[[wikilinks]]` |
| `gfm` | `github` | GitHub-flavored Markdown |
| `mkdocs` | — | MkDocs material |
| `mdx` | — | MDX (JSX in Markdown) |
| `quarto` | — | Quarto documents |
| `myst` | — | MyST Markdown |
| `standard` | — | CommonMark standard |
| `pandoc` | — | Pandoc Markdown |
| `kramdown` | — | kramdown |
| `azure_devops` | `azure`, `ado` | Azure DevOps wiki |

## Rumdl setup

rumdl handles all flavor-specific transformations. It is installed alongside kompend
when using the `kompend[rumdl]` optional dependency:

```bash
uv tool install kompend[rumdl]
# or
uv add kompend[rumdl]
```

If rumdl is not installed, kompend falls back to raw canonical Markdown with a warning.

### Custom rumdl configuration

The vault's rumdl config lives at `.kompend/rumdl.toml` (auto-materialized from
the shipped default on first use — edit that copy, it is never overwritten).
Point `compile.output.rumdl_config` elsewhere to use a custom file:

```toml
[global]
disable = ["MD013"]  # line length is style, not correctness

[MD025]
front-matter-title = ""  # frontmatter `title:` must not count as a second H1

[MD033]
allowed-elements = ["sup", "sub"]
```

> **Note:** If a flavor you need is not supported, contribute upstream to
> [rvben/rumdl](https://github.com/rvben/rumdl). kompend never ships custom
> Markdown converters.

## Frontmatter

Frontmatter is rendered as YAML between `---` delimiters at the top of each page:

```markdown
---
language: en
domain: ml
---

# My Page

Content...
```

### From config file

```yaml
output:
  frontmatter:
    defaults:
      language: en
      domain: general
```

### From config

```bash
kompend config set compile.output.frontmatter '{language="en", domain="ml"}'
# ...or for one run only:
kompend compile . --set compile.output.frontmatter='{language="en", domain="ml"}'
```

### In `kompend write`

```bash
kompend write "Transformer Architecture" --frontmatter language=en
```

Config defaults are merged first, then CLI flags override them.
