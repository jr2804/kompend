# Output Configuration Guide

KNOX supports multiple output formats for the compiled wiki, controlled by the
`compile.output.format` config key (default: ``okf``).

## Open Knowledge Format (OKF) v0.2 (default)

KNOX produces **OKF v0.2**-compliant knowledge bundles by default. OKF is an open,
human- and agent-friendly format for representing knowledge — a directory of
markdown files with YAML frontmatter. See the [OKF concept page](../concepts/open-knowledge-format.md)
for the full field reference, trust signals, and the v0.1→v0.2 migration table.

**Spec:** <https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md>

**Conformance (§11):**

- Frontmatter is built once by `VaultWriter.okf_frontmatter()`; every concept
  file has parseable YAML with a non-empty ``type`` field
- Trust fields: ``generated: {by, at}`` (supersedes v0.1 ``timestamp``),
  ``status`` (default ``stable``), optional ``verified``
- ``tags`` = topic tags only; ``source_files`` = provenance filename stems
  (KNOX extension); ``sources`` = §5.1 structured objects — never bare strings
- Cross-links use absolute bundle-relative paths: ``[Title](/concepts/title.md)``
- ``index.md`` has grouped sections with descriptions; root declares ``okf_version: "0.2"``
- ``log.md`` uses ISO date headings with bold action prefix
- Per-claim attribution via footnotes ``[^id]`` keyed to ``sources[].id``
  (replaces the v0.1 body ``# Citations`` numbered list)

## Legacy formats

### Markdown (Obsidian vault mode)

Select with ``compile.output.format = "markdown"`` or ``--set compile.output.format=markdown``.
Produces ``.md`` files with ``[[wikilinks]]``, ``index.md`` TOC, and ``concepts/`` subfolder.

### JSON

Select with ``compile.output.format = "json"``. Single ``wiki.json`` file.

### Typst

Select with ``compile.output.format = "typst"``. Single ``wiki.typ`` file.

## Flavor configuration

KNOX supports multiple Markdown flavors for the legacy Markdown output via the
``compile.output.flavor`` config key (set with ``knox config set compile.output.flavor <flavor>``
or override one run with ``--set compile.output.flavor=<flavor>``).

| Key / flag | Description | Default |
|------|-------------|---------|
| ``compile.output.flavor`` (``--set compile.output.flavor=``) | Markdown output flavor | ``obsidian`` (auto-detected) |
| ``compile.output.frontmatter`` | Frontmatter fields | — |
| ``--synthesize/--no-synthesize`` | Generate LLM overview page | ``True`` (default-on) |

## Examples

```bash
# OKF v0.2 bundle (default)
knox compile

# Legacy Obsidian vault mode
knox config set compile.output.format markdown

# GitHub-flavored Markdown (legacy mode)
knox config set compile.output.format markdown
knox config set compile.output.flavor gfm

# MkDocs site (with frontmatter field)
knox config set compile.output.format markdown
knox config set compile.output.flavor mkdocs
knox config set compile.output.frontmatter '{language="en"}'
```

## Config file (`.knox/config.toml`)

The `output` section configures flavor, frontmatter defaults, and rumdl behavior:

```toml
[compile.output]
flavor = "obsidian"        # auto-detect if omitted
frontmatter_defaults = { language = "en", domain = "general" }
rumdl_config = ".rumdl.toml"   # custom rumdl config (optional)
```

### Flavor auto-detection

Without an explicit `compile.output.flavor`, KNOX inspects the output directory:

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

rumdl handles all flavor-specific transformations. It is installed alongside KNOX
when using the `knox[rumdl]` optional dependency:

```bash
uv tool install knox[rumdl]
# or
uv add knox[rumdl]
```

If rumdl is not installed, KNOX falls back to raw canonical Markdown with a warning.

### Custom rumdl configuration

Create `.rumdl.toml` in your project root:

```toml
line-length = 100
rules.MD013.line-length = 100
rules.MD033.allowed_elements = ["sup", "sub"]
```

Then reference it in your KNOX config:

```yaml
output:
  rumdl_config: .rumdl.toml
```

> **Note:** If a flavor you need is not supported, contribute upstream to
> [rvben/rumdl](https://github.com/rvben/rumdl). KNOX never ships custom
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
knox config set compile.output.frontmatter '{language="en", domain="ml"}'
# ...or for one run only:
knox compile . --set compile.output.frontmatter='{language="en", domain="ml"}'
```

### In `knox write`

```bash
knox write "Transformer Architecture" --frontmatter language=en
```

Config defaults are merged first, then CLI flags override them.
