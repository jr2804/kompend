# Output Configuration Guide

KNOX supports multiple Markdown flavors for compiled output via `--flavor`
and the `output` section in your config file.

## CLI flags

| Flag | Description | Default |
|------|-------------|---------|
| `--flavor` | Markdown output flavor | `obsidian` (auto-detected) |
| `--frontmatter` | Frontmatter fields (`key=value`) | — |
| `--synthesize` | Generate LLM overview page | `False` |

### Examples

```bash
# Obsidian-compatible wiki (default)
knox compile . --flavor obsidian

# GitHub-flavored Markdown
knox compile . --flavor gfm

# MkDocs site
knox compile . --flavor mkdocs --frontmatter language=en
```

## Config file (`knox config.yaml`)

The `output` section configures flavor, frontmatter defaults, and rumdl behavior:

```yaml
output:
  flavor: obsidian         # auto-detect if omitted
  frontmatter:
    defaults:
      language: en
      domain: general
  rumdl_config: .rumdl.toml   # custom rumdl config (optional)
```

### Flavor auto-detection

Without `--flavor`, KNOX inspects the output directory:

| Marker | Flavor | Use case |
|--------|--------|----------|
| `.obsidian/` directory | `obsidian` | Obsidian vault |
| `mkdocs.yml` file | `mkdocs` | MkDocs site |
| `_quarto.yml` or `*.qmd` files | `quarto` | Quarto project |
| none | `obsidian` | Default fallback |

Explicit `--flavor` always overrides auto-detection.

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

### From CLI

```bash
knox compile . --frontmatter language=en --frontmatter domain=ml
```

### In `knox write`

```bash
knox write "Transformer Architecture" --frontmatter language=en
```

Config defaults are merged first, then CLI flags override them.
