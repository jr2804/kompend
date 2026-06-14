# Wiki

This directory is a KNOX wiki project. Sources are read from
`sources/`, the compiled wiki is written to `wiki/`, and runtime state
lives in `.knox/`.

## Quickstart

1. Drop source documents into `sources/`.
2. Run `knox compile` to build the wiki.
3. Inspect the output in `wiki/`.

## Configuration

Project configuration lives in `.knox/config.toml`. See
[the configuration guide](https://knox.readthedocs.io/guides/config.html)
for the full reference, or browse the comments in the config file
itself.

To inspect or modify configuration values from the CLI:

    knox config list
    knox config get compile.chunk_size
    knox config set compile.chunk_size 1024

## Layout

| Path           | Purpose                                          |
| -------------- | ------------------------------------------------ |
| `sources/`     | Input documents (PDF, DOCX, MD, …)               |
| `wiki/`        | Compiled output (Markdown + Obsidian frontmatter)|
| `.knox/`       | Internal state (cache, config) — gitignored      |
| `.knox/config.toml` | Project configuration (created by `knox init`) |
| `NOTES.md`     | User-authored notes (read-only to KNOX)          |

## Next steps

- See the [KNOX documentation](https://knox.readthedocs.io/) for
  ingestion, compilation, and deployment guides.
- Run `knox doctor` to verify the environment.
