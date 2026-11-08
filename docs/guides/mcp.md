# MCP Server

kompend exposes its pipeline, wiki, memory, lint and reference operations as
MCP tools, so agent clients such as Hermes, Claude Code and Codex can call them
directly. The server registers **56 tools** and is an adapter over the same
library layer the CLI uses — it does not fork CLI subprocesses.

---

## Starting the server

```bash
# stdio transport; the wiki is the current directory
kompend mcp

# point at a specific vault from the global options or the command flag
kompend -w ./my-wiki mcp
kompend mcp --wiki-dir ./my-wiki
```

For a long-lived HTTP endpoint, run the agent with the MCP port:

```bash
kompend agent --mcp-port 8765
```

That serves the same tools over the `streamable-http` transport while the agent
runs.

---

## Tool families

| Family | Tools |
| --- | --- |
| Wiki | `wiki_parse`, `wiki_compile`, `wiki_capture`, `wiki_learn`, `wiki_write`, `wiki_ingest`, `wiki_index`, `wiki_organize`, `wiki_init`, `wiki_query`, `wiki_status`, `wiki_distill`, `wiki_consolidate`, `wiki_list_attachments`, `wiki_read_attachment`, `wiki_attachment_info` |
| Memory | `memory_add`, `memory_search`, `memory_synthesize` |
| Attachments | `attachments_add`, `attachments_ingest`, `attachments_no_ingest`, `attachments_remove`, `attachments_verify`, `attachments_adopt` |
| Config | `config_get`, `config_set`, `config_list`, `config_path`, `config_check` |
| Sources | `sources_list`, `sources_tag`, `sources_untag`, `sources_attr`, `sources_attr_rm`, `sources_export` |
| References | `references_list`, `references_export`, `references_fetch` |
| Lint and quality | `lint_all`, `lint_check`, `lint_semantic`, `lint_history`, `lint_show`, `lint_prune`, `lint_gaps`, `lint_remediate`, `lint_format`, `doctor`, `explain`, `review`, `provenance`, `log`, `list_parsers` |
| Packs | `pack_list`, `pack_install` |

---

## Lint and quality tools

These fifteen tools mirror the `kompend lint`, `kompend doctor`,
`kompend explain`, `kompend review` and `kompend provenance` commands.

| Tool | Arguments |
| --- | --- |
| `lint_all` | `wiki_dir=None`, `fix=True`, `config_file=None` |
| `lint_check` | `wiki_dir=None`, `fix=True`, `config_file=None` |
| `lint_format` | `wiki_dir=None`, `check=False`, `config_file=None` |
| `lint_gaps` | `wiki_dir=None`, `json_output=False`, `config_file=None` |
| `lint_history` | `wiki_dir=None`, `last=10`, `trend=False`, `json_output=False`, `config_file=None` |
| `lint_prune` | `wiki_dir=None`, `keep_last=None`, `max_age_days=None`, `config_file=None` |
| `lint_remediate` | `wiki_dir=None`, `dry_run=False`, `json_output=False`, `config_file=None` |
| `lint_semantic` | `wiki_dir=None`, `config_file=None` |
| `lint_show` | `wiki_dir=None`, `run=None`, `json_output=False`, `config_file=None` |
| `list_parsers` | `json_output=False` |
| `doctor` | `wiki_dir=None`, `verbose=False`, `json_output=False`, `config_file=None` |
| `explain` | `source`, `wiki_dir=None`, `run_id=None`, `config_file=None` |
| `review` | `wiki_dir=None`, `action="list"`, `slug=None`, `reason=None`, `approve_all=False`, `json_output=False` |
| `provenance` | `wiki_dir=None`, `page_slugs=None`, `with_content=False`, `json_output=False` |
| `log` | `wiki_dir=None`, `last=25`, `write=False`, `config_file=None` |

Every tool returns a dictionary. `doctor` reports the same last-run section and
the same recorded-embedding-model warning as `kompend doctor`; `explain` reads
the same ledger and run log as `kompend explain`.

---

## The tool contract gate

Tool names and shapes are enforced mechanically, so a refactor cannot silently
drop a tool or change an argument:

- **Names** — `scripts/checks/check_mcp_parity.py` compares the registered MCP
  tools with the CLI command tree.
- **Shapes** — the same gate compares the 14 lint/quality tools' input schemas,
  output schemas and returned payload keys with
  `scripts/checks/snapshots/mcp_tool_surface.json` (`explain` stays pinned by
  `tests/unit/test_mcp_quality.py`).

After an intentional change to a tool signature or return shape, regenerate the
snapshot and review the diff:

```bash
uv run python scripts/checks/check_mcp_parity.py --update-snapshot
uv run python scripts/checks/check_mcp_parity.py
```

The gate exits `0` when both contracts match, `1` on drift, and `2` when it
cannot judge (for example, a missing snapshot or an unimportable server). Never
hand-edit the snapshot.

---

## Adding a tool

1. Implement the function in `src/kompend/mcp/tools/` (or
   `src/kompend/mcp/tools/quality/` for the lint and quality family).
2. Export it from the package `__init__.py`.
3. Register it in `src/kompend/mcp/server.py` with an explicit
   `srv.tool(name="...")(fn)` call.
4. Add it to the quality-name pin in `tests/unit/test_mcp_quality.py` when it
   belongs to that family.
5. Run `--update-snapshot` and commit the regenerated shape snapshot with the
   tool change.
