---
name: cli-fartrun
description: Fartrun (Vibecode Cleaner Fartrun) is a local code health and security scanner for AI-generated and vibe-coded projects. It provides a Rust-powered scanner (zero token consumption), 29 MCP tools across 6 categories, save points with rollback, frozen file protection, and health-aware prompt generation. Use this skill when the user mentions "fartrun", "vibecode cleaner", "fartrun save point", or "fartrun rollback" in the context of AI-generated or vibe-coded projects. Also use when the user explicitly asks to run a fartrun scan, set up the fartrun MCP server, or integrate fartrun into their project toolchain.
---

# Fartrun

Fartrun (Vibecode Cleaner Fartrun) is a local code health and security scanner for AI-generated and vibe-coded projects. Uses a Rust scanner (zero token consumption) and exposes 29 MCP tools.

## Install

Fartrun is not on PyPI. Install via uvx:

```shell
uvx --from git+https://github.com/ChuprinaDaria/Vibecode-Cleaner-Fartrun.git fartrun
```

## MCP Server

Start with stdio transport:

```shell
uvx --from git+https://github.com/ChuprinaDaria/Vibecode-Cleaner-Fartrun.git fartrun mcp
```

For HTTP/SSE transport:

```shell
uvx --from git+https://github.com/ChuprinaDaria/Vibecode-Cleaner-Fartrun.git fartrun mcp --http --port 3001
```

Register the MCP server in your AI tool configuration.

## 29 MCP Tools (6 categories)

### Health (15)

run_health_scan, get_health_summary, get_unused_code, get_tech_debt, get_security_issues, get_module_graph, get_complexity_report, get_git_health, get_test_coverage, get_docs_quality, get_ui_issues, get_framework_check, get_outdated_deps, get_config_map, generate_health_report

### Status (4)

get_status, get_activity, detect_project_stack, search_code

### Prompts (1)

build_prompt

### Save Points (3)

create_save_point, rollback_save_point, list_save_points

### Frozen Files (3)

freeze_file, unfreeze_file, list_frozen

### Integrations (3)

install_context7, uninstall_context7, list_prompts

## Agent Workflow

Follow this sequence when working with a fartrun-scanned project:

1. Call `get_health_summary` to understand project state
2. Call `list_frozen` to know what cannot be touched
3. Call `create_save_point` to create a safety net before changes
4. Read findings, make changes respecting frozen files
5. Call `get_health_summary` to verify score did not drop
6. Report results to the user

## Trust Levels for Findings

| Finding | Trust Level | Action |
|---------|-------------|--------|
| Hardcoded secrets | Very High | Always flag |
| Unused imports | Very High | Safe to remove |
| Dead functions (Python/TS) | High | Remove if no dynamic dispatch |
| Dead functions (Django) | Medium | Verify signals and receivers first |
| N+1 queries | High | Fix with select_related or prefetch_related |
| Missing tests | High | Note, do not auto-generate without asking |
| Monster functions | High | Suggest refactoring, do not auto-refactor |
| Missing docstrings | Medium | Add only if asked or when creating new code |
| Tech debt TODOs | Low | Informational only |

## Constraint Precedence

When multiple rules apply, follow them in this priority order (highest first):

| Priority | Rule | Details |
|----------|------|---------|
| 1 (Hard) | Never edit frozen files | Read-only access for API understanding; ask user to unfreeze first if changes are needed |
| 2 (Hard) | Create save point before changes | Call `create_save_point` before any edit so you can roll back |
| 3 (Hard) | Verify health after changes | Call `get_health_summary` to confirm the score did not drop |
| 4 | Follow trust-level actions | Use the Trust Levels table to decide whether to auto-fix or only suggest |
| 5 | Prefer MCP tools over CLI | MCP tools provide faster feedback loops than CLI commands |

## CLI Reference

| Command | Description |
|---------|-------------|
| `fartrun scan` | Full 9-phase health scan (auto-detects stack) |
| `fartrun status` | One-screen project health summary |
| `fartrun save "message"` | Create tagged save point |
| `fartrun rollback sp-20260418-1` | Restore to a save point |
| `fartrun freeze src/auth.py` | Protect a file from edits |
| `fartrun unfreeze src/auth.py` | Remove protection |
| `fartrun list frozen` | List protected files |
| `fartrun list saves` | List save points |
| `fartrun prompt` | Build health-aware prompt for AI assistants |
| `fartrun mcp` | Start MCP server (stdio) |
| `fartrun gui` | Launch GUI |

### Global Flags

`-C <dir>` set working directory, `-v` verbose output, `-y` skip confirmations, `--json` JSON output, `--no-color` disable colors.

## Key Principles

Fartrun is a zero-token health scanner. It does not consume LLM tokens to analyze code.
