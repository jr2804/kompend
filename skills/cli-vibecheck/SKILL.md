---
name: cli-vibecheck
description: vibecheck is an ESLint-like tool for catching AI-generated code smells. Use when the user mentions "vibecheck", "AI slop", "code smells", "AI-generated code", or wants to scan for patterns like hardcoded secrets, empty catch blocks, console.log pollution, TypeScript `any` types, hedging comments, god functions, or stub detection in JavaScript/TypeScript/Python codebases. Also use when setting up the vibecheck MCP server or integrating it into CI/CD.
---

# vibecheck

35 rules for catching AI-generated code smells. ESLint for AI slop.

## Install

```shell
# Run directly (no install)
npx @yuvrajangadsingh/vibecheck .

# Install globally
npm install -g @yuvrajangadsingh/vibecheck

# Standalone binary (no Node required)
curl -fsSL https://github.com/yuvrajangadsingh/vibecheck/releases/latest/download/vibecheck-darwin-arm64 -o vibecheck
chmod +x vibecheck
./vibecheck .
```

Or via mise (adds to PATH):

```shell
mise use npm:@yuvrajangadsingh/vibecheck
```

## CLI Commands

```shell
vibecheck [path] [options]

Options:
  -c, --config <file>     Path to config file
  -d, --diff              Only scan lines changed in git diff (unstaged)
  --staged                Only scan lines changed in git diff --cached (staged)
  --json                  Output as JSON (for CI pipelines)
  --ignore <patterns...>  Additional ignore patterns
  --severity <level>      Minimum severity: error, warn, info (default: warn)
  -q, --quiet             Only show summary
  --mcp                   Start MCP server (stdio transport)
  -v, --version           Show version
  -h, --help              Show help
```

## What It Catches

### Security (error severity)

| Rule | What it detects |
|------|-----------------|
| `no-hardcoded-secrets` | API keys, tokens, passwords in source code |
| `no-eval` | `eval()` and `new Function()` calls |
| `no-sql-concat` | SQL queries built with string concatenation |
| `no-innerhtml` | `innerHTML` and `dangerouslySetInnerHTML` usage |

### Error Handling

| Rule | Severity | What it detects |
|------|----------|-----------------|
| `no-empty-catch` | error | Empty catch blocks that silently swallow errors |
| `no-console-error-only` | warn | Catch blocks that only `console.error` without rethrowing |
| `no-swallowed-promise` | warn | `.then()` chains without `.catch()` |

### Code Quality

| Rule | Severity | What it detects |
|------|----------|-----------------|
| `no-console-pollution` | warn | `console.log`/`debug`/`info` left in production code |
| `no-ai-todo` | info | AI-generated placeholder TODOs (`TODO: implement`, `FIXME: add`) |
| `no-god-function` | warn | Functions over 80 lines |

### AI-Specific Tells

| Rule | Severity | What it detects |
|------|----------|-----------------|
| `no-obvious-comments` | info | Comments that restate code (`// initialize the counter`) |
| `no-ts-any` | warn | TypeScript `any` types and `as any` casts |

### Framework

| Rule | Severity | What it detects |
|------|----------|-----------------|
| `no-express-unhandled` | warn | Async Express routes without error handling |
| `no-error-info-leak` | error | Error internals (`err.message`, `err.stack`) leaked to HTTP responses |
| `no-flask-debug` | warn | Flask `app.run(debug=True)` in production |

### Python

| Rule | Severity | What it detects |
|------|----------|-----------------|
| `no-py-eval` | error | `eval()`, `exec()`, `os.system()`, `subprocess(shell=True)` |
| `no-py-sql-concat` | error | SQL queries built with f-strings or `.format()` |
| `no-bare-except` | error | `except:` without specifying exception type |
| `no-pass-except` | error | `except: pass` silently swallowing errors |
| `no-star-import` | warn | `from module import *` namespace pollution |
| `no-mutable-default` | warn | Mutable default arguments (`def foo(bar=[])`) |
| `no-py-print` | warn | `print()` left in production code |
| `no-py-obvious-comments` | info | Comments restating code (`# initialize the counter`) |
| `no-py-stub-function` | warn | Stub functions raising `NotImplementedError` with no body |
| `no-py-hedging-comments` | info | Hedging comments (`# should work`, `# might not be ideal`) |
| `no-type-ignore-blanket` | warn | Blanket `# type: ignore` without specific error code |
| `no-unused-protocol` | info | `Protocol` class defined but never referenced |

## MCP Server

vibecheck ships with a built-in MCP server so AI coding agents can scan code for slop automatically.

Start the MCP server:

```shell
vibecheck --mcp
# Or use the dedicated binary
vibecheck-mcp
```

Add to your AI assistant config:

```json
{
  "mcpServers": {
    "vibecheck": {
      "command": "npx",
      "args": ["-y", "@yuvrajangadsingh/vibecheck", "--mcp"]
    }
  }
}
```

### MCP Tools (3 total)

| Tool | Description |
|------|-------------|
| `scan-files` | Scan files/directories for AI slop patterns |
| `scan-diff` | Scan a git diff for slop in changes |
| `get-rules` | List all available rules |

## Diff Mode

Scan only the lines you changed, not the entire codebase. Useful for pre-commit hooks and incremental CI.

```shell
# Scan unstaged changes
vibecheck --diff .

# Scan staged changes (pre-commit hook)
vibecheck --staged .

# CI-friendly output
vibecheck --staged --json .
```

## Pre-commit Hook

Add to `.git/hooks/pre-commit` or use with [husky](https://github.com/typicode/husky):

```sh
#!/bin/sh
npx @yuvrajangadsingh/vibecheck --staged .
```

Or via pre-commit config:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/yuvrajangadsingh/vibecheck
    rev: v1.2.0
    hooks:
      - id: vibecheck
```

## Configuration

Create `.vibecheckrc` in your project root:

```json
{
  "rules": {
    "no-console-pollution": "off",
    "no-obvious-comments": "warn",
    "no-ts-any": "error"
  },
  "ignore": [
    "node_modules",
    "dist",
    "*.test.ts"
  ]
}
```

All rules are on by default at their recommended severity.

## GitHub Action

```yaml
- uses: yuvrajangadsingh/vibecheck@v1.2.0
  with:
    severity: warn       # minimum severity to report
    fail-on: error       # fail the check at this severity
    ignore: "tests/**"   # comma-separated ignore patterns
```

The action automatically scans only files changed in the PR. On push events, it scans the full repo.

## Comparison with Similar Tools

| Feature | vibecheck | ESLint | sloppylint |
|---------|-----------|--------|------------|
| AI-specific rules | 34 | 0 | 100+ |
| Hedging comments | yes | no | yes |
| Hallucinated imports | yes | no | yes |
| Stub detection | yes | no | yes |
| Section dividers | yes | no | no |
| Step comments | yes | no | no |
| Deep nesting | yes | plugin | no |
| GitHub Action | yes | community | no |
| MCP server | yes | no | no |
| VS Code extension | yes | yes | no |
| Zero config | yes | no | no |
| Languages | JS/TS/Python | JS/TS | Python |

## VS Code Extension

```sh
ext install yuvrajangadsingh.vibecheck-linter
```

Available on the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=yuvrajangadsingh.vibecheck-linter). Works in VS Code, Cursor, and Windsurf.

Features: inline error/warning/info markers, Problems panel integration, status bar count, runs on save.

## Example Output

```
  src/api/routes.ts
    12:5    error  no-hardcoded-secrets   Hardcoded secret detected. Use environment variables instead.
    45:3    error  no-empty-catch         Empty catch block swallows errors silently.
    89:1    warn   no-console-pollution   console.log left in production code.

  src/utils/db.ts
    34:5    error  no-sql-concat          SQL query built with string concatenation.

  4 problems (3 errors, 1 warning)
  2 files with issues out of 47 scanned (0.8s)
```