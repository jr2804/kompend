# Lint Reference

KNOX lint checks compiled wikis for structural and semantic quality issues.

## Quick Start

```bash
# Run all 8 structural rules (default, zero LLM cost)
knox lint run

# Auto-fix supported issues
knox lint run --fix

# Also run LLM-powered rules (slow, expensive)
knox lint run --llm

# Run only specific rules
knox lint run --rule broken-wikilinks --rule orphan-page

# Machine-readable output for CI
knox lint run --json
```

## Commands

`knox lint` is a command group with three sub-commands:

| Sub-command | Description |
|------------|-------------|
| `knox lint run` | Run lint rules and save the result to history |
| `knox lint history` | Query past lint runs (scores, trends, details) |
| `knox lint prune` | Remove old history entries |

### `knox lint run`

```text
knox lint run [OPTIONS]

Lint a compiled wiki for quality issues. Every run is appended to
.knox/lint_history.jsonl for later querying.

Options:
  --fix              Auto-fix structural issues where possible
  --llm              Enable LLM-powered lint rules (slow, expensive)
  --rule TEXT        Run only specific rule(s) (repeatable)
  --json             Output in JSON format for machine readability
```

Wiki paths are resolved from `--wiki-dir` (or `KNOX_WIKI_DIR`). See [Configuration](./configuration.md).

### `knox lint history`

```text
knox lint history [OPTIONS]

Show past lint runs from .knox/lint_history.jsonl.

Options:
  --last INT, -n     Show last N entries (default 10)
  --trend            Show score trend sparkline
  --json             Output in JSON format
```

Examples:

```bash
# Show last 10 lint runs
knox lint history

# Show only the most recent run
knox lint history -n 1

# Show score trend over time (Unicode sparkline)
knox lint history --trend

# Machine-readable output for dashboards
knox lint history --json
```

### `knox lint prune`

```text
knox lint prune [OPTIONS]

Remove old entries from .knox/lint_history.jsonl.
At least one of --keep-last or --max-age-days is required.

Options:
  --keep-last INT          Keep only the most recent N entries
  --max-age-days INT       Remove entries older than N days
```

Examples:

```bash
# Keep only the last 50 runs
knox lint prune --keep-last 50

# Remove entries older than 90 days
knox lint prune --max-age-days 90

# Combined: keep last 50 AND younger than 90 days
knox lint prune --keep-last 50 --max-age-days 90
```

### `knox compile --lint`

Run lint automatically after compilation:

```bash
knox compile ./sources --lint
```

## Structural Rules (8)

These rules are deterministic and require no LLM calls.

| Rule | Severity | Auto-fix | What it checks |
|------|----------|----------|---------------|
| `broken-wikilinks` | ERROR | ✅ | `[[wikilinks]]` pointing to non-existent pages |
| `missing-frontmatter` | WARNING | ✅ | Pages missing required frontmatter fields (`title`) |
| `orphan-page` | WARNING | ❌ | Pages with no incoming `[[wikilinks]]` from other pages |
| `duplicate-title` | ERROR | ❌ | Two or more pages with the same or similar title (case-insensitive) |
| `empty-page` | WARNING | ❌ | Pages with no body content |
| `heading-skip` | WARNING | ❌ | Heading level jumps (e.g. h1 → h3 without h2) |
| `stale-page` | WARNING | ❌ | Pages whose source files changed since last compile |
| `source-not-found` | ERROR | ❌ | Pages referencing source files that don't exist on disk |

### Auto-fix behavior

When `--fix` is passed:

- **broken-wikilinks**: Removes `[[...]]` brackets, keeping the link text
- **missing-frontmatter**: Adds `title` from the node title
- Other rules: Cannot be auto-fixed (require manual editing or re-compilation)

## LLM-Powered Rules (4)

These rules call an LLM via LiteLLM and are opt-in with `--llm` or `--rule <name>`.

| Rule | What it checks |
|------|---------------|
| `factual-consistency` | Claims in a page contradict source material |
| `clarity-score` | Readability, coherence, conciseness (1-10 score) |
| `coverage-gap` | Important concepts from sources missing in the wiki |
| `contradiction-detection` | Pairs of wiki pages that make contradictory claims, with confidence and severity |

LLM rules require a configured LLM. If no LLM is available, they report `SKIP`.

## Health Score

The health score is computed as:

```text
health_score = 100 × (PASS results) / (total results)
```

A perfect wiki scores 100%. `knox lint run` reports the score but does **not**
gate on it — the exit code is 0 unless the run itself fails. To fail CI on a
low score, parse the JSON output:

```bash
knox lint run --json > lint-report.json
python -c "import json,sys; sys.exit(0 if json.load(open('lint-report.json'))['health_score']>=80 else 1)"
```

## JSON Output

```bash
knox lint run --json
```

Returns:

```json
{
  "health_score": 87.5,
  "total_checks": 8,
  "failures": 1,
  "passes": 7,
  "skips": 0,
  "results": [
    {
      "rule": "broken-wikilinks",
      "outcome": "fail",
      "severity": "error",
      "message": "Broken wikilink [[Missing]] in page 'Introduction'",
      "page_id": "node_0",
      "fix_applied": false,
      "details": {"target": "Missing"}
    }
  ]
}
```

## History

Every `knox lint run` appends a structured entry to `.knox/lint_history.jsonl`. Each entry includes:

| Field | Content |
|-------|---------|
| `timestamp` | ISO 8601 UTC |
| `duration_seconds` | Wall-clock time of the lint run |
| `options` | Active rules, `enable_llm`, `fix` |
| `llm` | Model name + token usage (or `null` for structural-only) |
| `config_snapshot` | Relevant compile settings (compiler, chunk size, model, fallback chain) |
| `wiki_meta` | Node count |
| `report` | Full `LintReport` (health score, per-rule results) |

Query with `knox lint history` (see above). Prune old entries with `knox lint prune`.

## CI Integration

### GitHub Actions

```yaml
- name: Lint compiled wiki
  run: |
    knox compile ./sources
    knox lint run --json > lint-report.json
    python -c "import json,sys; sys.exit(0 if json.load(open('lint-report.json'))['health_score']>=80 else 1)"
```

### Pre-commit

```yaml
- repo: local
  hooks:
    - id: knox-lint
      name: knox lint
      entry: knox lint run --json
      language: system
      pass_filenames: false
```

The hook runs lint and writes JSON but, like `knox lint run` itself, exits 0
unless the run fails. To make a pre-commit hook gate on the score, wrap the
JSON parse (see [Health Score](#health-score)) in a small script and point
`entry` at it.

## FAQ

### How do I add a custom lint rule?

Create a class that satisfies the `LintRule` protocol:

```python
from knox.lint.base import LintResult, LintRuleOutcome

class MyCustomRule:
    name = "my-custom-rule"
    description = "What this rule checks"

    def check(self, wiki, *, sources_dir=None):
        # Return list of LintResult
        return [LintResult(rule=self.name, outcome=LintRuleOutcome.PASS)]

    def fix(self, wiki, results):
        return results  # or implement auto-fix
```

### Why does lint say I have broken wikilinks?

A `[[wikilink]]` points to a page title that doesn't exist in the compiled wiki. This can happen when:

- The linked page was not compiled (source file missing or parse failure)
- The link text has a typo or different capitalization
- The page was removed but links weren't updated

### What counts as a health score violation?

Any lint result with `outcome: "fail"` reduces the health score. The score is the percentage of passing checks out of total checks run.

### Why are some rules skipped?

Rules that require a sources directory (`stale-page`, `source-not-found`) will skip if the sources directory doesn't exist or is empty. LLM rules skip when no LLM client is configured.
