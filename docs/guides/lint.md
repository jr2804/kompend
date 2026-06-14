# Lint Reference

KNOX lint checks compiled wikis for structural and semantic quality issues.

## Quick Start

```bash
# Run all 8 structural rules (default, zero LLM cost)
knox lint

# Auto-fix supported issues
knox lint --fix

# Also run LLM-powered rules (slow, expensive)
knox lint --llm

# Run only specific rules
knox lint --rule broken-wikilinks --rule orphan-page

# Machine-readable output for CI
knox lint --json

# Fail CI if health score below threshold
knox lint --threshold 80
```

## Commands

### `knox lint`

```text
knox lint [OPTIONS]

Lint a compiled wiki for quality issues.

Options:
  --fix              Auto-fix structural issues where possible
  --llm              Enable LLM-powered lint rules (slow, expensive)
  --rule TEXT        Run only specific rule(s) (repeatable)
  --threshold INT    Exit with error if health score below this value (0-100)
  --json             Output in JSON format for machine readability
```

Wiki paths are resolved from `--wiki-dir` (or `KNOX_WIKI_DIR`). See [Configuration](./configuration.md).

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

A perfect wiki scores 100%. Use `--threshold` in CI:

```bash
knox lint --threshold 80
# Exit code 0 if score ≥ 80, exit code 1 otherwise
```

## JSON Output

```bash
knox lint --json
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

## CI Integration

### GitHub Actions

```yaml
- name: Lint compiled wiki
  run: |
    knox compile ./sources
    knox lint --threshold 80 --json > lint-report.json
```

### Pre-commit

```yaml
- repo: local
  hooks:
    - id: knox-lint
      name: knox lint
      entry: knox lint --threshold 80
      language: system
      pass_filenames: false
```

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
