# Lint Reference

KNOX lint checks compiled wikis for structural and semantic quality issues.

## Quick Start

```bash
# Run all structural + LLM-powered rules (LLM on by default)
knox lint run

# Structural rules only (no LLM calls)
knox lint run --no-llm

# Auto-fix supported issues
knox lint run --fix

# Run only specific rules
knox lint run --rule broken-wikilinks --rule orphan-page

# Machine-readable output for CI
knox lint run --json
```

LLM-powered rules use `compile.llm.default_model`; with no model configured
they are skipped with a note (structural rules still run).

## Commands

`knox lint` is a command group with five sub-commands:

| Sub-command | Description |
|------------|-------------|
| `knox lint run` | Run lint rules and save the result to history |
| `knox lint history` | Query past lint runs (scores, trends, details) |
| `knox lint gaps` | Analyze lint results for knowledge gaps and source recommendations |
| `knox lint remediate` | LLM-generate missing pages for lint findings |
| `knox lint prune` | Remove old history entries |

### `knox lint run`

```text
knox lint run [OPTIONS]

Lint a compiled wiki for quality issues. Every run is appended to
.knox/lint_history.jsonl for later querying.

Options:
  --fix              Auto-fix structural issues where possible
  --llm/--no-llm     LLM-powered lint rules (default). --no-llm runs
                     structural rules only
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

### `knox lint gaps`

Analyze the latest lint run for knowledge gaps and recommend source
material:

```bash
knox lint gaps
```

Gap recommendations cover missing topics, empty pages, isolated topics,
coverage gaps, and unresolved contradictions. `gaps` only recommends —
it never modifies the wiki.

### `knox lint remediate`

Generate missing concept pages for coverage-gap findings:

```bash
knox lint remediate           # generate pages (needs an LLM configured)
knox lint remediate --dry-run # preview without writing
```

Coverage-gap findings are matched against cached phase-1 extractions;
matched concepts get concept pages generated from that source material and
written to the wiki output directory. Unmatched concepts are skipped — run
`knox lint gaps` for recommendations on those.

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

Lint runs automatically after every compile (on by default). Disable it
per-run with `--no-lint`, or set `KNOX_COMPILE_LINT=0` in the environment:

```bash
knox compile ./sources          # lint runs automatically
knox compile ./sources --no-lint
KNOX_COMPILE_LINT=0 knox compile ./sources
```

## Structural Rules (11)

These rules are deterministic. Nine run entirely in-process; `rumdl` shells
out to the bundled rumdl binary.

| Rule | Severity | Auto-fix | What it checks |
|------|----------|----------|---------------|
| `image-embed` | WARNING | ✅ | Image links written as `[[img.png]]` instead of `![[img.png]]` |
| `broken-wikilinks` | ERROR | ✅ | `[[wikilinks]]` pointing to non-existent pages (image embeds excluded) |
| `missing-frontmatter` | WARNING | ✅ | Pages missing required frontmatter fields (`title`) |
| `orphan-page` | WARNING | ❌ | Pages with no incoming `[[wikilinks]]` from other pages |
| `duplicate-title` | ERROR | ❌ | Two or more pages with the same or similar title (case-insensitive) |
| `empty-page` | WARNING | ❌ | Pages with no body content |
| `heading-skip` | WARNING | ❌ | Heading level jumps (e.g. h1 → h3 without h2) |
| `stale-page` | WARNING | ❌ | Pages whose source files changed since last compile |
| `source-not-found` | ERROR | ❌ | Pages referencing source files that don't exist on disk |
| `orphan-attachments` | WARNING | ❌ | Attachment files whose slug matches no source (unrecoverable by compile-time auto-adoption) |
| `rumdl` | WARNING | ✅ | Markdown style issues on all vault folders except `sources/` (MD024 duplicate headings, MD013 line length, …) |

### Auto-fix behavior

When `--fix` is passed:

- **image-embed**: Rewrites `[[img.png]]` → `![[img.png]]`
- **broken-wikilinks**: Removes `[[...]]` brackets, keeping the link text
- **missing-frontmatter**: Adds `title` from the node title
- **rumdl**: Applies `rumdl check --fix` to the flagged files (self-written wiki notes get reformatted; `sources/` is never touched)
- Other rules: Cannot be auto-fixed (require manual editing or re-compilation)

The `rumdl` rule checks every `*.md` file under the vault root — wiki output,
index pages, and self-written notes — excluding `sources/`, `.knox/`, and
`attachments/`. Rule tuning comes from `compile.output.rumdl_config` in
`knox.toml` (same key as render-time formatting); when unset, rumdl
auto-discovers a `.rumdl.toml` in the vault. The rule reports `SKIP` when
rumdl is not installed.

## LLM-Powered Rules (4)

These rules call an LLM and are on by default; use `--no-llm` to skip them.

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
