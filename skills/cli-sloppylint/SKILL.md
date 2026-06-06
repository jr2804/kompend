---
name: cli-sloppylint
description: Sloppylint detects AI-generated code anti-patterns in Python codebases. Use when the user mentions "sloppylint", "AI slop", "detect AI code patterns", "cross-language leakage", "hallucinated imports", "placeholder code", or wants to audit AI-generated code for common failure patterns like mutable defaults, bare except, dead code, or wrong-language syntax in Python (e.g., .push(), .length from JS/Java).
---

# Sloppylint

Sloppylint detects AI-generated code anti-patterns that traditional linters miss. It catches patterns uniquely common in AI-generated code.

## Install

Install from PyPI:

```shell
pip install sloppylint
pip install sloppylint[rich]      # with colored output (recommended)
pip install sloppylint[all]       # with all optional features
```

Or via pipx:

```shell
pipx install sloppylint
pipx install sloppylint[rich]
```

## What It Catches

### The Three Axes of AI Slop

| Axis | What It Detects | Examples |
|------|----------------|----------|
| **Noise** | Debug artifacts, redundant comments | `print()`, `# increment x` above `x += 1` |
| **Lies** | Hallucinations, placeholders | `def process(): pass`, mutable defaults |
| **Soul** | Over-engineering, bad style | God functions, deep nesting, hedging comments |
| **Structure** | Anti-patterns | Bare except, star imports, single-method classes |

### Critical Severity

**mutable_default_arg** - AI's favorite mistake:
```python
# Bug: shared state between calls
def process_items(items=[]):
    items.append(1)
    return items

# Fix: Use None and initialize inside
def process_items(items=None):
    if items is None:
        items = []
    items.append(1)
    return items
```

**bare_except** - Catches SystemExit, KeyboardInterrupt:
```python
# Bug: swallows Ctrl+C!
try:
    risky_operation()
except:
    pass

# Fix: Catch specific exceptions
try:
    risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
```

### High Severity

**pass_placeholder** - AI gave up:
```python
def validate_email(email):
    pass  # TODO: implement
```

**hedging_comment** - AI uncertainty:
```python
x = calculate()  # should work hopefully
```

## Multi-Language Detection

LLMs leak patterns from other languages:

| Language | Example Mistakes | Python Fix |
|----------|-----------------|------------|
| JavaScript | `.push()`, `.length`, `.forEach()` | `.append()`, `len()`, `for` loop |
| Java | `.equals()`, `.toString()`, `.isEmpty()` | `==`, `str()`, `not obj` |
| Ruby | `.each`, `.nil?`, `.first`, `.last` | `for` loop, `is None`, `[0]`, `[-1]` |
| Go | `fmt.Println()`, `nil` | `print()`, `None` |
| C# | `.Length`, `.Count`, `.ToLower()` | `len()`, `len()`, `.lower()` |
| PHP | `strlen()`, `array_push()`, `explode()` | `len()`, `.append()`, `.split()` |

## CLI Commands

```shell
sloppylint .                    # Scan current directory
sloppylint src/ tests/          # Scan multiple directories
sloppylint --severity high      # Only critical/high issues
sloppylint --lenient           # Same as --severity high
sloppylint --strict             # Report everything
sloppylint --ci                 # Exit 1 if any issues
sloppylint --max-score 50       # Exit 1 if score > 50
sloppylint --output report.json # Export JSON report
sloppylint --ignore "tests/*"   # Exclude patterns
sloppylint --disable magic_number  # Skip specific checks
sloppylint --version            # Show version
```

## Output Format

```
# CRITICAL (2 issues)
# ============================================================
#   src/api.py:23  mutable_default_arg
#     Mutable default argument - use None instead
#     > def process(items=[]):
#
#   src/db.py:15  bare_except
#     Bare except catches everything including SystemExit
#     > except:

# SLOPPY INDEX
# ══════════════════════════════════════════════════
# Information Utility (Noise)    : 24 pts
# Information Quality (Lies)      : 105 pts
# Style / Taste (Soul)            : 31 pts
# Structural Issues               : 45 pts
# ──────────────────────────────────────────────────
# TOTAL SLOP SCORE                : 205 pts
#
# Verdict: SLOPPY
```

## Configuration

Via `pyproject.toml`:

```toml
[tool.sloppy]
ignore = ["tests/*", "migrations/*"]
disable = ["magic_number", "debug_print"]
severity = "medium"
max-score = 100
ci = false
format = "detailed"  # or "compact" or "json"
```

## Exit Codes

- `0` - No issues found or score below threshold
- `1` - Issues found (when `--ci` or `--max-score` exceeded)

## Key Principles

- Sloppylint **complements** traditional linters (Pylint, Flake8, Ruff), not replaces them
- It catches patterns uniquely common in AI-generated code that other tools miss
- Use `--ci` for automated gates, `--lenient` for quick audits
- JSON output (`--output report.json`) enables CI/CD integration