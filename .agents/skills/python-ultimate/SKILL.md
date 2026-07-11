---
name: python-ultimate
description: >-
  Comprehensive Python development skill covering coding standards, CLI development,
  linting, testing, debugging, refactoring, code review, auditing, documentation,
  project planning, and bulk operations. Use when writing, reviewing, refactoring,
  debugging, or documenting Python code; configuring linters; setting up CLI tools;
    planning features; performing code audits; checking Python antipatterns, forbidden
    methods, or bad style; or handling bulk operations (10+ files) that benefit from
    batch workflows instead of per-file iteration.
arguments:
  - name: sub-command
    description: >-
      Targeted guideline review to run. When omitted, runs a general antipattern check.
    enum:
      - naming
      - type-checking
      - imports
      - coding-standards
      - linter-rules
      - testing
      - debugging
      - audit
      - verification
      - code-review
      - help
    required: false
license: MIT
---

# Python Ultimate

Single Python reference with quick routes for standards, tooling, workflows, and best practices.

## Quick Start

**Writing code?** → Start with [Coding Standards](references/coding-standards.md)
**Checking naming?** → Go to [Naming Conventions](references/naming-conventions.md)
**Building a CLI?** → Go to [CLI Development](references/cli-development.md)
**Fixing linter errors?** → Go to [Linter Rules](references/linter-rules.md)
**Writing tests?** → Go to [Testing](references/testing.md)
**Debugging a bug?** → Go to [Debugging](references/debugging.md)
**Refactoring?** → Go to [Refactoring](references/refactoring.md)
**Reviewing code?** → Go to [Code Review](references/code-review.md)
**Auditing codebase?** → Go to [Auditing](references/auditing.md)
**Documenting?** → Go to [Documentation](references/documentation.md)
**Planning a feature?** → Go to [Planning](references/planning.md)
**Bulk operations?** → Go to [Refactoring](references/refactoring.md) (10+ files)

### Slash Commands

The `/python-ultimate` command accepts an optional `sub-command` argument to run
a targeted guideline review. When invoked without a sub-command (e.g., just
`/python-ultimate`), run a general antipattern check using the
Antipatterns / Forbidden Styles Index section below.

#### Routing

Match the `sub-command` argument to one of the sections below and follow its
workflow. If the argument does not match any known sub-command, explain the
available options (you can output the table from `python-ultimate help`).

______________________________________________________________________

### python-ultimate help

When the sub-command is `help` (or when the user asks for available commands),
render the following table so the user sees all available options:

```
Sub-Command                    │ What It Reviews                                        │ Reference
───────────────────────────────┼────────────────────────────────────────────────────────┼──────────────────────────────────
python-ultimate naming         │ File/dir variable naming (_file/_dir/_path suffixes)  │ references/naming-conventions.md
python-ultimate type-checking  │ TYPE_CHECKING guards and Optional[T] usage             │ references/type-checking.md
python-ultimate imports        │ Required-vs-optional import patterns                    │ references/imports-optional-dependencies.md
python-ultimate coding-standards│ Type hints, f-strings, pathlib, docstrings, comments, data modeling │ references/coding-standards.md
python-ultimate linter-rules   │ Ruff violations (E402, B007, B008, S108, etc.)         │ references/linter-rules.md
python-ultimate debugging      │ Systematic 4-phase debugging process                   │ references/debugging.md
python-ultimate testing        │ Test organization, fixtures, mocking, TDD, coverage    │ references/testing.md
python-ultimate audit          │ 6-dimension codebase audit                             │ references/auditing.md
python-ultimate verification   │ Evidence-based completion claims                       │ references/verification.md
python-ultimate code-review    │ Code review feedback evaluation                        │ references/code-review.md
```

______________________________________________________________________

### python-ultimate naming

Reviews file and directory variable naming conventions (`_file` / `_dir` / `_path`
suffixes).

**Workflow:**

1. Open `references/naming-conventions.md` and load the "1. Files and Directories" section
2. Scan the codebase for bare path names (`path`, `file`, `dir`, `output`, `source`, `target`) used as `Path` variables
3. Check for prefix patterns (`dir_output` → should be `output_dir`)
4. Find generic path variable names missing suffixes (`results` → ambiguous)
5. Report findings using the standard [antipattern response format](SKILL.md)

______________________________________________________________________

### python-ultimate type-checking

Scans for `TYPE_CHECKING` guards and `Optional[T]` usage.

**Workflow:**

1. Open `references/type-checking.md` and load the "Rule: Never Use TYPE_CHECKING Guards" section
2. Search for `TYPE_CHECKING` imports: `rg "TYPE_CHECKING" src/`
3. Search for `Optional[` usage: `rg "Optional\[" src/`
4. For each finding, identify the root cause (circular imports, type-only imports)
5. Recommend the appropriate alternative (shared types module, protocols, forward refs, local imports)
6. Report findings using the standard [antipattern response format](SKILL.md)

______________________________________________________________________

### python-ultimate imports

Reviews import patterns — distinguishes required vs optional dependencies.

**Workflow:**

1. Open `references/imports-optional-dependencies.md` and load the hard rule
2. Check `pyproject.toml` to determine which packages are required vs optional
3. Search for `try/except ImportError` patterns guarding required deps:
   `rg "except ImportError" src/`
4. For each match, classify: required dep → normal top-level import; optional dep → localized handling
5. Report findings using the standard [antipattern response format](SKILL.md)

______________________________________________________________________

### python-ultimate coding-standards

Reviews compliance with coding standards: type hints, f-strings, pathlib, docstrings,
comments, prohibited patterns, vague input/output types.

**Workflow:**

1. Open `references/coding-standards.md` and load relevant sections
2. For each prohibited pattern, search with targeted grep patterns:
   - `Optional\[` → must be `T | None`
   - `\.format\(` or `% ` formatting → must be f-strings
   - `os\.path\.` → must be `pathlib.Path`
   - `# noqa` → fix root issue
3. Check for vague input/output types with multiple `isinstance` checks
4. Scan for vague type annotations:
   - `rg ": object$"` or `rg "-> object"` → bare `object` as type
   - `rg "\bAny\b" src/ --include "*.py"` → `typing.Any` usage (flag each occurrence for review)
5. Scan for `None` misuse in dataclass fields and function signatures:
   - `rg "= None$"` → potential sentinel/absent patterns
   - Check dataclass fields with `list | None = None`, `dict | None = None` → should be `field(default_factory=...)`
   - Check functions returning `T | None` as error signal → should raise instead
6. Report findings using the standard [antipattern response format](SKILL.md)

______________________________________________________________________

### python-ultimate linter-rules

Reviews and fixes specific Ruff linter violations using context-aware patterns.

**Workflow:**

1. Open `references/linter-rules.md` and load the relevant rule section
2. Run `ruff check src/` to identify violations
3. For each violated rule, apply the context-specific fix pattern from the reference:
   - E402 → Move import to top of module
   - B007 → Prefix unused loop variable with `_`
   - B008 → Use `None` sentinel (except Typer `Annotated` parameters)
   - S108 → Use `tempfile` or `tmp_path` fixture
   - PLC0415 → Move import to module level
   - NPY002 → Use `default_rng()`
   - S311 → Use `secrets` for security contexts
4. Re-run `ruff check src/` to confirm fixes
5. Report findings using the standard [antipattern response format](SKILL.md)

______________________________________________________________________

### python-ultimate debugging

Initiates the systematic 4-phase debugging process.

**Workflow:**

1. Open `references/debugging.md` and load the full 4-phase process
2. **Phase 1 — Root Cause:** Reproduce the issue, read error messages, trace data flow from symptom to origin
3. **Phase 2 — Pattern:** Find working examples, compare against broken code, list every difference
4. **Phase 3 — Hypothesis:** Form a single testable hypothesis, make the smallest possible change to test it
5. **Phase 4 — Implementation:** Write a failing test first, implement the fix, verify all tests pass
6. Remember the iron law: **No fixes without root cause investigation first.**
7. If 3+ fixes have failed, stop and reassess architecture rather than continuing to guess

______________________________________________________________________

### python-ultimate testing

Reviews test organization, coverage, fixtures, mocking, and TDD compliance.

**Workflow:**

1. Open `references/testing.md` for patterns and standards
2. Check test file naming: `test_<module>.py` convention
3. Check test class naming: `Test<Name>` PascalCase
4. Check test method naming: `test_<description>` snake_case
5. Run coverage: `uv run pytest --cov=src --cov-report=term-missing`
6. Review fixture quality (descriptive names, proper scope, teardown)
7. Report findings using the standard [antipattern response format](SKILL.md)

______________________________________________________________________

### python-ultimate audit

Runs a 6-dimension codebase audit.

**Workflow:**

1. Open `references/auditing.md` and load all six dimensions
2. For each dimension (Architecture, Quality, Security, Performance, Testing, Maintainability):
   - Scan with grep/glob for relevant red flags
   - Rate findings by severity (Critical, High, Medium, Low)
3. Synthesize into an audit report using the format from `references/auditing.md`
4. Include an executive summary with health score and top recommendation
5. Include an action plan with immediate/short-term/medium-term/backlog items

______________________________________________________________________

### python-ultimate verification

Verifies that completion claims are backed by fresh evidence.

**Workflow:**

1. Open `references/verification.md` and load the iron law and gate function
2. For each claim, determine what command proves it
3. Run the full command, read the output, check the exit code
4. Only then state the result — with evidence, not assumptions
5. Forbidden words: `should`, `probably`, `might`, `likely`
6. Report results using the standard [antipattern response format](SKILL.md)

______________________________________________________________________

### python-ultimate code-review

Evaluates code review feedback and responds with technical rigor.

**Workflow:**

1. Open `references/code-review.md` and load the full workflow
2. Follow the READ → UNDERSTAND → VERIFY → EVALUATE → RESPOND → IMPLEMENT sequence
3. For each feedback item: verify against codebase reality, evaluate technical soundness
4. No performative agreement — respond with technical reasoning or push back with evidence
5. Push back when: suggestion breaks existing functionality, violates YAGNI, lacks full context

______________________________________________________________________

### Expected `/python-ultimate` Response Format

For consistency across agents, format all sub-command responses as:

1. **Summary** — total findings by severity and category
2. **Findings** — one item per finding: `file:line`, matched pattern class, short rationale
3. **Fix Guidance** — preferred replacement pattern with one concrete before/after example
4. **References** — direct links to the relevant section in `references/*.md`
5. **Verification** — exact command(s) run and observed result

Use concise, technical language. Avoid performative agreement and avoid speculative wording.

## Antipatterns / Forbidden Styles Index

Canonical quick-reference for common bad or forbidden patterns. Detailed rationale and examples stay in reference files.

| Category | Forbidden pattern | Preferred pattern | Source |
| -------------------- | ---------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Type checking | `TYPE_CHECKING` import guards | Refactor module boundaries, use forward refs/protocols | [references/type-checking.md](references/type-checking.md) |
| Type hints | `Optional[T]` | `T \| None` | [references/coding-standards.md](references/coding-standards.md) |
| String formatting | `.format()` and `%` formatting | f-strings | [references/coding-standards.md](references/coding-standards.md) |
| Paths | `os.path` usage | `pathlib.Path` | [references/coding-standards.md](references/coding-standards.md) |
| Lint suppression | `# noqa` to hide issues | Fix root issue | [references/coding-standards.md](references/coding-standards.md) |
| Import policy | Defensive `try/except ImportError` for required deps | Normal top-level imports for required deps | [references/imports-optional-dependencies.md](references/imports-optional-dependencies.md) |
| Path variable naming | Bare names like `path`, `file`, `output` for paths | Use `_file` / `_dir` suffixes | [references/naming-conventions.md](references/naming-conventions.md) |
| Path variable naming | Prefix forms `dir_x`, `file_x` | Suffix forms `x_dir`, `x_file` | [references/naming-conventions.md](references/naming-conventions.md) |
| Comments | Restating obvious code intent | Explain why/constraints only | [references/coding-standards.md](references/coding-standards.md) |
| Debugging workflow | Guess-and-check fixes before RCA | Follow 4-phase process | [references/debugging.md](references/debugging.md) |
| Debugging behavior | Repeated "one more try" after multiple failures | Stop and reassess architecture | [references/debugging.md](references/debugging.md) |
| Code review behavior | Performative agreement phrases | Technical response and evidence | [references/code-review.md](references/code-review.md) |
| Data modeling | Using `pydantic` for lightweight internal structs, or `dataclass` at trust boundaries | `dataclass` for internal DTOs; `pydantic` for validation/API boundaries | [references/coding-standards.md](references/coding-standards.md) |
| Type hints | `object` or `typing.Any` as type annotation without justification | Precise union types (`str \| int`), `typing.Protocol` for structural types | [references/coding-standards.md](references/coding-standards.md) |
| Default values | `None` as default for collections or as silent error signal | `field(default_factory=...)` for mutable defaults, exceptions for error states | [references/coding-standards.md](references/coding-standards.md) |

______________________________________________________________________

## Antipattern Tripwires (LLM-coding repeat offenders)

A focused list of the patterns that LLMs keep reintroducing, with the
one-line correction. This is the section to load when reviewing a
diff or starting a new file. The full table above (Antipatterns Index)
is the canonical reference; this is the spot-check for "am I about to
ship a known regression?".

| If you wrote... | Replace with... | Why |
|---|---|---|
| `from x import y` inside a function | Move to the module-level import block | PLC0415 — also a project rule (no inline imports) |
| `try: import x / except ImportError: x = None` | If `x` is required, just `import x`. If truly optional, declare in `pyproject.toml` `[project.optional-dependencies]` and document the probe | The guard hides the dependency and masks install failures |
| `model = "ollama/llama3.2"` as a function default | Read from `PipelineConfig` (single source of truth) | A hard-coded model in code fights user config |
| `text[:200]` or `x[:10]` as a literal | Read the limit from `DisplayConfig` / `BudgetConfig` / `PipelineConfig.display` | Silent data truncation past the cutoff |
| `role="compile.extract"` (bare string) | `role=LLMRole.COMPILE_EXTRACT` | Field-consistency scanner rejects; enum is the contract |
| `str(exc)[:200]` | `shorten_exception(exc)` (uses `MAX_CHARS_EXCEPTION`) | The limit belongs in a single constant |
| `global _FOO` | Pass as a function arg, class attr, or contextvar | Module state is hard to test and hard to reason about |
| `+ "/" +` for path construction | `pathlib.Path` with `/` | String concatenation breaks on Windows and on paths with spaces |
| `def x(self, ...)` where `self` is unused | `@staticmethod def x(...)` | The `no-self-use` warning is correct; make the method static |
| `except Exception: pass` / `except: return None` | Narrow the exception type, log it, and either re-raise or record the failure | Silent failures become "run compile first" mysteries |
| `def f(items: list[str] = [])` | `def f(items: list[str] | None = None)` and `items = items or []` inside, OR `field(default_factory=list)` in a dataclass | Mutable defaults are shared across all calls |

**Mental check before every commit:** "Did I just type a string literal
where an enum member should go, a number where a config key should
be, or an import inside a function?" If yes, fix it before staging.

______________________________________________________________________

## Coding Standards

Core Python coding rules. See [references/coding-standards.md](references/coding-standards.md) for full details.

### Type Hints

Mandatory everywhere. Use pipe syntax (`T | None`), never `Optional[T]`. Python 3.10+ required.

```python
def process(data: str, limit: int | None = None) -> list[str]: ...
```

**Never use `TYPE_CHECKING` guards.** See [references/type-checking.md](references/type-checking.md) for alternatives.

### String Formatting

Use f-strings only. No `.format()` or `%` formatting.

### Code Size Limits

| Target | Limit |
| -------- | ----------- |
| Module | < 250 lines |
| Function | < 75 lines |
| Class | < 200 lines |

### Docstrings

Google style. Required for public functions and classes.

### Prohibited Patterns

- `TYPE_CHECKING` guards
- `os.path` (use `pathlib.Path`)
- `Optional[T]` (use `T | None`)
- `# noqa` comments
- `sys.path` manipulation
- Defensive `try/except ImportError` for required dependencies (see [references/imports-optional-dependencies.md](references/imports-optional-dependencies.md))
- Vague or wide parameter/return types with hidden `isinstance`/`hasattr` checks and `None`-as-error returns (see [references/coding-standards.md](references/coding-standards.md))
- Using `pydantic` for lightweight internal structs, or `dataclass` for untrusted/API data (see [references/coding-standards.md](references/coding-standards.md))
- `object` or `typing.Any` as type annotations without justification (use precise union types, `typing.Protocol`, or narrowest possible type; see [references/coding-standards.md](references/coding-standards.md))
- `None` as default for mutable collections (use `field(default_factory=...)` in dataclasses; see [references/coding-standards.md](references/coding-standards.md))
- `None` as silent error return — prefer exceptions or result types (see [references/coding-standards.md](references/coding-standards.md))

______________________________________________________________________

## Naming Conventions

Variable naming standards for clarity and consistency. See [references/naming-conventions.md](references/naming-conventions.md) for full details.

### Files and Directories

| Pattern | Suffix | Example |
| ------------ | ------- | ------------------------------ |
| Files | `_file` | `output_file`, `config_file` |
| Directories | `_dir` | `cache_dir`, `output_dir` |
| Unknown type | `_path` | `data_path` (exceptional only) |

**Anti-patterns (always invalid for path variables):**

- Bare generic names: `path`, `file`, `folder`, `dir`, `directory`, `output`, `input`, `source`, `target`, `dest`
- Prefix instead of suffix: `dir_output`, `file_config`
- Missing suffix: `results`, `data`, `config` (ambiguous)

See [references/naming-conventions.md](references/naming-conventions.md) for the complete anti-pattern list.

### Test Naming

- Files: `test_<module>.py`
- Classes: `Test<DataProcessor>` (PascalCase with Test prefix)
- Methods: `test_<description>` (snake_case with test\_ prefix)

### Automated Validation

```bash
# Check a variable name
uv run assets/check_path_naming.py output_file
# Output: is_file

# Scan for violations
uv run assets/check_path_naming.py --check-files src/

# Scan for core forbidden patterns
uv run assets/check_path_naming.py --check-forbidden src/

# Repro fixture scan (see assets/examples)
uv run assets/check_path_naming.py --check-forbidden assets/examples/
```

Reference fixture files and expected output: [assets/examples/forbidden-scan-expected.md](assets/examples/forbidden-scan-expected.md)

______________________________________________________________________

## CLI Development

Building Python CLIs with Typer or Click. See [references/cli-development.md](references/cli-development.md).

### Framework Selection

Use **Typer** for new projects (type-hint driven, less boilerplate). Use **Click** for complex parameter handling.

### Key Patterns

- Parameter validation with type hints
- Rich output formatting
- Environment variable integration
- Exit codes for error states

______________________________________________________________________

## Linter Rules

Context-aware fixes for Ruff linter rules. See [references/linter-rules.md](references/linter-rules.md).

### Covered Rules

| Rule | Description | Quick Fix |
| ------- | ------------------------------ | -------------------------- |
| E402 | Module-level import not at top | Move imports to top |
| B007 | Unused loop variable | Prefix with `_` |
| B008 | Function call in default arg | Use `None` sentinel |
| S108 | Hardcoded temp file path | Use `tempfile` |
| PLC0415 | Import not at top-level | Move to module level |
| NPY002 | Legacy numpy random | Use `numpy.random` |
| S311 | Standard random | Use `secrets` for security |

### Typer Exception

B008 is allowed for Typer `Annotated` parameters. See [references/linter-rules.md](references/linter-rules.md).

______________________________________________________________________

## Testing

Test organization, fixtures, mocking, and TDD. See [references/testing.md](references/testing.md).

### Quick Commands

```bash
uv run pytest -v --tb=short
uv run pytest --cov=src --cov-report=term-missing
```

### Key Practices

- Co-located tests: `<module>_test.py` alongside implementation
- 90%+ coverage target
- Fixtures in `conftest.py`
- Parameterized testing for multiple inputs
- `unittest.mock` for external dependencies

### TDD Cycle

Red → Green → Refactor. No production code without a failing test first. See [references/testing.md](references/testing.md).

______________________________________________________________________

## Debugging

Systematic 4-phase debugging process. See [references/debugging.md](references/debugging.md).

### Iron Law

**No fixes without root cause investigation first.**

### 4-Phase Process

1. **Root Cause** — Reproduce, isolate, trace data flow
2. **Pattern Analysis** — Identify state changes, timing issues
3. **Hypothesis** — Form testable prediction
4. **Implementation** — Minimal fix, verify with test

### Red Flags

- "Let me just try changing X"
- Fixing symptoms without understanding cause
- Multiple failed fix attempts

______________________________________________________________________

## Refactoring

Find → Replace → Verify workflow. See [references/refactoring.md](references/refactoring.md).

### Workflow

1. **Find** — Grep for target pattern
2. **Replace** — Edit with `replace_all` for bulk changes
3. **Verify** — Run tests, check for regressions

### Code Transfer

Line-based code movement between files. See [references/refactoring.md](references/refactoring.md).

______________________________________________________________________

## Code Review

Receiving and evaluating code review feedback. See [references/code-review.md](references/code-review.md).

### Workflow

Read → Understand → Verify → Evaluate → Respond → Implement

### Key Principles

- No performative agreement
- Push back with technical reasoning
- Verify feedback before implementing
- Evaluate: is the suggestion correct?

______________________________________________________________________

## Auditing

6-dimension codebase analysis. See [references/auditing.md](references/auditing.md).

### Dimensions

1. **Architecture** — Structure, modularity, dependencies
2. **Quality** — Readability, complexity, duplication
3. **Security** — Input validation, secrets, injection
4. **Performance** — Bottlenecks, memory, I/O
5. **Testing** — Coverage, quality, edge cases
6. **Maintainability** — Documentation, technical debt

### Severity Ratings

Critical → High → Medium → Low

______________________________________________________________________

## Documentation

10-section documentation structure. See [references/documentation.md](references/documentation.md).

### Workflow

Explore → Map → Read → Synthesize

### Sections

Project Overview, Architecture, Key Components, Data Flow, API Reference, Configuration, Setup Guide, Development Guide, Testing, Deployment

### Mermaid Diagrams

Use for architecture, sequence, and flowchart visualizations.

______________________________________________________________________

## Planning

PLAN.md living document for feature implementation. See [references/planning.md](references/planning.md).

### When to Use

Features spanning 3-15 prompts. Self-contained for fresh sessions.

### Structure

Goal → Context → Phases → Validation → Progress → Decisions → Notes

______________________________________________________________________

## Project Setup

Project structure, dependencies, and imports. See [references/project-setup.md](references/project-setup.md).

### Key Tools

- **uv** for dependency management
- **src layout** for packages
- **pyproject.toml** for configuration

### Import Order

1. Standard library
2. Third-party
3. Local (absolute imports)

______________________________________________________________________

## File Analysis

Non-destructive file and codebase analysis. See [references/file-analysis.md](references/file-analysis.md).

### Tools

- `stat` for metadata
- `wc` for line counts
- Grep for pattern searching
- Glob for file discovery

______________________________________________________________________

## Type Checking Alternatives

Never use `TYPE_CHECKING` guards. See [references/type-checking.md](references/type-checking.md).

### Alternatives

1. Extract shared types to dedicated modules
2. Use protocols for structural typing
3. Forward references (string literals)
4. Local imports (last resort)

______________________________________________________________________

## Bulk Operations

High-efficiency Python execution for 10+ file operations. **90-99% token savings** vs. iterative approaches.

**When to use:**

- Bulk operations (10+ files)
- Complex multi-step workflows
- Iterative processing across many files
- User mentions efficiency/performance

**Workflow pattern:**

1. **Analyze locally** — Use metadata operations (file counts, grep patterns)
2. **Process locally** — Execute all transformations in Python
3. **Return summary** — Report counts, not full data

**Example patterns:**

```python
# Bulk refactor across 50 files
from pathlib import Path
import re

files = list(Path('.').glob('**/*.py'))
modified = 0

for f in files:
    content = f.read_text()
    new_content = re.sub(r'old_pattern', 'new_pattern', content)
    if new_content != content:
        f.write_text(new_content)
        modified += 1

result = {'files_scanned': len(files), 'files_modified': modified}
```

```python
# Code audit metadata extraction
from pathlib import Path
import ast

files = list(Path('src').glob('**/*.py'))
complexity_issues = []

for f in files:
    tree = ast.parse(f.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Calculate simple complexity metric
            nested = sum(1 for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While)))
            if nested > 10:
                complexity_issues.append({'file': str(f), 'function': node.name, 'complexity': nested})

result = {'files_audited': len(files), 'high_complexity': len(complexity_issues)}
```

**Best practices:**

- ✅ Return summaries, not full data
- ✅ Batch operations where possible
- ✅ Use `pathlib.Path` for file operations
- ✅ Handle errors gracefully, return error counts
- ❌ Don't read full source into context when metadata suffices
- ❌ Don't process files one-by-one interactively

**Token savings scale with file count:**

| Files | Interactive | Bulk Operation | Savings |
| ----- | ------------ | -------------- | ------- |
| 10 | ~5K tokens | ~500 tokens | 90% |
| 50 | ~25K tokens | ~600 tokens | 97.6% |
| 100 | ~150K tokens | ~1K tokens | 99.3% |

______________________________________________________________________

## Reference Files

All detailed content lives in `references/`. Load only what you need:

| File | Content |
| ---------------------------------- | --------------------------------------------------------- |
| `coding-standards.md` | Type hints, formatting, size limits, docstrings, comments, data modeling |
| `cli-development.md` | Typer/Click, parameters, Rich output, env vars |
| `linter-rules.md` | Ruff rules E402, B007, B008, S108, PLC0415, NPY002, S311 |
| `testing.md` | Fixtures, parameterized, mocking, TDD, coverage |
| `type-checking.md` | TYPE_CHECKING alternatives, protocols, forward refs |
| `debugging.md` | 4-phase process, red flags, rationalizations |
| `refactoring.md` | Bulk operations, code transfer, safety checks |
| `code-review.md` | Receiving feedback, push back, evaluation |
| `auditing.md` | 6-dimension analysis, severity ratings |
| `documentation.md` | 10-section structure, Mermaid diagrams |
| `planning.md` | PLAN.md template and example |
| `file-analysis.md` | Metadata, line counting, pattern searching |
| `project-setup.md` | Project structure, uv, imports |
| `verification.md` | Pre-commit hooks, tox, Makefile targets |
| `imports-optional-dependencies.md` | Required vs optional dependency import patterns |
