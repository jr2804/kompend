# DOX — kompend.lint

<!-- on entry: loaded when agent enters src/kompend/lint -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/CODE.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Linting and remediation for compiled wiki pages. Structural rules run
programmatically; LLM-powered rules use ``LintRuleAgent`` for semantic checks.
Remediation uses ``RemediationAgent`` to auto-fix findings.

## Ownership

- ``src/kompend/lint/`` — lint rules, runner, remediation, gap analysis, history

## Architecture

| Module | Contents |
|--------|----------|
| ``base.py`` | ``Severity``, ``LintRuleOutcome``, ``LintResult``, ``BaseLintRule`` ABC |
| ``runner.py`` | ``LintRunner``, ``LintReport`` — orchestrates rule execution |
| ``structural.py`` | ``STRUCTURAL_RULES`` — programmatic rules (image embeds, broken links, etc.); ``broken-wikilinks`` lookup is alias-aware (slug/acronym/normalized forms + fuzzy fallback tagged ``alias_of``) |
| ``llm_rules.py`` | ``LLM_RULE_BY_NAME`` — LLM-powered semantic rules |
| ``remediation.py`` | ``RemediationAutoFixer`` — auto-fix for structural + LLM findings |
| ``gap_analysis.py`` | Gap recommendations from lint reports; broken-wikilink findings stratified into ``alias`` vs ``missing`` resolutions |
| ``_remediation_gaps.py`` | Gap-specific remediation logic; generated pages route through ``MarkdownWriter`` (→ ``concepts/`` + index regen) |
| ``history.py`` | ``LintEntry`` + ``save_lint_entry``/``load_lint_history``/``prune_lint_history`` — lint run history in the ``lint_runs`` table of ``.kompend/kompend.db`` |

## Rules

- **An LLM rule call is long-context and slow.** `lint.llm_timeout` (default
  300 s) governs it, separately from `compile.llm.timeout`. The retry policy is
  not lint's own: `LlmLintRule._call_agent` makes one attempt and the agent base
  retries a transient failure (`compile.llm.transient_retries`). A failed call
  still yields no findings for the rule, so do not swallow the error.
- **The terminal shows a bounded report; the file shows everything.** A
  lint run over a big vault yields hundreds of findings, so the console prints
  the verdict plus (at most) one line per failing rule and points at
  ``.kompend/lint-log/<stamp>.md``. Never print `report_to_text` from a
  pipeline path: it is for `--verbose-lint` and `kompend lint show`.
- Import: ``from kompend.lint.runner import LintRunner``,
  ``from kompend.lint.base import LintResult``.
- Structural rules are fast and run always; LLM rules are gated by config.
- ``LlmLintRule`` takes ``data_format`` — the `compile.llm.data_format` the
  rule's structured prompt payloads are encoded in. `LintRunner` owns the value
  (``data_format=``) and hands it to every LLM rule through its kwargs; a rule
  must encode its payload with ``kompend.llm.format.encode_payload`` rather
  than ``json.dumps`` so the format setting has one source.
- ``RemediationAutoFixer`` calls ``RemediationAgent`` for LLM auto-fixes.
- ``LintRuleAgent`` and ``RemediationAgent`` live in ``kompend.agents.lint``.
- ``history.py`` I/O takes a vault path and resolves ``kompend.db`` via
  ``core.store.vault_db_path`` — it no longer writes ``lint_history.jsonl``.
