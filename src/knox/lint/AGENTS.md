# DOX — knox.lint

<!-- on entry: loaded when agent enters src/knox/lint -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

Linting and remediation for compiled wiki pages. Structural rules run
programmatically; LLM-powered rules use ``LintRuleAgent`` for semantic checks.
Remediation uses ``RemediationAgent`` to auto-fix findings.

## Ownership

- ``src/knox/lint/`` — lint rules, runner, remediation, gap analysis, history

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
| ``history.py`` | ``LintHistoryStore`` — persistent lint result history |

## Rules

- Import: ``from knox.lint.runner import LintRunner``,
  ``from knox.lint.base import LintResult``.
- Structural rules are fast and run always; LLM rules are gated by config.
- ``RemediationAutoFixer`` calls ``RemediationAgent`` for LLM auto-fixes.
- ``LintRuleAgent`` and ``RemediationAgent`` live in ``knox.agents.lint``.
