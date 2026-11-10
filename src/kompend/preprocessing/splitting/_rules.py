"""Document-category rules the split judge reasons over.

A rule is what distinguishes an amendment from a continuation of a contract, so
the judge cannot work without them. They are a resource, not a Python literal:
the shipped list lives in ``templates/splitting/rules.yaml`` and a vault can
override it by pointing ``splitting.rules_file`` at its own file with the same
shape.

The file is validated strictly — an unknown key, a duplicate id, a missing
description or a non-slug id is an error, not a warning. A rules file that is
half understood would ask the judge an ambiguous question, and the judge's
answer decides whether a real document is cut.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from kompend.core.templates import get_template_loader
from kompend.preprocessing.splitting._plan import SplitRule

#: Built-in rules, resolved through the layered template loader so a project or
#: user ``kompend/templates/splitting/rules.yaml`` overrides it the same way a
#: prompt template is overridden.
BUILTIN_RULES_TEMPLATE: str = "splitting/rules.yaml"

#: A category id is a filing word: lowercase, dash-joined, no spaces.
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")

#: The only keys a rules file may carry, at either level.
_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"categories"})
_CATEGORY_KEYS: frozenset[str] = frozenset({"id", "description"})


class SplitRulesError(ValueError):
    """A rules file that does not describe usable categories."""


def load_rules(path: Path | None = None) -> tuple[SplitRule, ...]:
    """Load the category rules.

    *path* is an explicit file; ``None`` loads the built-in list through the
    layered template loader (project override wins).
    """
    if path is None:
        source = get_template_loader().get_text(BUILTIN_RULES_TEMPLATE)
        origin = f"built-in {BUILTIN_RULES_TEMPLATE}"
    else:
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"Cannot read splitting rules from {path}: {exc}"
            raise SplitRulesError(msg) from exc
        origin = str(path)
    try:
        parsed = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        msg = f"Splitting rules in {origin} are not valid YAML: {exc}"
        raise SplitRulesError(msg) from exc
    return parse_rules(parsed, origin=origin)


def parse_rules(parsed: Any, *, origin: str = "rules") -> tuple[SplitRule, ...]:
    """Validate a loaded rules document and return its rules in file order."""
    if not isinstance(parsed, Mapping):
        msg = f"Splitting rules in {origin} must be a mapping with a 'categories' list"
        raise SplitRulesError(msg)
    unknown = set(parsed) - _TOP_LEVEL_KEYS
    if unknown:
        msg = f"Splitting rules in {origin} carry unknown key(s): {', '.join(sorted(map(str, unknown)))}"
        raise SplitRulesError(msg)
    categories = parsed.get("categories")
    if not isinstance(categories, list) or not categories:
        msg = f"Splitting rules in {origin} need a non-empty 'categories' list"
        raise SplitRulesError(msg)
    rules: list[SplitRule] = []
    seen: set[str] = set()
    for index, entry in enumerate(categories):
        where = f"{origin} categories[{index}]"
        if not isinstance(entry, Mapping):
            msg = f"{where} must be a mapping with 'id' and 'description'"
            raise SplitRulesError(msg)
        unknown = set(entry) - _CATEGORY_KEYS
        if unknown:
            msg = f"{where} carries unknown key(s): {', '.join(sorted(map(str, unknown)))}"
            raise SplitRulesError(msg)
        rule_id = entry.get("id")
        description = entry.get("description")
        if not isinstance(rule_id, str) or not _ID_PATTERN.match(rule_id):
            msg = f"{where} needs an 'id' of lowercase letters, digits and dashes, starting with a letter; got {rule_id!r}"
            raise SplitRulesError(msg)
        if rule_id in seen:
            msg = f"{where} repeats the id {rule_id!r}"
            raise SplitRulesError(msg)
        if not isinstance(description, str) or not description.strip():
            msg = f"{where} ({rule_id}) needs a non-empty 'description'"
            raise SplitRulesError(msg)
        seen.add(rule_id)
        rules.append(SplitRule(id=rule_id, description=description.strip()))
    return tuple(rules)


__all__ = ["BUILTIN_RULES_TEMPLATE", "SplitRulesError", "load_rules", "parse_rules"]
