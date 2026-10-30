#!/usr/bin/env python3
"""Install dev-feature tools, MCP servers, and skills from the CSV manifests.

The CSVs in ``.config/mise/data/`` are the single source of truth: edit them
(flip ``enabled`` from 0 to 1, add a row) and re-run the mise task. This script
is what ``mise run add-mcp-servers`` and ``mise run add-skills`` invoke.

Rows with ``enabled=1`` are installed/registered; rows with ``enabled=0`` are
dormant and actively removed (best-effort — removals of never-installed
items do not fail the run). Only installation failures affect the exit code.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True, slots=True)
class Command:
    """Immutable CLI invocation; chain .arg()/.opt() to build, .run() to execute."""

    argv: tuple[str, ...] = ()

    def arg(self, *values: str) -> Command:
        """Append positional arguments."""
        return Command(self.argv + values)

    def opt(self, flag: str, *values: str) -> Command:
        """Append a flag together with its values (e.g. ``.opt("-n", "name")``)."""
        return Command(self.argv + (flag, *values))

    def run(self) -> bool:
        """Echo and execute the command; return True on exit code 0."""
        print(f"$ {shlex.join(self.argv)}", flush=True)
        return subprocess.run(self.argv, check=False).returncode == 0


@dataclass(frozen=True, slots=True)
class SkillRef:
    """A skill installed via ``bun x skills add``."""

    repo: str
    name: str = ""
    enabled: bool = True

    def install_command(self) -> Command:
        """Build the install command; ``name`` selects one skill via ``-s``."""
        cmd = Command(("bun", "x", "skills", "add", self.repo))
        if self.name:
            cmd = cmd.opt("-s", self.name)
        return cmd.opt("-a", "universal", "-y")

    def remove_command(self) -> Command:
        """Build the best-effort removal command for a dormant skill."""
        cmd = Command(("bun", "x", "skills", "remove", self.repo))
        if self.name:
            cmd = cmd.opt("-s", self.name)
        return cmd.opt("-y")


@dataclass(frozen=True, slots=True)
class DevFeature:
    """One row of ``dev-features.csv``: an optional tool, MCP server, and skill."""

    category: str = ""
    tool: str = ""
    tool_value: str = ""
    mcp_command: str = ""
    mcp_name: str = ""
    skill_repo: str = ""
    skill: str = ""
    enabled: bool = True

    @classmethod
    def from_row(cls, row: dict[str, str]) -> DevFeature:
        """Build a feature from a CSV row, ignoring blank cells."""
        strings = {field: str(row.get(field) or "").strip() for field in cls.__dataclass_fields__ if field != "enabled"}
        return cls(enabled=row.get("enabled", "1").strip() != "0", **strings)

    @property
    def tool_table(self) -> dict[str, object] | None:
        """Parse an inline TOML table in ``tool_value`` (mise tool options)."""
        value = self.tool_value
        if not value.startswith("{"):
            return None
        try:
            return tomllib.loads(f"v = {value}")["v"]
        except (tomllib.TOMLDecodeError, KeyError):
            return None

    @property
    def mise_spec(self) -> str | None:
        """mise use spec honoring ``tool_value``.

        Empty or ``latest`` -> ``tool@latest``; an inline TOML table -> bare
        ``tool`` (its ``version``/``extras``/``uvx_args`` become
        ``--tool-option`` flags); anything else -> ``tool@<value>``.
        """
        if not self.tool:
            return None
        value = self.tool_value
        if value.startswith("{"):
            return self.tool if self.tool_table is not None else f"{self.tool}@latest"
        return f"{self.tool}@{value or 'latest'}"

    def mise_use_command(self, mise_env: str | None = None) -> Command | None:
        """Build ``mise use``; inline-table values expand to ``--tool-option``."""
        spec = self.mise_spec
        if spec is None:
            return None
        cmd = Command(("mise", "use"))
        for key, value in (self.tool_table or {}).items():
            cmd = cmd.opt("--tool-option", f"{key}={_inline_value(value)}")
        cmd = cmd.arg(spec)
        if mise_env:
            cmd = cmd.opt("--env", mise_env)
        return cmd

    @property
    def skill_ref(self) -> SkillRef | None:
        """The tool-linked skill, if the row declares one."""
        return SkillRef(self.skill_repo, self.skill, self.enabled) if self.skill_repo else None


def _inline_value(value: object) -> str:
    """Serialize a TOML value for ``mise use --tool-option``."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return f"[{', '.join(_inline_value(item) for item in value)}]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key} = {_inline_value(item)}" for key, item in value.items()) + "}"
    return json.dumps(str(value))


def _manifest_rows(name: str) -> Iterator[dict[str, str]]:
    """Yield all manifest rows (a missing ``enabled`` column means enabled)."""
    path = DATA / name
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            yield {key: value or "" for key, value in raw.items()}


def _features() -> list[DevFeature]:
    """All dev-feature rows, enabled and dormant."""
    return [DevFeature.from_row(row) for row in _manifest_rows("dev-features.csv")]


def load_features() -> list[DevFeature]:
    """Enabled dev-features (tools / MCP servers / tool-linked skills)."""
    return [feature for feature in _features() if feature.enabled]


def _skill_refs() -> list[SkillRef]:
    """All skills — tool-linked (dev-features.csv) plus community (skills.csv)."""
    refs = [ref for feature in _features() if (ref := feature.skill_ref)]
    refs += [
        SkillRef(row.get("repo", "").strip(), row.get("skill", "").strip(), row.get("enabled", "1").strip() != "0")
        for row in _manifest_rows("skills.csv")
        if row.get("repo", "").strip()
    ]
    return refs


def load_skills() -> list[SkillRef]:
    """Enabled skills: tool-linked (dev-features.csv) plus community (skills.csv)."""
    return [ref for ref in _skill_refs() if ref.enabled]


def add_mcp_servers(agent: str, mise_env: str | None = None) -> int:
    """Install enabled tools/MCP servers; best-effort remove dormant ones."""
    failures = 0
    for feature in _features():
        if not feature.enabled:
            if feature.tool:
                Command(("mise", "remove", "--env", "local", feature.tool)).run()
            if feature.mcp_name:
                Command(("bun", "x", "add-mcp", "remove", feature.mcp_name)).opt("-a", agent).opt("-y").run()
            continue
        if cmd := feature.mise_use_command(mise_env):
            failures += not cmd.run()
        if feature.mcp_command and feature.mcp_name:
            failures += not Command(("bun", "x", "add-mcp", feature.mcp_command)).opt("-y").opt("-a", agent).opt("-n", feature.mcp_name).run()
    return failures


def add_skills() -> int:
    """Install enabled skills; best-effort remove dormant ones."""
    failures = 0
    for ref in _skill_refs():
        if ref.enabled:
            failures += not ref.install_command().run()
        else:
            ref.remove_command().run()
    return failures


def main(argv: list[str] | None = None) -> int:
    """Dispatch a subcommand; return a nonzero exit code if anything failed."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    mcp = sub.add_parser("add-mcp-servers", help="install dev-feature tools and register MCP servers")
    mcp.add_argument("ai_agent", help="target assistant: claude, gemini, copilot, opencode, ...")
    sub.add_parser("add-skills", help="install agent skills")
    args = parser.parse_args(argv)

    failures = add_mcp_servers(args.ai_agent, mise_env="local") if args.command == "add-mcp-servers" else add_skills()
    if failures:
        print(f"{failures} command(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
