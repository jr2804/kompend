"""Format the generated markdown with rumdl."""

from __future__ import annotations

from pathlib import Path

from kompend.cli.args import LintFormatCheck
from kompend.cli.settings import build_cli_overrides, config_file_from_ctx, vault_paths_from_ctx
from kompend.core.config import resolve_config
from kompend.core.config.rumdl import resolve_rumdl_config
from kompend.core.paths import INDEX_FILE, OVERVIEW_MD
from kompend.logging import get_console, get_err_console
from kompend.output.renderer import format_output

__all__ = ["lint_format"]


def lint_format(
    *,
    check: LintFormatCheck = False,
) -> None:
    """Run ``rumdl fmt`` over the compiled output (bundle + root index).

    Formats every generated ``.md`` file with the vault's rumdl config
    (``compile.output.rumdl_config``, default ``.kompend/rumdl.toml``). Works
    without a compiled-wiki cache — the files on disk are the source of
    truth. ``--check`` exits 1 if any formatting changes are needed (CI).

    Examples:
        kompend lint format                 # format generated markdown
        kompend lint format --check         # CI: exit 1 if changes needed
    """
    vp = vault_paths_from_ctx()
    cfg = resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(),
        config_file=config_file_from_ctx(),
    )
    rumdl_cfg = resolve_rumdl_config(vp.base_dir, cfg.kompend.compile.output.rumdl_config)
    extra_files: list[Path] = []
    for name in (INDEX_FILE, OVERVIEW_MD):
        p = vp.base_dir / name
        if p.exists():
            extra_files.append(p)
    changed = format_output(
        vp.output_dir,
        flavor=cfg.kompend.compile.output.flavor,
        rumdl_config=rumdl_cfg,
        extra_files=extra_files or None,
        check_only=check,
    )
    if check:
        if changed:
            get_err_console().print(
                f"{len(changed)} file(s) need formatting. Run 'kompend lint format' to fix.",
            )
            raise SystemExit(1)
        get_console().print("All generated markdown is formatted.")
        return
    if changed:
        get_console().print(f"Formatted {len(changed)} file(s) with rumdl.")
    else:
        get_console().print("No formatting changes needed.")
