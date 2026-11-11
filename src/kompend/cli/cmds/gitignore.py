"""``kompend gitignore`` — render the vault's managed ``.gitignore`` block.

The command exists because "which artifacts may I commit?" has one answer that
must not drift: the template is rendered from the artifact-name constants, so a
renamed cache file or a new index never leaves a stale rule behind. It replaces
only the managed block — every rule the user wrote around it survives.
"""

from __future__ import annotations

from kompend.cli.args.gitignore import GitignoreCheck, GitignorePrint
from kompend.cli.settings import build_cli_overrides, vault_paths_from_ctx
from kompend.core.config import resolve_config
from kompend.core.keys import PATHS_OUTPUT_DIR, PATHS_SOURCES_DIR
from kompend.core.paths import GITIGNORE_FILE
from kompend.logging import get_console, get_err_console
from kompend.scaffolding import dir_name, is_current, merge_gitignore, render_gitignore


def gitignore_cmd(
    *,
    check: GitignoreCheck = False,
    print_rendered: GitignorePrint = False,
) -> None:
    """Write or check the managed `.gitignore` block for the current vault.

    Without flags the vault's `.gitignore` is created or updated in place: the
    kompend-managed block is replaced, the user's own rules are kept.
    `--check` reports drift without writing (exit 1 when the block is missing or
    outdated), and `--print` writes the rendered file to stdout instead.
    """
    vp = vault_paths_from_ctx()
    config = resolve_config(base_dir=vp.base_dir, cli_overrides=build_cli_overrides())
    rendered = render_gitignore(
        sources_dir=dir_name(config.get(PATHS_SOURCES_DIR), fallback=vp.sources_dir_name),
        output_dir=dir_name(config.get(PATHS_OUTPUT_DIR), fallback=vp.output_dir_name),
    )
    path = vp.base_dir / GITIGNORE_FILE

    if print_rendered:
        get_console().print(rendered, end="")
        return

    existing = path.read_text(encoding="utf-8") if path.is_file() else None

    if check:
        if existing is None:
            get_err_console().print(f"error: {path} does not exist. Run 'kompend gitignore' to create it.")
            raise SystemExit(1)
        if not is_current(existing, rendered):
            get_err_console().print(f"error: the kompend block in {path} is missing or outdated. Run 'kompend gitignore' to refresh it.")
            raise SystemExit(1)
        get_console().print(f"ok: {path} matches the current artifact layout")
        return

    text = rendered if existing is None else merge_gitignore(existing, rendered)
    if existing == text:
        get_console().print(f"ok: {path} is already up to date")
        return
    path.write_text(text, encoding="utf-8")
    get_console().print(f"{'updated' if existing is not None else 'created'}: {path}")
