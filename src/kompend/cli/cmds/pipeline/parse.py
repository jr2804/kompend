"""``kompend parse`` — show what the parser makes of a source."""

from __future__ import annotations

from kompend.cli.args import ParseFiles, PrintElements, RecursiveScan
from kompend.cli.settings import build_cli_overrides, vault_paths_from_ctx
from kompend.cli.sources.resolve import resolve_files
from kompend.core.config import resolve_config
from kompend.logging import get_console, get_err_console
from kompend.preprocessing.host_memory import InsufficientHostMemoryError
from kompend.preprocessing.parsers import BinarySourceError, ParserOptions, build_parser


def parse_cmd(
    files: ParseFiles,
    *,
    recursive: RecursiveScan = None,
    print_elements: PrintElements = False,
) -> None:
    """Parse documents and show their structure.

    Accepts one or more file paths, glob patterns (e.g. *.pdf), or directories.
    """
    vp = vault_paths_from_ctx(require_vault=False)
    cfg = resolve_config(
        base_dir=vp.base_dir,
        cli_overrides=build_cli_overrides(recursive=recursive),
    )
    resolved = resolve_files(
        *files,
        recursive=cfg.kompend.paths.recursive,
        ignore_patterns=None,
    )
    parser = build_parser(ParserOptions.from_config(cfg, base_dir=vp.base_dir))
    for path in resolved:
        try:
            doc = parser.parse(path)
        except BinarySourceError as exc:
            # The file's extension claims text but its bytes are binary —
            # say that instead of the generic no-parser wording.
            get_err_console().print(f"Mislabelled source: {exc}")
            continue
        except InsufficientHostMemoryError as exc:
            # The router planned a convert this host cannot back. It is not a
            # parse failure: nothing ran, and the message names the levers.
            get_err_console().print(f"Not attempted: {exc}")
            continue
        except ValueError as exc:
            # No backend claims the file, or every backend failed.
            get_err_console().print(f"No parser for {path}: {exc}")
            continue
        get_console().print(f"{path.name}: {len(doc.elements)} element(s)")
        if print_elements:
            element_text_limit = cfg.kompend.display.cli_element_chars
            for el in doc.elements:
                content = el.content if isinstance(el.content, str) else repr(el.content)
                get_console().print(f"  - {el.type}: {content[:element_text_limit]}")
