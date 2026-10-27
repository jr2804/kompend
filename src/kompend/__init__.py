"""kompend package."""

from __future__ import annotations

import importlib
import os

# pydantic imports every entry point in the ``pydantic`` plugin group the first
# time a BaseModel subclass is defined. logfire-sdk registers one, so a bare
# ``import kompend`` used to pull in logfire (~700 ms, plus ``requests`` and the
# whole OTel SDK) before any kompend code ran - and that is why
# ``kompend --version`` and ``--help`` paid for telemetry. kompend instruments
# pydantic-ai explicitly (kompend.telemetry, via the CLI callback), so the
# plugin is not needed. ``setdefault`` keeps an explicit user setting, and an
# empty value (``PYDANTIC_DISABLE_PLUGINS=""``) re-enables the plugin.
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "logfire-plugin")


def __getattr__(name: str) -> str:
    """Resolve ``kompend.__version__`` on first use.

    ``importlib.metadata`` pulls in ``email``, ``zipfile`` and friends (~200 ms
    of every interpreter start that imports kompend). Nothing on the CLI path
    reads the attribute, so it is resolved lazily instead of at import time.
    """
    if name != "__version__":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    metadata = importlib.import_module("importlib.metadata")
    try:
        return metadata.version(__name__)
    except metadata.PackageNotFoundError:
        return "0.0.0"
