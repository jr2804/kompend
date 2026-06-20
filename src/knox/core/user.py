"""Resolve the current OS user identity for provenance metadata (knox-gsb5).

Records *who* ran an ingest or memory command so shared (network/cloud)
wikis can audit changes. Tracking is opt-out via ``trust.user_tracking``
(see :mod:`knox.core.config`); when disabled, :func:`resolve_actor`
returns an empty string and no identity is recorded.
"""

from __future__ import annotations

import getpass

from knox.core.config import ResolvedConfig
from knox.core.config.keys import TRUST_USER_TRACKING

_UNKNOWN = "unknown"


def get_current_user() -> str:
    """Return the current OS user, or ``"unknown"`` if undeterminable.

    Checks ``$LOGNAME``/``$USER``/``$USERNAME`` then the password database;
    falls back to ``"unknown"`` on any failure.
    """
    try:
        return getpass.getuser() or _UNKNOWN
    except (ImportError, KeyError, OSError):
        # Cannot identify the OS user (no pwd module, user not in db, etc.).
        return _UNKNOWN


def resolve_actor(cfg: ResolvedConfig) -> str:
    """Return the acting user when tracking is enabled, else ``""``.

    Respects ``trust.user_tracking`` (CLI/env/config precedence). An empty
    return value means "do not record identity".
    """
    if not cfg.get(TRUST_USER_TRACKING, True):
        return ""
    return get_current_user()
