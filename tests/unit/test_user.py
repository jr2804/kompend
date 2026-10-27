"""Tests for OS user identity resolution (knox-gsb5)."""

from __future__ import annotations

import getpass
import os
from pathlib import Path

import pytest

from kompend.core.config import ResolvedConfig, resolve_config
from kompend.core.keys import TRUST_USER_TRACKING
from kompend.core.trust.actor import get_current_user, resolve_actor


class TestGetCurrentUser:
    def test_returns_username_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(getpass, "getuser", lambda: "alice")
        assert get_current_user() == "alice"

    def test_returns_unknown_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise() -> str:
            raise KeyError("no user")

        monkeypatch.setattr(getpass, "getuser", _raise)
        assert get_current_user() == "unknown"

    def test_returns_unknown_on_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(getpass, "getuser", lambda: "")
        assert get_current_user() == "unknown"


class TestResolveActor:
    def test_returns_user_when_tracking_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("kompend.core.trust.actor.get_current_user", lambda: "alice")
        cfg = ResolvedConfig(raw={"trust": {"user_tracking": True}})
        assert resolve_actor(cfg) == "alice"

    def test_returns_empty_when_tracking_disabled(self) -> None:
        cfg = ResolvedConfig(raw={"trust": {"user_tracking": False}})
        assert resolve_actor(cfg) == ""

    def test_defaults_to_enabled_when_unset(self) -> None:
        # No trust.user_tracking key at all -> default True -> resolves a user
        cfg = ResolvedConfig(raw={})
        assert resolve_actor(cfg) != ""

    def test_env_var_disables_tracking(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Isolate from the real project kompend.toml and clear KOMP_ env vars.
        monkeypatch.chdir(tmp_path)
        for key in list(os.environ):
            if key.startswith("KOMP_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("KOMP_USER_TRACKING", "false")
        cfg = resolve_config()
        assert cfg.get(TRUST_USER_TRACKING) is False
        assert resolve_actor(cfg) == ""
