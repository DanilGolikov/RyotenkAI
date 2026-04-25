"""Tests for the runtime helpers on :class:`BasePlugin` (PR9 / B2).

The helpers (``_env`` / ``_secret``) are the contract plugin authors
should use instead of poking ``os.environ`` or ``self._secrets``
directly. Centralising the access point gives us a place to add
validation, telemetry, and per-test overrides without touching every
plugin folder.
"""

from __future__ import annotations

import pytest

from src.utils.plugin_base import BasePlugin


class _Plugin(BasePlugin):
    """Minimal BasePlugin subclass for tests — no kind-specific contract."""


# ---------------------------------------------------------------------------
# _env
# ---------------------------------------------------------------------------


def test_env_returns_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MY_VAR", raising=False)
    p = _Plugin()
    assert p._env("MY_VAR") is None
    assert p._env("MY_VAR", default="fallback") == "fallback"


def test_env_reads_from_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_VAR", "from-process")
    p = _Plugin()
    assert p._env("MY_VAR") == "from-process"


def test_env_injected_dict_wins_over_process_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_injected_env`` (set by the registry from the project's env.json
    plus test overrides) wins so tests get deterministic values without
    monkey-patching the global ``os.environ``."""
    monkeypatch.setenv("MY_VAR", "from-process")
    p = _Plugin()
    object.__setattr__(p, "_injected_env", {"MY_VAR": "from-injected"})
    assert p._env("MY_VAR") == "from-injected"


def test_env_falls_back_to_process_when_injected_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin instance with an empty ``_injected_env`` dict still sees
    the process env — useful in tests that init the plugin off the
    registry path."""
    monkeypatch.setenv("MY_VAR", "from-process")
    p = _Plugin()
    object.__setattr__(p, "_injected_env", {})
    assert p._env("MY_VAR") == "from-process"


def test_env_treats_empty_string_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators routinely "blank out" env vars with ``MY_VAR=``; ``_env``
    treats that as unset rather than handing the plugin an empty
    string. The default kicks in instead."""
    monkeypatch.setenv("MY_VAR", "")
    p = _Plugin()
    assert p._env("MY_VAR", default="fallback") == "fallback"


# ---------------------------------------------------------------------------
# _secret
# ---------------------------------------------------------------------------


def test_secret_raises_when_no_secrets_resolved() -> None:
    """A plugin that never received an injection raises with a hint
    pointing at the manifest's ``[[required_env]]`` block."""
    p = _Plugin()
    with pytest.raises(KeyError, match=r"\[\[required_env\]\]"):
        p._secret("EVAL_SOMETHING")


def test_secret_returns_resolved_value() -> None:
    p = _Plugin()
    object.__setattr__(p, "_secrets", {"EVAL_KEY": "abc123"})
    assert p._secret("EVAL_KEY") == "abc123"


def test_secret_raises_when_key_missing_from_resolved_set() -> None:
    """Asking for a key the resolver didn't fetch is a programming error
    in the manifest — we surface a clear message instead of falling back
    to ``os.environ`` (which would mask the contract violation)."""
    p = _Plugin()
    object.__setattr__(p, "_secrets", {"EVAL_KEY": "abc"})
    with pytest.raises(KeyError, match=r"EVAL_OTHER.*not in resolved set"):
        p._secret("EVAL_OTHER")
