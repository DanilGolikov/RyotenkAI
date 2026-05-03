"""Tests for the reward↔strategy compatibility check in
:mod:`src.api.services.config_service`.

The check runs at ``POST /projects/{id}/config/validate`` time and
surfaces per-field errors so the UI can pinpoint the offending
``training.strategies[i].params.reward_plugin`` path.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from src.api.services.config_service import _check_reward_strategy_compat
from src.api.schemas.config_validate import ConfigCheck


def _fake_reward_plugin(plugin_id: str, supported: list[str]) -> MagicMock:
    spec = MagicMock()
    spec.id = plugin_id
    spec.supported_strategies = supported
    entry = MagicMock()
    entry.manifest.plugin = spec
    return entry


@pytest.fixture
def fake_catalog(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the catalog singleton methods so the test doesn't touch
    the real ``community/`` folder. We replace the methods on the
    singleton instance itself (rather than the module-level binding),
    which is the simplest path to a ``CommunityCatalog`` re-export
    gotcha this project has — ``src.community.__init__`` re-exports
    ``catalog`` as an attribute, which shadows the module path for
    ``monkeypatch.setattr`` with a dotted string."""
    from src.community.catalog import catalog as real_catalog

    monkeypatch.setattr(real_catalog, "ensure_loaded", MagicMock())
    fake_plugins = MagicMock()
    monkeypatch.setattr(real_catalog, "plugins", fake_plugins)

    # Return a façade that exposes the same .plugins attribute so tests
    # can set return_value on it.
    facade = MagicMock()
    facade.plugins = fake_plugins
    facade.ensure_loaded = real_catalog.ensure_loaded
    return facade


def _make_cfg(strategy_type: str, reward_plugin: str | None) -> MagicMock:
    """Minimal ducktyped PipelineConfig."""
    strat = MagicMock()
    strat.strategy_type = strategy_type
    strat.params = {"reward_plugin": reward_plugin} if reward_plugin else {}
    training = MagicMock()
    training.strategies = [strat]
    cfg = MagicMock()
    cfg.training = training
    return cfg


def test_compat_ok_for_supported_strategy(fake_catalog: MagicMock) -> None:
    fake_catalog.plugins.return_value = [
        _fake_reward_plugin("helixql_compiler_semantic", ["grpo", "sapo"]),
    ]
    cfg = _make_cfg("grpo", "helixql_compiler_semantic")
    checks: list[ConfigCheck] = []
    field_errors: dict[str, list[str]] = {}

    _check_reward_strategy_compat(cfg, checks, field_errors)

    assert field_errors == {}
    assert any(c.status == "ok" and "compatible" in c.label for c in checks)


def test_compat_fails_for_unsupported_strategy(fake_catalog: MagicMock) -> None:
    fake_catalog.plugins.return_value = [
        _fake_reward_plugin("helixql_compiler_semantic", ["grpo", "sapo"]),
    ]
    cfg = _make_cfg("dpo", "helixql_compiler_semantic")
    checks: list[ConfigCheck] = []
    field_errors: dict[str, list[str]] = {}

    _check_reward_strategy_compat(cfg, checks, field_errors)

    path = "training.strategies.0.params.reward_plugin"
    assert path in field_errors
    assert "supports ['grpo', 'sapo']" in field_errors[path][0]
    assert any(c.status == "fail" for c in checks)


def test_compat_fails_for_missing_plugin(fake_catalog: MagicMock) -> None:
    fake_catalog.plugins.return_value = []  # empty catalog
    cfg = _make_cfg("grpo", "non_existent_reward")
    checks: list[ConfigCheck] = []
    field_errors: dict[str, list[str]] = {}

    _check_reward_strategy_compat(cfg, checks, field_errors)

    path = "training.strategies.0.params.reward_plugin"
    assert path in field_errors
    assert "not in the community catalog" in field_errors[path][0]


def test_compat_noop_when_no_strategies() -> None:
    cfg = MagicMock()
    cfg.training = None
    checks: list[ConfigCheck] = []
    field_errors: dict[str, list[str]] = {}
    _check_reward_strategy_compat(cfg, checks, field_errors)
    assert checks == []
    assert field_errors == {}


def test_compat_noop_when_no_reward_plugin(fake_catalog: MagicMock) -> None:
    # SFT phase without a reward plugin — nothing to check.
    fake_catalog.plugins.return_value = []
    cfg = _make_cfg("sft", None)
    checks: list[ConfigCheck] = []
    field_errors: dict[str, list[str]] = {}
    _check_reward_strategy_compat(cfg, checks, field_errors)
    # Emits "No reward plugins configured" info check.
    assert any("No reward plugins" in c.label for c in checks)
    assert field_errors == {}
