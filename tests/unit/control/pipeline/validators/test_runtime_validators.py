"""
Unit tests for src/config/validators/runtime.py.

Tests:
- test_passes_when_evaluation_disabled   — eval.enabled=False → no validation
- test_passes_when_plugin_disabled       — enabled plugin skipped
- test_passes_when_secret_present        — correct EVAL_* key → no error
- test_raises_when_secret_missing        — missing key → ValueError at init time
- test_error_message_contains_context    — error includes plugin id/name and hints
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ryotenkai_control.pipeline.validators.runtime import validate_eval_plugin_secrets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(*, eval_enabled: bool, plugin_enabled: bool = True) -> MagicMock:
    """Build a minimal mock PipelineConfig with cerebras_judge configured."""
    cfg = MagicMock()
    eval_cfg = MagicMock()
    eval_cfg.enabled = eval_enabled

    plugin_cfg = MagicMock()
    plugin_cfg.id = "judge_main"
    plugin_cfg.plugin = "cerebras_judge"
    plugin_cfg.enabled = plugin_enabled

    eval_cfg.evaluators.plugins = [plugin_cfg]
    cfg.evaluation = eval_cfg
    return cfg


def _make_secrets(extra: dict) -> MagicMock:
    s = MagicMock()
    s.model_extra = extra
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_community_catalog_for_eval_plugin_tests() -> None:
    """Other suites may call ``catalog.reload()`` which re-imports plugin
    classes under synthetic module names — including ``cerebras_judge`` with
    a fresh ``_required_secrets`` attribute. When the process has already
    loaded the catalog the ``ensure_loaded()`` path used by
    ``validate_eval_plugin_secrets`` becomes a no-op, so ask for an explicit
    reload here to make sure our lookup resolves to the up-to-date class.
    """
    from ryotenkai_community.catalog import catalog

    catalog.reload()


class TestValidateEvalPluginSecrets:
    def test_passes_when_evaluation_disabled(self):
        """No validation occurs when evaluation.enabled=False."""
        cfg = _make_config(eval_enabled=False)
        secrets = _make_secrets({})
        validate_eval_plugin_secrets(cfg, secrets)  # must not raise

    def test_passes_when_plugin_disabled(self):
        """Disabled plugin is skipped even if its secret is missing."""
        cfg = _make_config(eval_enabled=True, plugin_enabled=False)
        secrets = _make_secrets({})
        validate_eval_plugin_secrets(cfg, secrets)  # must not raise

    def test_passes_when_secret_present(self):
        """Enabled plugin with its EVAL_* key present → no error."""
        cfg = _make_config(eval_enabled=True, plugin_enabled=True)
        secrets = _make_secrets({"eval_cerebras_api_key": "csk-real-key"})
        validate_eval_plugin_secrets(cfg, secrets)  # must not raise

    def test_raises_when_secret_missing(self):
        """Enabled plugin with missing EVAL_* secret → ProviderAuthFailedError at init time."""
        from ryotenkai_shared.errors import ProviderAuthFailedError

        cfg = _make_config(eval_enabled=True, plugin_enabled=True)
        secrets = _make_secrets({})  # EVAL_CEREBRAS_API_KEY absent

        with pytest.raises(ProviderAuthFailedError, match="cerebras_judge"):
            validate_eval_plugin_secrets(cfg, secrets)

    def test_error_message_contains_plugin_name_and_id(self):
        """Typed error carries plugin id/name in detail + structured context."""
        from ryotenkai_shared.errors import ProviderAuthFailedError

        cfg = _make_config(eval_enabled=True, plugin_enabled=True)
        secrets = _make_secrets({})

        with pytest.raises(ProviderAuthFailedError) as exc_info:
            validate_eval_plugin_secrets(cfg, secrets)

        # User-facing detail still carries the hint for stdout/log.
        msg = str(exc_info.value)
        assert "cerebras_judge" in msg
        assert "judge_main" in msg
        assert "secrets.env" in msg
        # Structured context — clients dispatch on these keys, not on regex.
        ctx = exc_info.value.context
        assert ctx["plugin_name"] == "cerebras_judge"
        assert ctx["plugin_id"] == "judge_main"
        assert ctx["role"] == "evaluation"
        assert "missing_secrets" in ctx
