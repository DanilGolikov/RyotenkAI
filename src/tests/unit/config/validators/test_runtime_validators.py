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

from src.config.validators.runtime import validate_eval_plugin_secrets

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
        """Enabled plugin with missing EVAL_* secret → ValueError at init time."""
        cfg = _make_config(eval_enabled=True, plugin_enabled=True)
        secrets = _make_secrets({})  # EVAL_CEREBRAS_API_KEY absent

        with pytest.raises(ValueError, match="cerebras_judge"):
            validate_eval_plugin_secrets(cfg, secrets)

    def test_error_message_contains_plugin_name_and_id(self):
        """ValueError message includes plugin id/name and a hint about secrets.env."""
        cfg = _make_config(eval_enabled=True, plugin_enabled=True)
        secrets = _make_secrets({})

        with pytest.raises(ValueError) as exc_info:
            validate_eval_plugin_secrets(cfg, secrets)

        msg = str(exc_info.value)
        assert "cerebras_judge" in msg
        assert "judge_main" in msg
        assert "secrets.env" in msg
