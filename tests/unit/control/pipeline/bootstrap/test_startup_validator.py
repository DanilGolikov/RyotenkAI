"""Tests for :class:`StartupValidator` — fail-fast startup checks.

Coverage split (positive / negative / boundary / invariants / dep-errors /
regressions / combinatorial) as required by project policy.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ryotenkai_control.pipeline.bootstrap import StartupValidator
from ryotenkai_shared.errors import ProviderAuthFailedError, StrategyChainInvalidError
from ryotenkai_shared.errors import StrategyChainInvalidError


def _mk_config(
    *,
    active_provider: str = "single_node",
    inference_enabled: bool = False,
    inference_provider: str | None = None,
    strategies: list[Any] | None = None,
    raise_provider_exc: bool = False,
) -> MagicMock:
    cfg = MagicMock()
    if raise_provider_exc:
        cfg.get_active_provider_name.side_effect = ValueError("no provider")
    else:
        cfg.get_active_provider_name.return_value = active_provider
    cfg.inference = SimpleNamespace(enabled=inference_enabled, provider=inference_provider)
    cfg.training = SimpleNamespace(strategies=strategies or [])
    return cfg


def _mk_secrets(*, hf_token: str = "hf_tok", runpod_api_key: str | None = None) -> MagicMock:
    s = SimpleNamespace(hf_token=hf_token, runpod_api_key=runpod_api_key)
    return s


def _mk_strategy(name: str = "sft") -> MagicMock:
    s = SimpleNamespace(strategy_type=name)
    return s


# ===========================================================================
# 1. POSITIVE
# ===========================================================================


class TestPositive:
    def test_validate_all_passes_with_valid_config(self) -> None:
        cfg = _mk_config()
        secrets = _mk_secrets()
        with patch(
            "ryotenkai_control.pipeline.bootstrap.startup_validator.validate_eval_plugin_secrets",
            return_value=None,
        ):
            StartupValidator.validate(config=cfg, secrets=secrets)

    def test_hf_token_is_set_in_env(self) -> None:
        secrets = _mk_secrets(hf_token="my_hf_tok")
        StartupValidator.set_hf_token_env(secrets=secrets)
        assert os.environ["HF_TOKEN"] == "my_hf_tok"

    def test_training_provider_runpod_with_api_key(self) -> None:
        cfg = _mk_config(active_provider="runpod")
        secrets = _mk_secrets(runpod_api_key="rp_key")
        StartupValidator.check_training_provider_secrets(config=cfg, secrets=secrets)

    def test_inference_provider_disabled_skips_check(self) -> None:
        cfg = _mk_config(inference_enabled=False, inference_provider="runpod")
        secrets = _mk_secrets(runpod_api_key=None)
        # Should not raise despite missing runpod key — inference is disabled.
        StartupValidator.check_inference_provider_secrets(config=cfg, secrets=secrets)

    def test_strategy_chain_valid(self) -> None:
        strategies = [_mk_strategy("sft"), _mk_strategy("dpo")]
        cfg = _mk_config(strategies=strategies)
        with patch(
            "ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain",
            return_value=None,
        ):
            StartupValidator.check_strategy_chain(config=cfg)


# ===========================================================================
# 2. NEGATIVE
# ===========================================================================


class TestNegative:
    def test_training_provider_runpod_without_api_key_raises(self) -> None:
        cfg = _mk_config(active_provider="runpod")
        secrets = _mk_secrets(runpod_api_key=None)
        with pytest.raises(ProviderAuthFailedError, match="RUNPOD_API_KEY") as ei:
            StartupValidator.check_training_provider_secrets(config=cfg, secrets=secrets)
        # Pin the structured context — clients dispatch on these keys.
        assert ei.value.context["provider"] == "runpod"
        assert ei.value.context["secret_name"] == "RUNPOD_API_KEY"
        assert ei.value.context["role"] == "training"

    def test_inference_provider_runpod_without_api_key_raises(self) -> None:
        cfg = _mk_config(inference_enabled=True, inference_provider="runpod")
        secrets = _mk_secrets(runpod_api_key=None)
        with pytest.raises(ProviderAuthFailedError, match="RUNPOD_API_KEY") as ei:
            StartupValidator.check_inference_provider_secrets(config=cfg, secrets=secrets)
        assert ei.value.context["role"] == "inference"

    def test_invalid_strategy_chain_raises(self) -> None:
        strategies = [_mk_strategy("sft"), _mk_strategy("grpo")]
        cfg = _mk_config(strategies=strategies)
        # check_strategy_chain now propagates StrategyChainInvalidError
        # unchanged — the legacy StartupValidationError wrapper was
        # removed so typed semantics survive to the CLI rendering layer.
        with patch(
            "ryotenkai_control.pipeline.bootstrap.startup_validator.validate_strategy_chain",
            side_effect=StrategyChainInvalidError(detail="incompatible chain"),
        ), pytest.raises(StrategyChainInvalidError, match="incompatible chain"):
            StartupValidator.check_strategy_chain(config=cfg)


# ===========================================================================
# 3. BOUNDARY
# ===========================================================================


class TestBoundary:
    def test_empty_strategy_chain_skips_check(self) -> None:
        cfg = _mk_config(strategies=[])
        # No patch needed — validate_strategy_chain should not be called.
        StartupValidator.check_strategy_chain(config=cfg)

    def test_provider_exception_treated_as_no_provider(self) -> None:
        cfg = _mk_config(raise_provider_exc=True)
        secrets = _mk_secrets(runpod_api_key=None)
        # Should not raise — missing active_provider means no secret needed.
        StartupValidator.check_training_provider_secrets(config=cfg, secrets=secrets)

    def test_inference_without_provider_attr_skips(self) -> None:
        cfg = MagicMock(spec=[])  # no .inference at all
        secrets = _mk_secrets()
        StartupValidator.check_inference_provider_secrets(config=cfg, secrets=secrets)


# ===========================================================================
# 4. INVARIANTS
# ===========================================================================


class TestInvariants:
    def test_validate_fails_fast_on_first_failure(self) -> None:
        """Regression/invariant: the first failing check raises; later
        checks must not then fire with unvalidated input."""
        cfg = _mk_config(active_provider="runpod")
        secrets = _mk_secrets(runpod_api_key=None)
        with patch(
            "ryotenkai_control.pipeline.bootstrap.startup_validator.validate_eval_plugin_secrets"
        ) as mock_eval:
            with pytest.raises(ProviderAuthFailedError):
                StartupValidator.validate(config=cfg, secrets=secrets)
            # Training-provider check failed before we reached eval plugin.
            mock_eval.assert_not_called()

    def test_hf_token_is_str_cast(self) -> None:
        """Invariant: HF_TOKEN env var is always str (MagicMock tolerance)."""
        secrets = SimpleNamespace(hf_token=MagicMock())
        StartupValidator.set_hf_token_env(secrets=secrets)
        assert isinstance(os.environ["HF_TOKEN"], str)


# ===========================================================================
# 5. DEPENDENCY ERRORS
# ===========================================================================


class TestDependencyErrors:
    def test_eval_plugin_validator_exception_propagates(self) -> None:
        cfg = _mk_config()
        secrets = _mk_secrets()
        with patch(
            "ryotenkai_control.pipeline.bootstrap.startup_validator.validate_eval_plugin_secrets",
            side_effect=RuntimeError("missing slack token"),
        ), pytest.raises(RuntimeError, match="missing slack token"):
            StartupValidator.check_eval_plugin_secrets(config=cfg, secrets=secrets)


# ===========================================================================
# 6. REGRESSIONS
# ===========================================================================


class TestRegressions:
    def test_provider_auth_failed_is_ryotenkai_error(self) -> None:
        """Regression (2026-05-16): legacy StartupValidationError(ValueError)
        wrapper was removed because it stripped typed semantics — the
        worker subprocess printed raw Python tracebacks instead of the
        unified kubectl-style CLI render. Validator now raises typed
        ProviderAuthFailedError (HTTP 401) directly so RyotenkAIError
        catch-points (CLI wrap_command, worker top-level handler,
        FastAPI EXCEPTION_HANDLERS) all converge."""
        from ryotenkai_shared.errors import RyotenkAIError

        assert issubclass(ProviderAuthFailedError, RyotenkAIError)
        assert ProviderAuthFailedError(detail="x").status == 401

    def test_runpod_inference_with_runpod_training_and_key_ok(self) -> None:
        """Regression: both provider checks should tolerate the same key."""
        cfg = _mk_config(
            active_provider="runpod", inference_enabled=True, inference_provider="runpod"
        )
        secrets = _mk_secrets(runpod_api_key="rp_key")
        StartupValidator.check_training_provider_secrets(config=cfg, secrets=secrets)
        StartupValidator.check_inference_provider_secrets(config=cfg, secrets=secrets)


# ===========================================================================
# 7. COMBINATORIAL
# ===========================================================================


@pytest.mark.parametrize("provider", ["runpod", "single_node", "custom"])
@pytest.mark.parametrize("has_key", [True, False])
def test_training_provider_matrix(provider: str, has_key: bool) -> None:
    cfg = _mk_config(active_provider=provider)
    secrets = _mk_secrets(runpod_api_key="rp" if has_key else None)
    if provider == "runpod" and not has_key:
        with pytest.raises(ProviderAuthFailedError):
            StartupValidator.check_training_provider_secrets(config=cfg, secrets=secrets)
    else:
        StartupValidator.check_training_provider_secrets(config=cfg, secrets=secrets)


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("provider", ["runpod", None, "other"])
@pytest.mark.parametrize("has_key", [True, False])
def test_inference_provider_matrix(
    enabled: bool, provider: str | None, has_key: bool
) -> None:
    cfg = _mk_config(inference_enabled=enabled, inference_provider=provider)
    secrets = _mk_secrets(runpod_api_key="rp" if has_key else None)
    if enabled and provider == "runpod" and not has_key:
        with pytest.raises(ProviderAuthFailedError):
            StartupValidator.check_inference_provider_secrets(config=cfg, secrets=secrets)
    else:
        StartupValidator.check_inference_provider_secrets(config=cfg, secrets=secrets)
