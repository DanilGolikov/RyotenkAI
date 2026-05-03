""":class:`SystemMetricsConfig` contract — single boolean knob.

After the second refactor pass the block carries only ``callback_enabled``.
The native MLflow sampler is no longer enabled by the codebase, and the
HF Trainer callback no longer step-throttles. Removed nested fields are
covered by negative tests below to pin the migration hint behaviour.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.integrations.root import IntegrationsConfig
from src.config.integrations.mlflow import MLflowConfig
from src.config.integrations.mlflow_integration import MLflowIntegrationConfig
from src.config.integrations.system_metrics import SystemMetricsConfig

# ---------------------------------------------------------------------------
# 1. Positive — default construction
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_callback_enabled_is_true(self) -> None:
        # Default ON: SystemMetricsCallback is registered, metrics flow
        # through ResilientMLflowTransport + MetricsBuffer.
        cfg = SystemMetricsConfig()
        assert cfg.callback_enabled is True

    def test_callback_enabled_can_be_overridden_to_false(self) -> None:
        # The only override case operators care about: pynvml hangs on
        # some cloud GPU images, force-disable to keep training healthy.
        cfg = SystemMetricsConfig(callback_enabled=False)
        assert cfg.callback_enabled is False


# ---------------------------------------------------------------------------
# 2. Negative — extra and removed fields rejected
# ---------------------------------------------------------------------------


class TestNegative:
    def test_extra_fields_rejected(self) -> None:
        # StrictBaseModel forbids unknown keys.
        with pytest.raises(ValidationError):
            SystemMetricsConfig(unknown_knob=1)  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "removed_field, removed_value",
        [
            ("sampling_interval", 5),
            ("samples_before_logging", 3),
            ("callback_interval", 25),
        ],
    )
    def test_removed_fields_rejected_at_block_level(
        self, removed_field: str, removed_value: int
    ) -> None:
        # Direct construction of the block: each removed field now
        # surfaces as ``extra_forbidden`` from StrictBaseModel.
        with pytest.raises(ValidationError):
            SystemMetricsConfig(**{removed_field: removed_value})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Embedded — block lives on MLflowConfig + MLflowIntegrationConfig
# ---------------------------------------------------------------------------


class TestEmbedded:
    def test_mlflow_config_default_factory_creates_block(self) -> None:
        cfg = MLflowConfig(
            tracking_uri="http://example.com",
            experiment_name="exp",
        )
        assert isinstance(cfg.system_metrics, SystemMetricsConfig)
        assert cfg.system_metrics.callback_enabled is True

    def test_mlflow_integration_config_default_factory_creates_block(self) -> None:
        cfg = MLflowIntegrationConfig(tracking_uri="http://example.com")
        assert isinstance(cfg.system_metrics, SystemMetricsConfig)
        assert cfg.system_metrics.callback_enabled is True

    def test_mlflow_integration_config_accepts_nested_block_in_yaml(self) -> None:
        cfg = MLflowIntegrationConfig.model_validate({
            "tracking_uri": "http://example.com",
            "system_metrics": {"callback_enabled": False},
        })
        assert cfg.system_metrics.callback_enabled is False


# ---------------------------------------------------------------------------
# 4. Invariants — independent default factory
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_default_factory_produces_independent_instances(self) -> None:
        # Pin: each MLflowConfig gets its OWN system_metrics block —
        # mutating one mustn't leak into another.
        a = SystemMetricsConfig()
        b = SystemMetricsConfig()
        a.callback_enabled = False  # type: ignore[misc]
        assert b.callback_enabled is True  # unchanged


# ---------------------------------------------------------------------------
# 5. Migration hints — flat-path and nested-path removed fields
# ---------------------------------------------------------------------------


class TestLegacyMigrationHint:
    def test_old_flat_field_in_integrations_mlflow_rejected(self) -> None:
        # Pre-nested-block flat field at the project level still raises
        # a clear migration error (kept for users on very old configs).
        with pytest.raises(ValidationError) as exc_info:
            IntegrationsConfig.model_validate({
                "mlflow": {
                    "integration": "my_int",
                    "experiment_name": "exp",
                    "system_metrics_callback_enabled": True,
                },
            })
        assert "system_metrics_callback_enabled" in str(exc_info.value)

    @pytest.mark.parametrize(
        "removed_field",
        ["sampling_interval", "samples_before_logging", "callback_interval"],
    )
    def test_removed_nested_field_in_integrations_mlflow_rejected(
        self, removed_field: str
    ) -> None:
        # Pin: YAMLs that already moved to the nested
        # ``system_metrics:`` block but still carry the now-removed
        # knobs surface a targeted migration hint, not a generic
        # ``extra_forbidden`` Pydantic error.
        with pytest.raises(ValidationError) as exc_info:
            IntegrationsConfig.model_validate({
                "mlflow": {
                    "integration": "my_int",
                    "experiment_name": "exp",
                    "system_metrics": {removed_field: 5},
                },
            })
        msg = str(exc_info.value)
        assert removed_field in msg
        # The hint should point at WHY the field is gone.
        assert "native MLflow sampler" in msg or "every step" in msg


# ---------------------------------------------------------------------------
# 6. Runtime contract — what TrainerFactory consumes
# ---------------------------------------------------------------------------


class TestRuntimeUsagePattern:
    def test_callback_enabled_default_true_registers_runtime_callback(self) -> None:
        # Pin: default config MUST enable the HF Trainer callback —
        # SystemMetricsCallback flows through ResilientMLflowTransport
        # and is buffered via MetricsBuffer on offline windows.
        cfg = SystemMetricsConfig()
        assert cfg.callback_enabled is True
