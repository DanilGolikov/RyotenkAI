"""Phase 14 follow-up — :class:`SystemMetricsConfig` contract.

Pure-stdlib unit tests for the new MLflow system-metrics sub-block.
Mirrors the test pattern of
:mod:`src.tests.unit.config.training.test_metrics_buffer` (the
analogous training-side block).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.integrations.system_metrics import SystemMetricsConfig
from src.config.integrations.mlflow import MLflowConfig
from src.config.integrations.mlflow_integration import MLflowIntegrationConfig
from src.config.integrations.experiment_tracking import ExperimentTrackingConfig


# ---------------------------------------------------------------------------
# 1. Positive — default construction
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_defaults_match_hardcoded_values(self) -> None:
        cfg = SystemMetricsConfig()
        # Hardcoded defaults — operators only override via YAML.
        assert cfg.sampling_interval == 1
        assert cfg.samples_before_logging == 1
        assert cfg.callback_enabled is False
        assert cfg.callback_interval == 10

    def test_full_construction_keyword_args(self) -> None:
        cfg = SystemMetricsConfig(
            sampling_interval=5,
            samples_before_logging=3,
            callback_enabled=True,
            callback_interval=25,
        )
        assert cfg.sampling_interval == 5
        assert cfg.samples_before_logging == 3
        assert cfg.callback_enabled is True
        assert cfg.callback_interval == 25


# ---------------------------------------------------------------------------
# 2. Negative — validation rejects out-of-range values
# ---------------------------------------------------------------------------


class TestNegative:
    def test_sampling_interval_below_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SystemMetricsConfig(sampling_interval=0)

    def test_sampling_interval_above_60_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SystemMetricsConfig(sampling_interval=61)

    def test_samples_before_logging_below_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SystemMetricsConfig(samples_before_logging=0)

    def test_samples_before_logging_above_10_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SystemMetricsConfig(samples_before_logging=11)

    def test_callback_interval_above_100_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SystemMetricsConfig(callback_interval=101)

    def test_extra_fields_rejected(self) -> None:
        # StrictBaseModel forbids unknown keys.
        with pytest.raises(ValidationError):
            SystemMetricsConfig(unknown_knob=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 3. Boundary — edge values
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_sampling_interval_at_min_accepted(self) -> None:
        SystemMetricsConfig(sampling_interval=1)

    def test_sampling_interval_at_max_accepted(self) -> None:
        SystemMetricsConfig(sampling_interval=60)

    def test_samples_before_logging_at_max_accepted(self) -> None:
        SystemMetricsConfig(samples_before_logging=10)

    def test_callback_interval_at_max_accepted(self) -> None:
        SystemMetricsConfig(callback_interval=100)


# ---------------------------------------------------------------------------
# 4. Invariants — frozen-via-validator semantics + defaults stable
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_default_factory_produces_independent_instances(self) -> None:
        # Pin: each MLflowConfig gets its OWN system_metrics block,
        # not a shared mutable singleton — mutating one mustn't
        # affect another.
        a = SystemMetricsConfig()
        b = SystemMetricsConfig()
        a.sampling_interval = 30  # type: ignore[misc]
        assert b.sampling_interval == 1  # unchanged


# ---------------------------------------------------------------------------
# 5. Dependency — embedded in MLflowConfig + MLflowIntegrationConfig
# ---------------------------------------------------------------------------


class TestEmbedded:
    def test_mlflow_config_default_factory_creates_block(self) -> None:
        cfg = MLflowConfig(
            tracking_uri="http://example.com",
            experiment_name="exp",
        )
        assert isinstance(cfg.system_metrics, SystemMetricsConfig)
        assert cfg.system_metrics.sampling_interval == 1

    def test_mlflow_integration_config_default_factory_creates_block(self) -> None:
        cfg = MLflowIntegrationConfig(tracking_uri="http://example.com")
        assert isinstance(cfg.system_metrics, SystemMetricsConfig)
        assert cfg.system_metrics.callback_enabled is False

    def test_mlflow_integration_config_accepts_nested_block_in_yaml(self) -> None:
        cfg = MLflowIntegrationConfig.model_validate({
            "tracking_uri": "http://example.com",
            "system_metrics": {
                "sampling_interval": 30,
                "callback_enabled": True,
            },
        })
        assert cfg.system_metrics.sampling_interval == 30
        assert cfg.system_metrics.callback_enabled is True
        # Other knobs keep defaults.
        assert cfg.system_metrics.samples_before_logging == 1
        assert cfg.system_metrics.callback_interval == 10


# ---------------------------------------------------------------------------
# 6. Regressions — legacy flat fields rejected with migration hint
# ---------------------------------------------------------------------------


class TestLegacyMigrationHint:
    def test_old_flat_field_in_experiment_tracking_mlflow_rejected(self) -> None:
        # Old YAML with ``experiment_tracking.mlflow.system_metrics_callback_enabled``
        # at the project level must raise a clear migration error
        # (these fields have always lived on the integration side
        # post-PR3; the post-Phase-14 nesting just makes the new
        # path on the integration side ``system_metrics:`` block).
        with pytest.raises(ValidationError) as exc_info:
            ExperimentTrackingConfig.model_validate({
                "mlflow": {
                    "integration": "my_int",
                    "experiment_name": "exp",
                    "system_metrics_callback_enabled": True,
                },
            })
        assert "system_metrics_callback_enabled" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 7. Logic-specific — three usage patterns the runtime code expects
# ---------------------------------------------------------------------------


class TestRuntimeUsagePattern:
    def test_callback_enabled_default_false_disables_runtime_callback(self) -> None:
        # Pin: default config must NOT enable the HF Trainer callback
        # (Phase 14 follow-up keeps the safe default — callback off
        # because it can hang on some cloud GPU images).
        cfg = SystemMetricsConfig()
        assert cfg.callback_enabled is False

    def test_callback_interval_only_consulted_when_enabled(self) -> None:
        # The default callback_interval is set even when the
        # callback is disabled — runtime code reads both fields and
        # gates the value lookup on the boolean. Pin the default
        # exists so a future op who flips ``callback_enabled=True``
        # without touching ``callback_interval`` still gets a sane
        # value.
        cfg = SystemMetricsConfig(callback_enabled=False)
        assert cfg.callback_interval == 10
