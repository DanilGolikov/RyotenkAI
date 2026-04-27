"""Phase 12.A.2 — :class:`MetricsBufferConfig` Pydantic schema contract.

Pin the structure that ships in user YAML so that:
* default factory yields ``keep_all=True`` (lossless — user mandate).
* every numeric knob is positive (``ge=1``).
* unknown fields are rejected (``StrictBaseModel`` extra=forbid).
* the legacy hard-coded Phase 9 thresholds (1 / 2 / 5 over 10 / 30 / late)
  remain the **defaults** so existing operators flipping
  ``keep_all=false`` don't suddenly see different behaviour.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.training import (
    DecimationWindowConfig,
    MetricsBufferConfig,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Positive — defaults match user mandate
# ---------------------------------------------------------------------------


class TestPositiveDefaults:
    def test_default_factory_is_lossless(self) -> None:
        cfg = MetricsBufferConfig()
        assert cfg.keep_all is True

    def test_default_decimation_matches_phase_9_baseline(self) -> None:
        # Pin Phase 9 hard-coded values as the *new* defaults so flipping
        # keep_all=false reproduces the legacy behaviour.
        d = DecimationWindowConfig()
        assert d.window_first_minutes == 10
        assert d.window_first_keep_every == 1
        assert d.window_mid_minutes == 30
        assert d.window_mid_keep_every == 2
        assert d.window_late_keep_every == 5

    def test_keep_all_false_with_explicit_decimation(self) -> None:
        cfg = MetricsBufferConfig(
            keep_all=False,
            decimation=DecimationWindowConfig(
                window_first_minutes=5,
                window_first_keep_every=1,
                window_mid_minutes=20,
                window_mid_keep_every=4,
                window_late_keep_every=10,
            ),
        )
        assert cfg.keep_all is False
        assert cfg.decimation.window_first_minutes == 5
        assert cfg.decimation.window_late_keep_every == 10


# ---------------------------------------------------------------------------
# 2. Negative — invalid values rejected
# ---------------------------------------------------------------------------


class TestNegative:
    def test_zero_minutes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DecimationWindowConfig(window_first_minutes=0)

    def test_negative_minutes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DecimationWindowConfig(window_mid_minutes=-1)

    def test_zero_keep_every_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DecimationWindowConfig(window_first_keep_every=0)

    def test_extra_field_rejected_on_metrics_buffer(self) -> None:
        with pytest.raises(ValidationError):
            MetricsBufferConfig.model_validate(
                {"keep_all": False, "unknown": 42},
            )

    def test_extra_field_rejected_on_decimation(self) -> None:
        with pytest.raises(ValidationError):
            DecimationWindowConfig.model_validate(
                {"window_first_minutes": 10, "extra_knob": True},
            )


# ---------------------------------------------------------------------------
# 3. Boundary — minimum 1 accepted
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_keep_every_one_accepted(self) -> None:
        # Equivalent to "keep all in this window" but slower than
        # keep_all=True (still walks the decimator).
        cfg = DecimationWindowConfig(window_first_keep_every=1)
        assert cfg.window_first_keep_every == 1

    def test_minutes_one_accepted(self) -> None:
        cfg = DecimationWindowConfig(
            window_first_minutes=1, window_mid_minutes=1
        )
        assert cfg.window_first_minutes == 1
        assert cfg.window_mid_minutes == 1


# ---------------------------------------------------------------------------
# 4. Invariants — model-level guarantees
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_decimation_default_factory_is_independent(self) -> None:
        # Two instances must have independent decimation sub-models —
        # mutating one must not propagate to the other.
        a = MetricsBufferConfig()
        b = MetricsBufferConfig()
        assert a.decimation is not b.decimation

    def test_strict_model_accepts_alias_population(self) -> None:
        # populate_by_name=True (StrictBaseModel default) — keys must
        # match field names verbatim.
        cfg = MetricsBufferConfig.model_validate(
            {"keep_all": False, "decimation": {"window_first_minutes": 7}},
        )
        assert cfg.decimation.window_first_minutes == 7


# ---------------------------------------------------------------------------
# 5. Dependency errors — propagation through TrainingConfig schema
# ---------------------------------------------------------------------------


class TestTrainingConfigWiring:
    def test_training_config_has_metrics_buffer_field(self) -> None:
        # Field-level introspection avoids the heavy lift of building a
        # full TrainingConfig (lora.target_modules + 6 other required
        # knobs). What matters here is the schema: `metrics_buffer`
        # field exists, default factory returns lossless config.
        from src.config.training import TrainingConfig

        fields = TrainingConfig.model_fields
        assert "metrics_buffer" in fields

        # Default factory returns MetricsBufferConfig with keep_all=True.
        default_factory = fields["metrics_buffer"].default_factory
        assert default_factory is not None
        instance = default_factory()
        assert isinstance(instance, MetricsBufferConfig)
        assert instance.keep_all is True

    def test_decimation_inside_training_config_yaml_shape(self) -> None:
        # Pin the YAML key naming for the decimation block as it
        # appears under training.metrics_buffer. Tests the dict path
        # without instantiating the parent TrainingConfig.
        cfg = MetricsBufferConfig.model_validate(
            {
                "keep_all": False,
                "decimation": {
                    "window_first_minutes": 5,
                    "window_first_keep_every": 1,
                    "window_mid_minutes": 15,
                    "window_mid_keep_every": 3,
                    "window_late_keep_every": 10,
                },
            }
        )
        assert cfg.keep_all is False
        assert cfg.decimation.window_late_keep_every == 10


# ---------------------------------------------------------------------------
# 6. Regressions — old configs still load
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_default_factory_for_missing_block_is_lossless(self) -> None:
        # Pre-Phase-12.A.2 YAMLs have no metrics_buffer field. Pydantic
        # falls through the default factory → keep_all=True. NO data
        # loss vs Phase 9 hard-coded behaviour: keep_all=True is
        # *strictly more permissive*.
        from src.config.training import TrainingConfig

        default_factory = TrainingConfig.model_fields["metrics_buffer"].default_factory
        assert default_factory is not None
        cfg = default_factory()
        assert cfg.keep_all is True
        assert cfg.decimation.window_first_minutes == 10  # Phase 9 baseline


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_metrics_buffer_block_partial_overrides(self) -> None:
        # User can specify only keep_all and let decimation defaults
        # carry through (handy for "I want lossless, full stop").
        cfg = MetricsBufferConfig.model_validate({"keep_all": False})
        assert cfg.keep_all is False
        # Decimation still has Phase 9 baseline defaults.
        assert cfg.decimation.window_first_minutes == 10
        assert cfg.decimation.window_late_keep_every == 5

    def test_decimation_partial_override_keeps_other_defaults(self) -> None:
        # Override only window_late_keep_every — others default.
        cfg = MetricsBufferConfig.model_validate(
            {
                "keep_all": False,
                "decimation": {"window_late_keep_every": 100},
            }
        )
        assert cfg.decimation.window_late_keep_every == 100
        assert cfg.decimation.window_first_keep_every == 1
        assert cfg.decimation.window_mid_keep_every == 2
