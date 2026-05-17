"""Unit tests: :class:`PodLifecycleConfig` (E-СРЕД config wire-through).

The optional ``pipeline_config.yaml`` block that drives the pod-side
:class:`IdleDetector` thresholds. These tests focus on the schema
(defaults, bounds) plus the Mac→pod env-var bridging done by
``training_launcher._build_job_env``.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from ryotenkai_shared.config import PodLifecycleConfig


class TestPositive:
    def test_defaults_match_legacy_thresholds(self) -> None:
        c = PodLifecycleConfig()
        # Mirrors DEFAULT_MAX_LIFETIME (48h) / DEFAULT_IDLE_THRESHOLD (20m).
        assert c.max_lifetime_hours == 48.0
        assert c.idle_threshold_minutes == 20.0

    def test_custom_values_accepted(self) -> None:
        c = PodLifecycleConfig(
            max_lifetime_hours=12.5,
            idle_threshold_minutes=5.0,
        )
        assert c.max_lifetime_hours == 12.5
        assert c.idle_threshold_minutes == 5.0


class TestNegative:
    @pytest.mark.parametrize("bad", [0, -1, -0.5])
    def test_non_positive_max_lifetime_rejected(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            PodLifecycleConfig(max_lifetime_hours=bad)

    @pytest.mark.parametrize("bad", [0, -1, -0.5])
    def test_non_positive_idle_threshold_rejected(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            PodLifecycleConfig(idle_threshold_minutes=bad)


class TestBoundary:
    def test_max_lifetime_at_30_days_accepted(self) -> None:
        # 30 days × 24h = 720h — the documented upper bound.
        c = PodLifecycleConfig(max_lifetime_hours=720.0)
        assert c.max_lifetime_hours == 720.0

    def test_max_lifetime_above_30_days_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PodLifecycleConfig(max_lifetime_hours=720.1)

    def test_idle_threshold_at_24h_accepted(self) -> None:
        c = PodLifecycleConfig(idle_threshold_minutes=60 * 24)
        assert c.idle_threshold_minutes == 60 * 24

    def test_idle_threshold_above_24h_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PodLifecycleConfig(idle_threshold_minutes=60 * 24 + 1)


class TestInvariants:
    def test_extra_fields_forbidden(self) -> None:
        # StrictBaseModel forbids unknown keys so a YAML typo
        # surfaces at load instead of being silently dropped.
        with pytest.raises(ValidationError):
            PodLifecycleConfig(spurious_key="oops")  # type: ignore[call-arg]


class TestEnvBridging:
    """Mac→pod bridge in ``training_launcher._build_job_env``.

    The launcher only consumes ``pod_lifecycle`` when present on the
    config and only emits the matching env vars when each sub-field
    is set. The pod-side ``resolve_thresholds_from_env`` then maps
    those env vars back into the in-pod constants.
    """

    def test_env_round_trip_through_resolver(self) -> None:
        from ryotenkai_pod.runner.idle_detector import (
            ENV_IDLE_THRESHOLD_MINUTES,
            ENV_MAX_LIFETIME_HOURS,
            resolve_thresholds_from_env,
        )

        cfg = PodLifecycleConfig(
            max_lifetime_hours=24.0,
            idle_threshold_minutes=10.0,
        )
        env: dict[str, str] = {
            ENV_MAX_LIFETIME_HOURS: str(cfg.max_lifetime_hours),
            ENV_IDLE_THRESHOLD_MINUTES: str(cfg.idle_threshold_minutes),
        }
        max_s, idle_s = resolve_thresholds_from_env(env=env)
        # 24h → 86_400s; 10m → 600s.
        assert max_s == 86_400.0
        assert idle_s == 600.0

    def test_training_launcher_emits_env_vars_when_pod_lifecycle_set(
        self,
    ) -> None:
        """``_build_job_env`` injects the two env vars when the operator
        opts into a custom ``pod_lifecycle`` block."""
        from ryotenkai_control.pipeline.stages.managers.deployment.training_launcher import (
            TrainingLauncher,
        )

        # Synthesise the minimum config surface ``_build_job_env`` reads.
        # We bypass the full pipeline validator (no providers needed
        # for this slice) by constructing a lightweight stand-in.
        class _FakeIntegrations:
            mlflow = None

        class _FakeSecrets:
            hf_token = None

        class _FakeConfig:
            integrations = _FakeIntegrations()
            pod_lifecycle = PodLifecycleConfig(
                max_lifetime_hours=12.0,
                idle_threshold_minutes=5.0,
            )

        launcher = TrainingLauncher.__new__(TrainingLauncher)
        launcher.config = _FakeConfig()  # type: ignore[assignment]
        launcher.secrets = _FakeSecrets()  # type: ignore[assignment]
        launcher._workspace = "/workspace"  # type: ignore[attr-defined]

        env: dict[str, str] = launcher._build_job_env(
            context={},
            provider=None,
            extra_env_vars={},
        )
        assert env["RYOTENKAI_POD_MAX_LIFETIME_HOURS"] == "12.0"
        assert env["RYOTENKAI_POD_IDLE_THRESHOLD_MINUTES"] == "5.0"

    def test_training_launcher_omits_env_vars_when_pod_lifecycle_absent(
        self,
    ) -> None:
        """When ``pod_lifecycle`` is ``None`` (default), no env vars are
        injected — the pod falls through to its in-pod defaults."""
        from ryotenkai_control.pipeline.stages.managers.deployment.training_launcher import (
            TrainingLauncher,
        )

        class _FakeIntegrations:
            mlflow = None

        class _FakeSecrets:
            hf_token = None

        class _FakeConfig:
            integrations = _FakeIntegrations()
            pod_lifecycle = None

        launcher = TrainingLauncher.__new__(TrainingLauncher)
        launcher.config = _FakeConfig()  # type: ignore[assignment]
        launcher.secrets = _FakeSecrets()  # type: ignore[assignment]
        launcher._workspace = "/workspace"  # type: ignore[attr-defined]

        env: dict[str, Any] = launcher._build_job_env(
            context={},
            provider=None,
            extra_env_vars={},
        )
        assert "RYOTENKAI_POD_MAX_LIFETIME_HOURS" not in env
        assert "RYOTENKAI_POD_IDLE_THRESHOLD_MINUTES" not in env
