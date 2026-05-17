"""
Isolated unit tests for MLflowDomainLogger.
No real MLflow SDK calls — primitives are mocked.

Phase 7: ``MLflowEventLog`` was retired; the domain logger now writes
only params/metrics/tags/dicts through the primitives. The legacy
event-log fanout is gone, so most domain helpers are no-ops by design
and only callability/no-crash is asserted.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ryotenkai_pod.trainer.mlflow.domain_logger import MLflowDomainLogger


def _make_domain_logger() -> tuple[MLflowDomainLogger, MagicMock]:
    primitives = MagicMock()
    domain_logger = MLflowDomainLogger(primitives)
    return domain_logger, primitives


class TestMLflowDomainLoggerProviderInfo:
    def test_log_provider_info_sets_tags(self):
        dl, primitives = _make_domain_logger()
        dl.log_provider_info("runpod", "cloud", gpu_type="A100", resource_id="pod-123")
        primitives.set_tags.assert_called_once()
        tags = primitives.set_tags.call_args[0][0]
        assert tags["provider.name"] == "runpod"
        assert tags["provider.type"] == "cloud"
        assert tags["provider.gpu_type"] == "A100"
        assert tags["provider.resource_id"] == "pod-123"

    def test_log_provider_info_minimal(self):
        dl, primitives = _make_domain_logger()
        dl.log_provider_info("local", "single_node")
        tags = primitives.set_tags.call_args[0][0]
        assert "provider.gpu_type" not in tags
        assert "provider.resource_id" not in tags

    def test_log_strategy_info(self):
        dl, primitives = _make_domain_logger()
        dl.log_strategy_info("sft", phase_idx=0, total_phases=3)
        primitives.set_tags.assert_called_once()
        tags = primitives.set_tags.call_args[0][0]
        assert tags["current_strategy"] == "sft"
        assert tags["current_phase"] == "0"
        assert tags["total_phases"] == "3"


class TestMLflowDomainLoggerMetrics:
    def test_log_gpu_metrics_basic(self):
        dl, primitives = _make_domain_logger()
        dl.log_gpu_metrics(40.0, 80.0)
        primitives.log_metrics.assert_called_once()
        metrics = primitives.log_metrics.call_args[0][0]
        assert metrics["gpu_memory_used_gb"] == 40.0
        assert metrics["gpu_memory_total_gb"] == 80.0
        assert metrics["gpu_memory_pct"] == 50.0

    def test_log_gpu_metrics_with_utilization(self):
        dl, primitives = _make_domain_logger()
        dl.log_gpu_metrics(40.0, 80.0, gpu_utilization=85.0, step=100)
        metrics = primitives.log_metrics.call_args[0][0]
        assert metrics["gpu_utilization"] == 85.0

    def test_log_gpu_metrics_zero_total_no_div_by_zero(self):
        dl, primitives = _make_domain_logger()
        dl.log_gpu_metrics(0.0, 0.0)
        metrics = primitives.log_metrics.call_args[0][0]
        assert metrics["gpu_memory_pct"] == 0

    def test_log_throughput(self):
        dl, primitives = _make_domain_logger()
        dl.log_throughput(1000.0, 32.0, step=50)
        primitives.log_metrics.assert_called_once()
        metrics = primitives.log_metrics.call_args[0][0]
        assert metrics["tokens_per_second"] == 1000.0
        assert metrics["samples_per_second"] == 32.0


class TestMLflowDomainLoggerMemoryEvents:
    """Phase 6.b: MLflowEventLog is now a no-op shim, so the legacy
    ``log_<domain>`` methods that fanned out to ``log_event`` no longer
    record anything. We retain the calls (they still log params/tags)
    and assert only on the side effects that survived.
    """

    def test_log_gpu_detection_records_params(self):
        dl, primitives = _make_domain_logger()
        dl.log_gpu_detection("A100", 80.0, "high")

        primitives.log_params.assert_called_once()
        params = primitives.log_params.call_args[0][0]
        assert params["gpu_name"] == "A100"
        assert params["gpu_vram_gb"] == 80.0
        assert params["gpu_tier"] == "high"

    def test_log_memory_warning_is_callable(self):
        dl, _ = _make_domain_logger()
        dl.log_memory_warning(85.0, 6800, 8000, is_critical=False)

    def test_log_memory_critical_is_callable(self):
        dl, _ = _make_domain_logger()
        dl.log_memory_warning(95.0, 7600, 8000, is_critical=True)

    def test_log_oom_is_callable(self):
        dl, _ = _make_domain_logger()
        dl.log_oom("forward_pass", free_mb=512)

    def test_log_oom_without_free_mb(self):
        dl, _ = _make_domain_logger()
        dl.log_oom("backward_pass")

    def test_log_oom_recovery_is_callable(self):
        dl, _ = _make_domain_logger()
        dl.log_oom_recovery("forward_pass", attempt=2, max_attempts=3)

    def test_log_cache_cleared_when_freed(self):
        dl, _ = _make_domain_logger()
        dl.log_cache_cleared(freed_mb=200)

    def test_log_cache_cleared_skips_zero_freed(self):
        dl, _ = _make_domain_logger()
        dl.log_cache_cleared(freed_mb=0)


class TestMLflowDomainLoggerPipelineEvents:
    """Phase 6.b: pipeline ``log_<stage>`` helpers still call into the
    event-log but it's now a no-op shim. We assert that the methods
    are callable; behavioral coverage moved to the typed-journal
    tests in ``tests/unit/control/pipeline/events`` / shared events.
    """

    def test_log_stage_start_is_callable(self):
        dl, _ = _make_domain_logger()
        dl.log_stage_start("Training", stage_idx=0, total_stages=3)

    def test_log_stage_complete_is_callable(self):
        dl, _ = _make_domain_logger()
        dl.log_stage_complete("Training", stage_idx=0, duration_seconds=120.0)

    def test_log_stage_failed_is_callable(self):
        dl, _ = _make_domain_logger()
        dl.log_stage_failed("Training", stage_idx=1, error="OOM error")

    def test_log_pipeline_initialized_is_callable(self):
        dl, _ = _make_domain_logger()
        dl.log_pipeline_initialized("run-abc", total_phases=2, strategy_chain=["sft", "dpo"])

    def test_log_state_saved_is_callable(self):
        dl, _ = _make_domain_logger()
        dl.log_state_saved("run-abc", "/tmp/state.json")

    def test_log_checkpoint_cleanup_is_callable(self):
        dl, _ = _make_domain_logger()
        dl.log_checkpoint_cleanup(cleaned_count=3, freed_mb=1500)

    def test_log_checkpoint_cleanup_skips_zero_count(self):
        dl, _ = _make_domain_logger()
        dl.log_checkpoint_cleanup(cleaned_count=0, freed_mb=0)


class TestMLflowDomainLoggerEnvironment:
    def test_log_environment_with_snapshot(self):
        dl, primitives = _make_domain_logger()
        env = {"python_version": "3.11.0", "platform": "Linux"}
        dl.log_environment(env_snapshot=env)
        primitives.log_params.assert_called_once()
        params = primitives.log_params.call_args[0][0]
        assert "env.python_version" in params
        assert params["env.python_version"] == "3.11.0"

    def test_log_environment_filters_none_values(self):
        dl, primitives = _make_domain_logger()
        dl.log_environment(env_snapshot={"key": "val", "null_key": None})
        params = primitives.log_params.call_args[0][0]
        assert "env.null_key" not in params
        assert "env.key" in params
