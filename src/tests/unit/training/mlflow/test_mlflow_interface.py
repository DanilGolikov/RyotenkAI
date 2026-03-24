"""
Unit tests for src/training/mlflow/__init__.py — IMLflowManager Protocol.

Covers:
- Protocol is @runtime_checkable → isinstance() checks work
- All Protocol method stubs are executable (covering the `...` lines)
- Module-level exports (__all__) are importable
- Each subcomponent class is re-exported
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.training.mlflow import (
    IMLflowManager,
    IMLflowPrimitives,
    MLflowAutologManager,
    MLflowDatasetLogger,
    MLflowDomainLogger,
    MLflowEventLog,
    MLflowModelRegistry,
    MLflowRunAnalytics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ConcreteManager:
    """Minimal duck-type implementation that satisfies IMLflowManager."""

    @property
    def is_enabled(self) -> bool:
        return True

    @property
    def is_active(self) -> bool:
        return True

    @property
    def run_id(self) -> str | None:
        return "run-1"

    @property
    def parent_run_id(self) -> str | None:
        return None

    def setup(self, timeout=5.0, max_retries=3, disable_system_metrics=False) -> bool:
        return True

    def start_run(self, run_name=None, description=None):
        return MagicMock()

    def start_nested_run(self, run_name="", tags=None):
        return MagicMock()

    def log_params(self, params) -> None:
        pass

    def log_metrics(self, metrics, step=None) -> None:
        pass

    def set_tags(self, tags) -> None:
        pass

    def log_dataset_info(self, name, path=None, source=None, version=None,
                         num_rows=0, num_samples=None, num_features=None,
                         context="training", extra_info=None, extra_tags=None) -> None:
        pass

    def create_mlflow_dataset(self, data, name, source, targets=None):
        return MagicMock()

    def log_dataset_input(self, dataset, context="training") -> bool:
        return True

    def log_event(self, event_type, message, *, category="info", source="system", **kwargs):
        return {}

    def log_event_start(self, message, **kwargs):
        return {}

    def log_event_complete(self, message, **kwargs):
        return {}

    def log_event_error(self, message, **kwargs):
        return {}

    def log_event_warning(self, message, **kwargs):
        return {}

    def log_event_info(self, message, **kwargs):
        return {}

    def log_event_checkpoint(self, message, **kwargs):
        return {}

    def log_pipeline_initialized(self, run_id, total_phases, strategy_chain) -> None:
        pass

    def log_state_saved(self, run_id, path) -> None:
        pass

    def log_checkpoint_cleanup(self, cleaned_count, freed_mb) -> None:
        pass

    def log_gpu_detection(self, name, vram_gb, tier) -> None:
        pass

    def log_cache_cleared(self, freed_mb) -> None:
        pass

    def log_memory_warning(self, utilization_percent, used_mb, total_mb, is_critical) -> None:
        pass

    def log_oom(self, operation, free_mb) -> None:
        pass

    def log_oom_recovery(self, operation, attempt, max_attempts) -> None:
        pass


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    def test_all_symbols_importable(self):
        import src.training.mlflow as mlflow_pkg
        for sym in mlflow_pkg.__all__:
            assert hasattr(mlflow_pkg, sym), f"Missing export: {sym}"

    def test_subcomponent_classes_are_classes(self):
        for cls in (
            MLflowAutologManager,
            MLflowDatasetLogger,
            MLflowDomainLogger,
            MLflowEventLog,
            MLflowModelRegistry,
            MLflowRunAnalytics,
        ):
            assert isinstance(cls, type)

    def test_imlflow_primitives_exported(self):
        assert IMLflowPrimitives is not None


# ---------------------------------------------------------------------------
# @runtime_checkable behaviour
# ---------------------------------------------------------------------------

class TestIMLflowManagerProtocol:
    def test_isinstance_passes_for_concrete_impl(self):
        obj = _ConcreteManager()
        assert isinstance(obj, IMLflowManager)

    def test_isinstance_fails_for_empty_object(self):
        class Empty:
            pass
        assert not isinstance(Empty(), IMLflowManager)

    def test_protocol_is_runtime_checkable(self):
        # Should not raise TypeError when used in isinstance
        try:
            isinstance(object(), IMLflowManager)
        except TypeError:
            pytest.fail("IMLflowManager is not @runtime_checkable")


# ---------------------------------------------------------------------------
# Protocol stub methods (execute `...` bodies to achieve line coverage)
# ---------------------------------------------------------------------------

class TestProtocolStubMethods:
    """
    Directly invoke Protocol stubs to cover the `...` lines.
    Protocol methods are callable on the class itself with an explicit self.
    """

    def _dummy(self):
        return MagicMock()

    def test_is_enabled_stub(self):
        result = IMLflowManager.is_enabled.fget(self._dummy())
        assert result is None  # `...` evaluates to None as function return

    def test_is_active_stub(self):
        result = IMLflowManager.is_active.fget(self._dummy())
        assert result is None

    def test_run_id_stub(self):
        result = IMLflowManager.run_id.fget(self._dummy())
        assert result is None

    def test_parent_run_id_stub(self):
        result = IMLflowManager.parent_run_id.fget(self._dummy())
        assert result is None

    def test_setup_stub(self):
        dummy = self._dummy()
        result = IMLflowManager.setup(dummy)
        assert result is None

    def test_start_run_stub(self):
        result = IMLflowManager.start_run(self._dummy())
        assert result is None

    def test_start_nested_run_stub(self):
        result = IMLflowManager.start_nested_run(self._dummy(), "nested")
        assert result is None

    def test_log_params_stub(self):
        result = IMLflowManager.log_params(self._dummy(), {})
        assert result is None

    def test_log_metrics_stub(self):
        result = IMLflowManager.log_metrics(self._dummy(), {})
        assert result is None

    def test_set_tags_stub(self):
        result = IMLflowManager.set_tags(self._dummy(), {})
        assert result is None

    def test_log_dataset_info_stub(self):
        result = IMLflowManager.log_dataset_info(self._dummy(), "ds")
        assert result is None

    def test_create_mlflow_dataset_stub(self):
        result = IMLflowManager.create_mlflow_dataset(self._dummy(), [], "ds", "src")
        assert result is None

    def test_log_dataset_input_stub(self):
        result = IMLflowManager.log_dataset_input(self._dummy(), MagicMock())
        assert result is None

    def test_log_event_stub(self):
        result = IMLflowManager.log_event(self._dummy(), "start", "msg")
        assert result is None

    def test_log_event_start_stub(self):
        result = IMLflowManager.log_event_start(self._dummy(), "msg")
        assert result is None

    def test_log_event_complete_stub(self):
        result = IMLflowManager.log_event_complete(self._dummy(), "done")
        assert result is None

    def test_log_event_error_stub(self):
        result = IMLflowManager.log_event_error(self._dummy(), "err")
        assert result is None

    def test_log_event_warning_stub(self):
        result = IMLflowManager.log_event_warning(self._dummy(), "warn")
        assert result is None

    def test_log_event_info_stub(self):
        result = IMLflowManager.log_event_info(self._dummy(), "info")
        assert result is None

    def test_log_event_checkpoint_stub(self):
        result = IMLflowManager.log_event_checkpoint(self._dummy(), "ckpt")
        assert result is None

    def test_log_pipeline_initialized_stub(self):
        result = IMLflowManager.log_pipeline_initialized(self._dummy(), "r1", 3, ["a", "b"])
        assert result is None

    def test_log_state_saved_stub(self):
        result = IMLflowManager.log_state_saved(self._dummy(), "r1", "/path")
        assert result is None

    def test_log_checkpoint_cleanup_stub(self):
        result = IMLflowManager.log_checkpoint_cleanup(self._dummy(), 5, 200)
        assert result is None

    def test_log_gpu_detection_stub(self):
        result = IMLflowManager.log_gpu_detection(self._dummy(), "A100", 40.0, "high")
        assert result is None

    def test_log_cache_cleared_stub(self):
        result = IMLflowManager.log_cache_cleared(self._dummy(), 512)
        assert result is None

    def test_log_memory_warning_stub(self):
        result = IMLflowManager.log_memory_warning(self._dummy(), 95.0, 15000, 16000, True)
        assert result is None

    def test_log_oom_stub(self):
        result = IMLflowManager.log_oom(self._dummy(), "train_step", 128)
        assert result is None

    def test_log_oom_recovery_stub(self):
        result = IMLflowManager.log_oom_recovery(self._dummy(), "train_step", 1, 3)
        assert result is None
