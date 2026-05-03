"""
Unit tests for early training-pod release (terminate_after_retrieval).

Covers:
  1. GPUDeployer.release() is idempotent
  2. GPUDeployer.cleanup() is a no-op after release()
  3. Orchestrator calls release() after MODEL_RETRIEVER when flag is true
  4. Orchestrator does NOT call release() when flag is false
  5. IEarlyReleasable isinstance check works for GPUDeployer
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.pipeline.stages.constants import StageNames
from src.pipeline.stages.gpu_deployer import GPUDeployer, IEarlyReleasable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gpu_deployer() -> GPUDeployer:
    """Build a GPUDeployer instance bypassing __init__ (no config/secrets needed)."""
    deployer = object.__new__(GPUDeployer)
    deployer.stage_name = StageNames.GPU_DEPLOYER
    deployer.config = MagicMock()
    deployer.secrets = MagicMock()
    deployer._callbacks = MagicMock()
    deployer._callbacks.on_cleanup = None
    deployer._provider_name = "runpod"
    deployer._provider = MagicMock()
    deployer._ssh_client = None
    deployer._released = False
    deployer.deployment = MagicMock()
    deployer.metadata = {}
    return deployer


# ---------------------------------------------------------------------------
# 1. release() is idempotent
# ---------------------------------------------------------------------------


class TestReleaseIdempotent:
    def test_release_calls_disconnect_exactly_once(self) -> None:
        """release() called twice must call provider.disconnect() only once."""
        deployer = _make_gpu_deployer()
        mock_provider = MagicMock()
        deployer._provider = mock_provider

        deployer.release()
        deployer.release()  # second call — must be a no-op

        mock_provider.disconnect.assert_called_once()

    def test_release_sets_released_flag(self) -> None:
        """After release(), _released must be True."""
        deployer = _make_gpu_deployer()
        assert deployer._released is False
        deployer.release()
        assert deployer._released is True

    def test_release_clears_provider(self) -> None:
        """After release(), _provider must be None."""
        deployer = _make_gpu_deployer()
        deployer.release()
        assert deployer._provider is None

    def test_release_when_provider_none_is_noop(self) -> None:
        """release() with _provider=None must not raise."""
        deployer = _make_gpu_deployer()
        deployer._provider = None
        deployer.release()  # must not raise
        assert deployer._released is False  # nothing happened

    def test_release_when_already_released_is_noop(self) -> None:
        """release() with _released=True must not call disconnect."""
        deployer = _make_gpu_deployer()
        mock_provider = MagicMock()
        deployer._provider = mock_provider
        deployer._released = True

        deployer.release()

        mock_provider.disconnect.assert_not_called()


# ---------------------------------------------------------------------------
# 2. cleanup() is a no-op after release()
# ---------------------------------------------------------------------------


class TestCleanupNopAfterRelease:
    def test_cleanup_skipped_when_released(self) -> None:
        """GPUDeployer.cleanup() must not call disconnect() if _released=True."""
        deployer = _make_gpu_deployer()
        mock_provider = MagicMock()
        deployer._provider = mock_provider
        deployer._released = True

        deployer.cleanup()

        mock_provider.disconnect.assert_not_called()

    def test_cleanup_runs_normally_when_not_released(self) -> None:
        """GPUDeployer.cleanup() must call disconnect() if _released=False."""
        deployer = _make_gpu_deployer()
        mock_provider = MagicMock()
        deployer._provider = mock_provider
        deployer._released = False

        deployer.cleanup()

        mock_provider.disconnect.assert_called_once()

    def test_release_then_cleanup_disconnect_called_once_total(self) -> None:
        """release() + cleanup() together must call disconnect() exactly once."""
        deployer = _make_gpu_deployer()
        mock_provider = MagicMock()
        deployer._provider = mock_provider

        deployer.release()
        deployer.cleanup()

        mock_provider.disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Orchestrator calls release() after MODEL_RETRIEVER when flag=true
# ---------------------------------------------------------------------------


class TestOrchestratorCallsRelease:
    def _build_orchestrator_with_deployer(
        self, *, terminate_after_retrieval: bool
    ) -> tuple[object, GPUDeployer]:
        """Build a minimal orchestrator mock with a real GPUDeployer."""
        from src.pipeline.execution import StageRegistry
        from src.pipeline.orchestrator import PipelineOrchestrator

        orch = object.__new__(PipelineOrchestrator)

        deployer = _make_gpu_deployer()
        mock_provider = MagicMock()
        deployer._provider = mock_provider

        # Config mock that returns cleanup dict
        mock_cfg = MagicMock()
        mock_cfg.get_provider_config.return_value = {
            "cleanup": {"terminate_after_retrieval": terminate_after_retrieval}
        }
        orch.config = mock_cfg

        # StageRegistry owns stages + cleanup policy after PR-A10; tests
        # that bypass __init__ must wire a registry manually.
        orch._registry = StageRegistry(
            config=mock_cfg, stages=[deployer], collectors={}
        )
        orch.stages = orch._registry.stages
        orch._mlflow_manager = None

        return orch, deployer

    def test_release_called_when_flag_true(self) -> None:
        """_maybe_early_release_gpu() must call GPUDeployer.release() when flag=true."""
        orch, deployer = self._build_orchestrator_with_deployer(terminate_after_retrieval=True)
        mock_provider = deployer._provider  # capture before release() clears it

        orch._maybe_early_release_gpu()  # type: ignore[attr-defined]

        assert deployer._released is True
        mock_provider.disconnect.assert_called_once()

    def test_release_not_called_when_flag_false(self) -> None:
        """_maybe_early_release_gpu() must NOT call release() when flag=false."""
        orch, deployer = self._build_orchestrator_with_deployer(terminate_after_retrieval=False)

        orch._maybe_early_release_gpu()  # type: ignore[attr-defined]

        assert deployer._released is False
        deployer._provider.disconnect.assert_not_called()

    def test_release_not_called_when_flag_missing(self) -> None:
        """_maybe_early_release_gpu() must NOT call release() if flag not in config."""
        from src.pipeline.execution import StageRegistry
        from src.pipeline.orchestrator import PipelineOrchestrator

        orch = object.__new__(PipelineOrchestrator)
        deployer = _make_gpu_deployer()

        mock_cfg = MagicMock()
        mock_cfg.get_provider_config.return_value = {"cleanup": {}}  # no key
        orch.config = mock_cfg
        orch._registry = StageRegistry(config=mock_cfg, stages=[deployer], collectors={})
        orch.stages = orch._registry.stages

        orch._maybe_early_release_gpu()  # type: ignore[attr-defined]

        assert deployer._released is False

    def test_release_not_called_when_config_raises(self) -> None:
        """_maybe_early_release_gpu() must be silent if config access raises."""
        from src.pipeline.execution import StageRegistry
        from src.pipeline.orchestrator import PipelineOrchestrator

        orch = object.__new__(PipelineOrchestrator)
        deployer = _make_gpu_deployer()

        mock_cfg = MagicMock()
        mock_cfg.get_provider_config.side_effect = RuntimeError("config error")
        orch.config = mock_cfg
        orch._registry = StageRegistry(config=mock_cfg, stages=[deployer], collectors={})
        orch.stages = orch._registry.stages

        orch._maybe_early_release_gpu()  # must not raise

        assert deployer._released is False

    def test_only_first_releasable_stage_is_released(self) -> None:
        """If multiple IEarlyReleasable stages exist, only the first one is released."""
        from src.pipeline.execution import StageRegistry
        from src.pipeline.orchestrator import PipelineOrchestrator

        orch = object.__new__(PipelineOrchestrator)

        deployer1 = _make_gpu_deployer()
        deployer1._provider = MagicMock()
        deployer2 = _make_gpu_deployer()
        deployer2._provider = MagicMock()

        mock_cfg = MagicMock()
        mock_cfg.get_provider_config.return_value = {
            "cleanup": {"terminate_after_retrieval": True}
        }
        orch.config = mock_cfg
        orch._registry = StageRegistry(
            config=mock_cfg, stages=[deployer1, deployer2], collectors={}
        )
        orch.stages = orch._registry.stages

        orch._maybe_early_release_gpu()  # type: ignore[attr-defined]

        assert deployer1._released is True
        assert deployer2._released is False  # second one untouched


# ---------------------------------------------------------------------------
# 4. IEarlyReleasable isinstance check
# ---------------------------------------------------------------------------


class TestIEarlyReleasableProtocol:
    def test_gpu_deployer_satisfies_protocol(self) -> None:
        """GPUDeployer must satisfy IEarlyReleasable via structural subtyping."""
        deployer = _make_gpu_deployer()
        assert isinstance(deployer, IEarlyReleasable)

    def test_plain_object_does_not_satisfy_protocol(self) -> None:
        """A plain object without release() must NOT satisfy IEarlyReleasable."""

        class NoRelease:
            pass

        assert not isinstance(NoRelease(), IEarlyReleasable)

    def test_object_with_release_satisfies_protocol(self) -> None:
        """Any object with a release() method satisfies IEarlyReleasable."""

        class FakeReleasable:
            def release(self) -> None:
                pass

        assert isinstance(FakeReleasable(), IEarlyReleasable)
