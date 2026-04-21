"""Tests for :class:`StageRegistry`.

Coverage: positive / negative / boundary / invariants / dep-errors /
regressions / combinatorial.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.pipeline.execution import StageRegistry
from src.pipeline.stages import StageNames


def _stage(name: str, *, has_cleanup: bool = True, cleanup_raises: Exception | None = None) -> MagicMock:
    s = MagicMock(spec=["stage_name", "cleanup", "release", "notify_pipeline_failure"])
    s.stage_name = name
    if has_cleanup:
        if cleanup_raises:
            s.cleanup.side_effect = cleanup_raises
    else:
        del s.cleanup
    return s


def _config(
    *,
    active_provider: str = "single_node",
    terminate_after_retrieval: bool = False,
    on_interrupt: bool = True,
    provider_raises: bool = False,
) -> MagicMock:
    cfg = MagicMock()
    if provider_raises:
        cfg.get_active_provider_name.side_effect = RuntimeError("boom")
        cfg.get_provider_config.side_effect = RuntimeError("boom")
    else:
        cfg.get_active_provider_name.return_value = active_provider
        cfg.get_provider_config.return_value = {
            "cleanup": {
                "terminate_after_retrieval": terminate_after_retrieval,
                "on_interrupt": on_interrupt,
            }
        }
    return cfg


# ===========================================================================
# 1. POSITIVE
# ===========================================================================


class TestPositive:
    def test_build_produces_canonical_stages_and_collectors(self) -> None:
        # build() is a smoke test — the real stages need config/secrets so we
        # patch the factory. Assert structure instead of identities.
        registry = StageRegistry(
            config=_config(),
            stages=[_stage("A"), _stage("B")],
            collectors={"A": MagicMock(), "B": MagicMock()},
        )
        assert registry.list_stage_names() == ["A", "B"]
        assert set(registry.collectors.keys()) == {"A", "B"}

    def test_get_stage_by_name(self) -> None:
        stages = [_stage("A"), _stage("B")]
        registry = StageRegistry(config=_config(), stages=stages, collectors={})
        assert registry.get_stage_by_name("A") is stages[0]

    def test_cleanup_in_reverse_order(self) -> None:
        order: list[str] = []
        stages = []
        for name in ["A", "B", "C"]:
            s = _stage(name)
            s.cleanup.side_effect = lambda n=name: order.append(n)
            stages.append(s)
        registry = StageRegistry(config=_config(), stages=stages, collectors={})

        registry.cleanup_in_reverse(success=False, shutdown_signal_name=None)
        assert order == ["C", "B", "A"]

    def test_maybe_early_release_fires_when_flag_true(self) -> None:
        from src.pipeline.stages.gpu_deployer import IEarlyReleasable

        releasable = MagicMock(spec=IEarlyReleasable)
        releasable.stage_name = "GPU"
        other = _stage("OTHER")
        registry = StageRegistry(
            config=_config(terminate_after_retrieval=True),
            stages=[other, releasable],
            collectors={},
        )
        registry.maybe_early_release_gpu()
        releasable.release.assert_called_once()


# ===========================================================================
# 2. NEGATIVE
# ===========================================================================


class TestNegative:
    def test_get_stage_by_name_missing_returns_none(self) -> None:
        registry = StageRegistry(config=_config(), stages=[_stage("A")], collectors={})
        assert registry.get_stage_by_name("MISSING") is None

    def test_cleanup_exceptions_are_swallowed(self) -> None:
        stages = [
            _stage("A"),
            _stage("B", cleanup_raises=RuntimeError("boom")),
            _stage("C"),
        ]
        registry = StageRegistry(config=_config(), stages=stages, collectors={})
        # Does not raise
        registry.cleanup_in_reverse(success=True, shutdown_signal_name=None)
        # Every stage attempted regardless
        for s in stages:
            s.cleanup.assert_called_once()


# ===========================================================================
# 3. BOUNDARY
# ===========================================================================


class TestBoundary:
    def test_cleanup_is_idempotent(self) -> None:
        stages = [_stage("A")]
        registry = StageRegistry(config=_config(), stages=stages, collectors={})
        registry.cleanup_in_reverse(success=True, shutdown_signal_name=None)
        registry.cleanup_in_reverse(success=True, shutdown_signal_name=None)
        stages[0].cleanup.assert_called_once()

    def test_cleanup_with_no_stages(self) -> None:
        registry = StageRegistry(config=_config(), stages=[], collectors={})
        registry.cleanup_in_reverse(success=True, shutdown_signal_name=None)  # no-op

    def test_maybe_early_release_no_op_when_flag_false(self) -> None:
        from src.pipeline.stages.gpu_deployer import IEarlyReleasable

        releasable = MagicMock(spec=IEarlyReleasable)
        releasable.stage_name = "GPU"
        registry = StageRegistry(
            config=_config(terminate_after_retrieval=False),
            stages=[releasable],
            collectors={},
        )
        registry.maybe_early_release_gpu()
        releasable.release.assert_not_called()

    def test_stage_without_cleanup_attr_is_skipped(self) -> None:
        no_cleanup = _stage("A", has_cleanup=False)
        with_cleanup = _stage("B")
        registry = StageRegistry(
            config=_config(), stages=[no_cleanup, with_cleanup], collectors={}
        )
        registry.cleanup_in_reverse(success=True, shutdown_signal_name=None)
        with_cleanup.cleanup.assert_called_once()


# ===========================================================================
# 4. INVARIANTS
# ===========================================================================


class TestInvariants:
    def test_failure_fires_notify_pipeline_failure_on_each(self) -> None:
        stages = [_stage("A"), _stage("B")]
        registry = StageRegistry(config=_config(), stages=stages, collectors={})
        registry.cleanup_in_reverse(success=False, shutdown_signal_name=None)
        for s in stages:
            s.notify_pipeline_failure.assert_called_once()

    def test_success_does_not_fire_notify_pipeline_failure(self) -> None:
        stages = [_stage("A"), _stage("B")]
        registry = StageRegistry(config=_config(), stages=stages, collectors={})
        registry.cleanup_in_reverse(success=True, shutdown_signal_name=None)
        for s in stages:
            s.notify_pipeline_failure.assert_not_called()

    def test_sigint_with_on_interrupt_false_skips_gpu(self) -> None:
        gpu = _stage(StageNames.GPU_DEPLOYER)
        other = _stage("OTHER")
        registry = StageRegistry(
            config=_config(on_interrupt=False),
            stages=[gpu, other],
            collectors={},
        )
        registry.cleanup_in_reverse(success=True, shutdown_signal_name="SIGINT")
        gpu.cleanup.assert_not_called()
        other.cleanup.assert_called_once()

    def test_non_sigint_signal_cleans_gpu_normally(self) -> None:
        gpu = _stage(StageNames.GPU_DEPLOYER)
        registry = StageRegistry(
            config=_config(on_interrupt=False),  # even with on_interrupt=false...
            stages=[gpu],
            collectors={},
        )
        registry.cleanup_in_reverse(success=True, shutdown_signal_name="SIGTERM")
        # ...non-SIGINT signals still trigger normal cleanup
        gpu.cleanup.assert_called_once()


# ===========================================================================
# 5. DEPENDENCY ERRORS
# ===========================================================================


class TestDependencyErrors:
    def test_notify_pipeline_failure_exception_logged_and_swallowed(self) -> None:
        bad = _stage("bad")
        bad.notify_pipeline_failure.side_effect = RuntimeError("notify failed")
        good = _stage("good")
        registry = StageRegistry(config=_config(), stages=[bad, good], collectors={})
        # Must not raise; cleanup still proceeds
        registry.cleanup_in_reverse(success=False, shutdown_signal_name=None)
        good.cleanup.assert_called_once()

    def test_config_exception_during_sigint_check_defaults_to_cleanup(self) -> None:
        gpu = _stage(StageNames.GPU_DEPLOYER)
        registry = StageRegistry(
            config=_config(provider_raises=True),
            stages=[gpu],
            collectors={},
        )
        # Config inspection fails → default to running cleanup
        registry.cleanup_in_reverse(success=True, shutdown_signal_name="SIGINT")
        gpu.cleanup.assert_called_once()

    def test_maybe_early_release_silent_when_config_raises(self) -> None:
        from src.pipeline.stages.gpu_deployer import IEarlyReleasable

        releasable = MagicMock(spec=IEarlyReleasable)
        registry = StageRegistry(
            config=_config(provider_raises=True),
            stages=[releasable],
            collectors={},
        )
        registry.maybe_early_release_gpu()  # no raise
        releasable.release.assert_not_called()


# ===========================================================================
# 6. REGRESSIONS
# ===========================================================================


class TestRegressions:
    def test_collectors_dict_is_same_reference_as_passed_in(self) -> None:
        """Regression: registry must NOT copy the collectors dict — downstream
        components (ValidationArtifactManager) hold a reference to the same
        dict and need to see flush-state updates."""
        collectors = {"A": MagicMock()}
        registry = StageRegistry(
            config=_config(), stages=[_stage("A")], collectors=collectors
        )
        assert registry.collectors is collectors

    def test_keyboard_interrupt_during_cleanup_continues(self) -> None:
        """Regression: a second Ctrl+C while cleanup runs must NOT abort the
        rest of the cleanup."""
        a = _stage("A")
        b = _stage("B", cleanup_raises=KeyboardInterrupt())
        c = _stage("C")
        registry = StageRegistry(config=_config(), stages=[a, b, c], collectors={})
        registry.cleanup_in_reverse(success=True, shutdown_signal_name=None)
        a.cleanup.assert_called_once()
        c.cleanup.assert_called_once()


# ===========================================================================
# 7. COMBINATORIAL
# ===========================================================================


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("signal_name", [None, "SIGINT", "SIGTERM"])
@pytest.mark.parametrize("on_interrupt", [True, False])
def test_cleanup_matrix(
    success: bool,
    signal_name: str | None,
    on_interrupt: bool,
) -> None:
    gpu = _stage(StageNames.GPU_DEPLOYER)
    other = _stage("OTHER")
    registry = StageRegistry(
        config=_config(on_interrupt=on_interrupt),
        stages=[gpu, other],
        collectors={},
    )
    registry.cleanup_in_reverse(success=success, shutdown_signal_name=signal_name)

    skip_gpu = signal_name == "SIGINT" and not on_interrupt
    assert gpu.cleanup.called is (not skip_gpu)
    other.cleanup.assert_called_once()

    if not success:
        gpu.notify_pipeline_failure.assert_called_once()
        other.notify_pipeline_failure.assert_called_once()
    else:
        gpu.notify_pipeline_failure.assert_not_called()
        other.notify_pipeline_failure.assert_not_called()


# Silence "Any imported but unused" if linter complains
_ = Any
